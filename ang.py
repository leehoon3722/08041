import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
import bluetooth  # serial 대신 bluetooth 임포트
import time
import threading
import sys

# --- 0. OpenCV GStreamer 지원 여부 확인 ---
def check_opencv_gstreamer():
    build_info = cv2.getBuildInformation()
    if "GStreamer" in build_info and "YES" in build_info.split("GStreamer")[1].split("\n")[0]:
        pass 
    else:
        print("\n🚨 [치명적 오류] OpenCV가 GStreamer를 지원하지 않습니다!")
        sys.exit(1)

# --- 1. CSI 카메라 파이프라인 ---
def gstreamer_pipeline(
    sensor_id=0, capture_width=1280, capture_height=720,
    display_width=640, display_height=480, framerate=30, flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

class CSICameraStream:
    def __init__(self):
        pipeline = gstreamer_pipeline(flip_method=0)
        self.stream = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.stream.isOpened():
            print("\n❌ 에러: 카메라를 열 수 없습니다.")
            sys.exit(1) 
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream.isOpened():
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

# --- 2. 블루투스 통신 스레드 ---
ESP32_MAC_ADDR = "F4:2D:C9:89:B1:A6" 

sock = None
try:
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.settimeout(0.2) # 수신 대기 시 무한루프(멈춤) 방지용 타임아웃
    sock.connect((ESP32_MAC_ADDR, 1))
    print(f"✅ 블루투스 연결 성공: {ESP32_MAC_ADDR}")
except Exception as e:
    sock = None
    print(f"🚨 블루투스 연결 실패 (시뮬레이션 모드로 작동): {e}")

target_state = "OFF"
is_running = True
lock = threading.Lock()

def bluetooth_thread():
    global target_state, is_running
    current_state = "OFF"
    last_heartbeat_time = time.time()

    def send_command(cmd_text):
        """명령어 전송 및 ESP32의 'A' 응답(ACK) 대기"""
        full_packet = f"!{cmd_text}#\n"
        retry_count = 0
        
        while is_running and retry_count < 10:
            if not sock: break
            try:
                sock.send(full_packet)
                print(f">> 상태 전송: {full_packet.strip()}")
                
                time.sleep(0.1) 
                # ESP32로부터 'A' 응답을 받았는지 확인 (Handshaking)
                try:
                    res = sock.recv(1024).decode('utf-8', errors='ignore').strip()
                    if "A" in res:
                        print(f"✅ 수신 확인 완료 (ACK)")
                        break
                except bluetooth.btcommon.BluetoothError:
                    pass # 타임아웃 시 응답이 없는 것으로 간주하고 재전송
                
            except Exception as e:
                print(f"통신 에러: {e}")
            
            retry_count += 1
            time.sleep(0.1)

    while is_running:
        with lock:
            cmd_to_send = target_state
        
        # 1. 상태가 변했을 때 확실하게 전송
        if cmd_to_send != current_state:
            send_command(cmd_to_send)
            current_state = cmd_to_send
            last_heartbeat_time = time.time()
            
        # 2. 하트비트 전송 (!H#\n)
        curr_time = time.time()
        if curr_time - last_heartbeat_time >= 1.0:
            if sock and current_state == cmd_to_send:
                try:
                    sock.send("!H#\n")
                    time.sleep(0.05)
                    # 하트비트 응답 버퍼 비우기
                    try: sock.recv(1024) 
                    except: pass
                except: pass
            last_heartbeat_time = curr_time
            
        time.sleep(0.1)

# --- 3. 유틸리티 (눈 크기 계산) ---
def calculate_ear(eye_pts):
    v1 = dist.euclidean(eye_pts[1], eye_pts[5])
    v2 = dist.euclidean(eye_pts[2], eye_pts[4])
    h = dist.euclidean(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

# --- 4. 메인 카메라 루프 ---
def main():
    global target_state, is_running
    check_opencv_gstreamer()
    
    vs = CSICameraStream().start()
    time.sleep(2.0) 
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5, # 뒤 사람 포함 다중 탐지
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    prev_time = time.time()
    calibration_data = []
    ear_threshold = 0
    is_calibrated = False
    calib_duration = 5.0
    calib_start_time = 0
    
    eye_closed_start_time = 0 
    face_missing_start_time = 0

    try:
        while True:
            frame = vs.read()
            if frame is None: 
                time.sleep(0.1)
                continue

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
            prev_time = curr_time

            small_frame = cv2.resize(frame, (320, 240))
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_small)
            
            h, w, _ = frame.shape
            avg_ear = 0

            if results.multi_face_landmarks:
                face_missing_start_time = 0 

                # 가장 크기가 큰 얼굴(맨 앞사람) 찾기
                largest_face = None
                max_area = 0

                for face_lms in results.multi_face_landmarks:
                    xs = [lm.x for lm in face_lms.landmark]
                    ys = [lm.y for lm in face_lms.landmark]
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    if area > max_area:
                        max_area = area
                        largest_face = face_lms

                if largest_face:
                    lms = largest_face.landmark
                    left_pts = np.array([(lms[i].x * w, lms[i].y * h) for i in LEFT_EYE])
                    right_pts = np.array([(lms[i].x * w, lms[i].y * h) for i in RIGHT_EYE])
                    avg_ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                    # 캘리브레이션
                    if not is_calibrated:
                        if calib_start_time == 0:
                            calib_start_time = time.time()
                        elapsed = time.time() - calib_start_time
                        if elapsed < calib_duration:
                            calibration_data.append(avg_ear)
                            cv2.putText(frame, f"CALIBRATING... {int(calib_duration - elapsed)}s", (30, 80), 1, 2, (0, 255, 255), 2)
                        else:
                            if len(calibration_data) > 0:
                                ear_threshold = (sum(calibration_data) / len(calibration_data)) * 0.75
                                is_calibrated = True
                    # 졸음 판별 로직
                    else:
                        if avg_ear < ear_threshold:
                            if eye_closed_start_time == 0:
                                eye_closed_start_time = time.time()
                            
                            closed_duration = time.time() - eye_closed_start_time
                            
                            if closed_duration >= 2.0:
                                with lock: target_state = "LV2_DANGER"
                            elif closed_duration >= 0.5:
                                with lock: target_state = "LV1_WARN"
                            else:
                                with lock: target_state = "OFF"
                        else: 
                            eye_closed_start_time = 0
                            with lock: target_state = "OFF"

                        # 상태 표시
                        color_map = {"OFF": (0, 255, 0), "LV1_WARN": (0, 165, 255), "LV2_DANGER": (0, 0, 255)}
                        cv2.putText(frame, target_state, (30, 90), 1, 3, color_map.get(target_state, (255, 255, 255)), 3)

            else:
                eye_closed_start_time = 0 
                if is_calibrated:
                    if face_missing_start_time == 0:
                        face_missing_start_time = time.time()
                    missing_duration = time.time() - face_missing_start_time
                    
                    if missing_duration >= 5.0:
                        with lock: target_state = "LV2_DANGER"
                        cv2.putText(frame, "NO FACE - DANGER!", (30, 90), 1, 3, (0, 0, 255), 3)
                    else:
                        cv2.putText(frame, f"FACE MISSING... {int(5.0 - missing_duration)}s", (30, 90), 1, 2, (0, 165, 255), 2)
                else:
                    cv2.putText(frame, "WAITING FOR FACE...", (30, 90), 1, 2, (0, 255, 255), 2)

            cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40), 1, 1.5, (255, 0, 0), 2)
            cv2.imshow("Jetson Drowsiness System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    finally:
        is_running = False
        vs.stop()
        if sock:
            try: sock.send("!OFF#\n")
            except: pass
            sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("--- 젯슨 나노 졸음 감지 시스템 (블루투스 모드) 시작 ---")
    t_bt = threading.Thread(target=bluetooth_thread)
    t_bt.start()
    main()
