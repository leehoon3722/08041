import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
import serial
import time
import threading

# --- 1. CSI 카메라 파이프라인 생성 함수 ---
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# --- 2. CSI 카메라 캡처 스레드 ---
class CSICameraStream:
    def __init__(self):
        # 파이프라인 적용 (화면이 뒤집혀 나오면 flip_method=2 로 변경)
        pipeline = gstreamer_pipeline(flip_method=0)
        self.stream = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self.stream.isOpened():
            print("❌ 에러: CSI 카메라를 열 수 없습니다.")
            print("터미널에서 'sudo systemctl restart nvargus-daemon'을 입력해 카메라 데몬을 재시작해보세요.")
            
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
        self.stream.release()

# --- 3. UART 통신 설정 ---
try:
    ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=0.1)
except:
    ser = None

current_stage = 0 
is_running = True
lock = threading.Lock()

def uart_thread():
    global current_stage, is_running
    last_sent_stage = -1
    last_heartbeat_time = time.time()
    
    while is_running:
        if ser:
            if current_stage != last_sent_stage:
                cmd_map = {0: "!OFF#", 1: "!LV1_WARN#", 2: "!LV2_DANGER#"}
                cmd = cmd_map.get(current_stage, "!OFF#")
                with lock:
                    try:
                        ser.write(cmd.encode())
                        print(f">> SEND STATUS: {cmd}")
                    except: pass
                last_sent_stage = current_stage
            
            curr_time = time.time()
            if curr_time - last_heartbeat_time >= 1.0:
                with lock:
                    try:
                        ser.write(b'H')
                    except: pass
                last_heartbeat_time = curr_time
            
            try:
                if ser.in_waiting > 0:
                    response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    if 'A' in response:
                        pass # 통신 정상 출력은 로그가 너무 길어지므로 생략가능 (필요시 활성화)
                        # print("✅ Received: A (통신 정상!)") 
            except: 
                pass

        time.sleep(0.1)

# --- 4. 유틸리티 ---
def calculate_ear(eye_pts):
    v1 = dist.euclidean(eye_pts[1], eye_pts[5])
    v2 = dist.euclidean(eye_pts[2], eye_pts[4])
    h = dist.euclidean(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

# --- 5. 메인 실행부 ---
def main():
    global current_stage, is_running
    
    vs = CSICameraStream().start()
    time.sleep(2.0) 
    
    # [수정됨] refine_landmarks 옵션 완벽 제거 (젯슨 호환성)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
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
    calibration_started = False
    calib_start_time = 0
    
    eye_closed_start_time = 0 
    face_missing_start_time = 0

    try:
        while True:
            frame = vs.read()
            if frame is None: continue

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

                for face_lms in results.multi_face_landmarks:
                    lms = face_lms.landmark
                    left_pts = np.array([(lms[i].x * w, lms[i].y * h) for i in LEFT_EYE])
                    right_pts = np.array([(lms[i].x * w, lms[i].y * h) for i in RIGHT_EYE])
                    avg_ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                    if not is_calibrated:
                        if not calibration_started:
                            calibration_started = True
                            calib_start_time = time.time()
                            
                        elapsed = time.time() - calib_start_time
                        if elapsed < calib_duration:
                            calibration_data.append(avg_ear)
                            cv2.putText(frame, f"CALIBRATING... {int(calib_duration - elapsed)}s", (30, 80), 1, 2, (0, 255, 255), 2)
                        else:
                            if len(calibration_data) > 0:
                                ear_threshold = (sum(calibration_data) / len(calibration_data)) * 0.75
                                is_calibrated = True
                    else:
                        if avg_ear < ear_threshold:
                            if eye_closed_start_time == 0:
                                eye_closed_start_time = time.time()
                            
                            closed_duration = time.time() - eye_closed_start_time
                            
                            if closed_duration >= 10.0:      
                                current_stage = 2
                            elif closed_duration >= 2.0:     
                                current_stage = 1
                            else:
                                current_stage = 0            
                        else:
                            eye_closed_start_time = 0
                            current_stage = 0

                        status_list = [("NORMAL", (0, 255, 0)), ("WARNING", (0, 165, 255)), ("DANGER", (0, 0, 255))]
                        text, color = status_list[current_stage]
                        cv2.putText(frame, text, (30, 90), 1, 3, color, 3)

            else:
                eye_closed_start_time = 0 
                
                if is_calibrated:
                    if face_missing_start_time == 0:
                        face_missing_start_time = time.time()
                        
                    missing_duration = time.time() - face_missing_start_time
                    
                    if missing_duration >= 5.0:
                        current_stage = 2 
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
        if ser:
            ser.write("!OFF#".encode())
            ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("--- 젯슨 나노 CSI IR 카메라 모드 시작 ---")
    t_uart = threading.Thread(target=uart_thread)
    t_uart.start()
    main()
