import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
import serial
import time
import threading

# --- 1. 카메라 캡처 스레드 클래스 (병목 방지) ---
class WebcamStream:
    def __init__(self, src=0):
        # 젯슨 나노용 V4L2 백엔드 사용
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- 2. UART 통신 설정 및 스레드 ---
try:
    # 젯슨 나노 전용 포트 /dev/ttyTHS1
    ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=0.1)
except:
    print("[WARNING] UART Port Error. Running without ESP32 connection.")
    ser = None

current_stage = 0  # 0: Normal, 1: Warning, 2: Danger
is_running = True
lock = threading.Lock()

def uart_thread():
    global current_stage, is_running
    last_sent_stage = -1
    while is_running:
        if ser and current_stage != last_sent_stage:
            cmd_map = {0: "!OFF#", 1: "!LV1_WARN#", 2: "!LV2_DANGER#"}
            cmd = cmd_map.get(current_stage, "!OFF#")
            with lock:
                try:
                    ser.write(cmd.encode())
                    print(f">> SEND: {cmd}")
                except Exception as e:
                    print(f">> SEND ERROR: {e}")
            last_sent_stage = current_stage
        time.sleep(0.1)

# --- 3. 유틸리티 함수 ---
def apply_fast_hist_eq(image):
    # 밝기(Y) 채널만 빠르게 평활화
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def calculate_ear(eye_pts):
    v1 = dist.euclidean(eye_pts[1], eye_pts[5])
    v2 = dist.euclidean(eye_pts[2], eye_pts[4])
    h = dist.euclidean(eye_pts[0], eye_pts[3])
    # [수정] 0으로 나누기 방지 안전장치 추가
    if h == 0.0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

# --- 4. 메인 실행부 ---
def main():
    global current_stage, is_running
    
    print("--- Starting Jetson Nano High-FPS Drowsiness System ---")
    vs = WebcamStream(src=0).start()
    time.sleep(1.0) # 카메라 예열 대기
    
    # refine_landmarks=False 로 설정하여 연산량 최소화
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    prev_time = time.time()
    calibration_data = []
    ear_threshold = 0
    is_calibrated = False
    calib_duration = 5
    start_time = time.time()
    counter = 0

    try:
        while True:
            frame = vs.read()
            if frame is None: continue

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
            prev_time = curr_time

            # AI 분석용 이미지 축소 (320x240)
            small_frame = cv2.resize(frame, (320, 240))
            small_frame = apply_fast_hist_eq(small_frame)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(rgb_small)
            
            h, w, _ = frame.shape
            avg_ear = 0

            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    lms = face_lms.landmark
                    left_pts = np.array([(lms[i].x * w, lms[i].y * h) for i in LEFT_EYE])
                    right_pts = np.array([(lms[i].x * w, lms[i].y * h) for i in RIGHT_EYE])
                    avg_ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                    # 1. 보정 단계
                    if not is_calibrated:
                        elapsed = curr_time - start_time
                        if elapsed < calib_duration:
                            calibration_data.append(avg_ear)
                            remain = int(calib_duration - elapsed)
                            cv2.putText(frame, f"CALIBRATING: {remain}s", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
                        else:
                            # [핵심 수정 부분] 데이터가 1개 이상 모였을 때만 기준값 계산
                            if len(calibration_data) > 0:
                                ear_threshold = (sum(calibration_data) / len(calibration_data)) * 0.75
                                is_calibrated = True
                                print(f"Calibration Done. THR: {ear_threshold:.3f}")
                            else:
                                # 얼굴 인식이 늦어져 데이터가 없다면 타이머 초기화 후 재시작
                                start_time = time.time()
                                cv2.putText(frame, "RESTARTING CALIBRATION...", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    
                    # 2. 감지 단계
                    else:
                        if avg_ear < ear_threshold:
                            counter += 1
                            if counter >= 60: current_stage = 2
                            elif counter >= 20: current_stage = 1
                        else:
                            counter = 0
                            current_stage = 0

                        # 상태 텍스트 출력
                        status_list = [("NORMAL", (0, 255, 0)), ("WARNING", (0, 165, 255)), ("DANGER", (0, 0, 255))]
                        text, color = status_list[current_stage]
                        cv2.putText(frame, text, (30, 90), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

            # 화면 정보 출력 (FPS, EAR)
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            if is_calibrated:
                cv2.putText(frame, f"EAR: {avg_ear:.2f} / THR: {ear_threshold:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Jetson Fast Drowsiness System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        is_running = False
        vs.stop()
        t_uart.join()
        if ser:
            with lock:
                ser.write("!OFF#".encode())
                ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    t_uart = threading.Thread(target=uart_thread)
    t_uart.start()
    main()