import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
import serial
import time
import threading

# --- 1. 카메라 캡처 스레드 ---
class WebcamStream:
    def __init__(self, src=0):
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

# --- 2. UART 통신 설정 ---
try:
    ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=1)
except:
    ser = None

current_stage = 0 
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
                except: pass
            last_sent_stage = current_stage
        time.sleep(0.1)

# --- 3. 유틸리티 ---
def calculate_ear(eye_pts):
    v1 = dist.euclidean(eye_pts[1], eye_pts[5])
    v2 = dist.euclidean(eye_pts[2], eye_pts[4])
    h = dist.euclidean(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

# --- 4. 메인 실행부 ---
def main():
    global current_stage, is_running
    
    vs = WebcamStream(src=0).start()
    time.sleep(1.0)
    
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
    calib_duration = 5
    start_time = time.time()
    
    # 시간 측정을 위한 변수 추가
    eye_closed_start_time = 0 

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
                for face_lms in results.multi_face_landmarks:
                    lms = face_lms.landmark
                    left_pts = np.array([(lms[i].x * w, lms[i].y * h) for i in LEFT_EYE])
                    right_pts = np.array([(lms[i].x * w, lms[i].y * h) for i in RIGHT_EYE])
                    avg_ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                    if not is_calibrated:
                        elapsed = curr_time - start_time
                        if elapsed < calib_duration:
                            calibration_data.append(avg_ear)
                            cv2.putText(frame, f"CALIBRATING... {int(calib_duration - elapsed)}s", (30, 80), 1, 2, (0, 255, 255), 2)
                        elif len(calibration_data) > 0:
                            ear_threshold = (sum(calibration_data) / len(calibration_data)) * 0.75
                            is_calibrated = True
                    
                    else:
                        # --- [수정된 감지 로직] ---
                        if avg_ear < ear_threshold:
                            # 눈을 처음 감았을 때 시간 기록
                            if eye_closed_start_time == 0:
                                eye_closed_start_time = time.time()
                            
                            # 눈을 감고 있는 지속 시간 계산
                            closed_duration = time.time() - eye_closed_start_time
                            
                            if closed_duration >= 10.0:      # 5초 이상: DANGER (2단계)
                                current_stage = 2
                            elif closed_duration >= 2.0:    # 2초 이상: WARNING (1단계)
                                current_stage = 1
                        else:
                            # 눈을 뜨면 시간 및 단계 초기화
                            eye_closed_start_time = 0
                            current_stage = 0
                        # -------------------------

                        status_list = [("NORMAL", (0, 255, 0)), ("WARNING", (0, 165, 255)), ("DANGER", (0, 0, 255))]
                        text, color = status_list[current_stage]
                        cv2.putText(frame, text, (30, 90), 1, 3, color, 3)

            cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40), 1, 1.5, (255, 0, 0), 2)
            cv2.imshow("Jetson Drowsiness System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        is_running = False
        vs.stop()
        if ser:
            ser.write("!OFF_SAFE#".encode())
            ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    t_uart = threading.Thread(target=uart_thread)
    t_uart.start()
    main()
