import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
import serial
import time
import threading
from PIL import ImageFont, ImageDraw, Image

# --- 1. 카메라 캡처 스레드 클래스 ---
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- 2. UART 통신 설정 및 스레드 ---
try:
    ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=0.1)
except:
    print("[경고] UART 연결 실패. 시뮬레이션 모드로 작동합니다.")
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
                ser.write(cmd.encode())
            last_sent_stage = current_stage
        time.sleep(0.1)

# --- 3. 유틸리티 함수 (평활화, UI, EAR) ---
def apply_hist_eq(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def draw_korean(img, text, pos, size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def calculate_ear(eye_pts):
    v1 = dist.euclidean(eye_pts[1], eye_pts[5])
    v2 = dist.euclidean(eye_pts[2], eye_pts[4])
    h = dist.euclidean(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h)

# --- 4. 메인 실행부 ---
def main():
    global current_stage, is_running
    
    # 카메라 스레드 시작
    vs = WebcamStream(src=0).start()
    time.sleep(1.0) # 카메라 안정화 시간
    
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
    
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    calibration_data = []
    ear_threshold = 0
    is_calibrated = False
    calib_duration = 5
    start_time = time.time()
    counter = 0

    print("멀티스레딩 시스템 가동 시작")

    try:
        while True:
            frame = vs.read() # 스레드가 미리 읽어둔 최신 프레임 가져오기
            if frame is None: continue
            
            curr_time = time.time()
            
            # 1. 히스토그램 평활화 (전처리)
            processed_frame = apply_hist_eq(frame)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # 2. 얼굴 분석
            results = face_mesh.process(rgb_frame)
            h, w, _ = frame.shape

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
                            remain = int(calib_duration - elapsed)
                            frame = draw_korean(frame, f"보정 중... ({remain}초)", (30, 50), 30, (255, 255, 0))
                        else:
                            ear_threshold = (sum(calibration_data) / len(calibration_data)) * 0.75
                            is_calibrated = True
                    else:
                        if avg_ear < ear_threshold:
                            counter += 1
                            if counter >= 60: current_stage = 2
                            elif counter >= 20: current_stage = 1
                        else:
                            counter = 0
                            current_stage = 0

                        msg_list = ["정상 상태", "1단계 경고", "2단계 위험"]
                        color_list = [(255, 255, 255), (255, 165, 0), (255, 0, 0)]
                        frame = draw_korean(frame, msg_list[current_stage], (30, 100), 40, color_list[current_stage])
                        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Multi-Threaded System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        is_running = False
        vs.stop() # 카메라 스레드 정지
        t_uart.join()
        if ser:
            with lock:
                ser.write("!OFF#".encode())
                ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # UART 스레드 시작
    t_uart = threading.Thread(target=uart_thread)
    t_uart.start()
    main()