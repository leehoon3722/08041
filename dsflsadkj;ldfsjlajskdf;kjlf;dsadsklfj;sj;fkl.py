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

            # [핵심 최적화] AI 분석용 이미지 축소 (320x240)
            small_frame = cv2.resize(frame, (320, 240))
            small_frame = apply_fast