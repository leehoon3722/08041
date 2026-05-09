import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
import serial
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

# --- 2. UART 통신 스레드 (ESP32 프로토콜 완벽 동기화) ---
try:
    ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=0.1)
    print("✅ UART 통신 연결 성공")
except Exception as e:
    ser = None
    print(f"🚨 UART 연결 실패 (시뮬레이션 모드로 작동): {e}")

target_state = "OFF"
is_running = True
lock = threading.Lock()

def uart_thread():
    global target_state, is_running
    current_state = "OFF"
    last_heartbeat_time = time.time()

    def send_command(cmd_text):
        """명령어 전송 및 ESP32의 'A\n' 응답 대기"""
        full_packet = f"!{cmd_text}#\n"
        retry_count = 0
        
        while is_running and retry_count < 10: # 최대 10번 재시도
            if not ser: break
            try:
                ser.reset_input_buffer() # 이전의 쓰레기 응답값 비우기
                ser.write(full_packet.encode())
                print(f">> 상태 전송: {full_packet.strip()}")
                
                time.sleep(0.1) # ESP32가 처리하고 응답할 시간 제공
                if ser.in_waiting > 0:
                    res = ser.readline().decode('utf-8', errors='ignore').strip()
                    if res == "A": 
                        print(f"✅ 수신 확인 완료 (ACK)")
                        break
            except Exception as e:
                print(f"🚨 통신 에러: {e}")
            
            retry_count += 1
            time.sleep(0.1)

    while is_running:
        with lock:
            cmd_to_send = target_state
        
        # 1. 메인 루프에서 상태가 변했을 때만 ESP32로 전송
        if cmd_to_send != current_state:
            send_command(cmd_to_send)
            current_state = cmd_to_send
            # 상태 명령을 보냈다면, 하트비트 타이머를 리셋! (패킷 겹침 방지)
            last_heartbeat_time = time.time()
            
        # 2. 하트비트 전송 (상태 변화가 없을 때 1초마다)
        curr_time = time.time()
        if curr_time - last_heartbeat_time >= 1.0:
            if ser:
                try:
                    ser.write(b"!H#\n")
                    print("💓 하트비트 전송: !H#") # 터미널 확인용
                    
                    time.sleep(0.05)
                    # 하트비트에 대한 ESP32의 응답을 버퍼에서 비워주기
                    if ser.in_waiting > 0:
                        ser.readline() 
                except Exception as e:
                    print(f"🚨 하트비트 전송 에러: {e}")
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
