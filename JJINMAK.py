import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
import serial
import time

# --- [Hardware] Serial Communication Class ---
class DrowsinessController:
    def __init__(self, port='/dev/ttyTHS1', baudrate=115200):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=2) # 2초 타임아웃 설정
            time.sleep(2)
            self.is_connected = True
            print(f"[SUCCESS] UART port connected: {port}")
        except Exception as e:
            print(f"[WARNING] Serial port connection failed (Running without hardware): {e}")
            self.ser = None
            self.is_connected = False

    def send_and_wait(self, command):
        """데이터를 보내고 ACK를 받을 때까지 대기"""
        if not self.is_connected:
            return False
            
        packet = f"!{command}#"
        self.ser.write(packet.encode('utf-8'))
        
        # 응답 읽기 (끝까지 기다림)
        try:
            response = self.ser.readline().decode('utf-8').strip()
            
            if response:
                print(f"[Success] ESP32 Response: {response}")
                return True
            else:
                print("[Fail] No response from ESP32")
                return False
        except Exception as e:
            print(f"[Error] Reading serial response: {e}")
            return False

# --- [Vision] EAR Calculation ---
def calculate_ear(eye_landmarks):
    v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

# --- System Initialization ---
print("--- 3-Stage Drowsiness Detection System Started ---")

# 새로운 하드웨어 컨트롤러 객체 생성
ctrl = DrowsinessController(port='/dev/ttyTHS1', baudrate=115200)

# refine_landmarks 옵션 제거 (젯슨 나노 구버전 호환)
face_mesh = mp_face_mesh.FaceMesh()

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Calibration Variables
CALIBRATION_FRAMES = 50
calibration_data = []
EAR_THRESHOLD = 0
is_calibrated = False

# Stage Control Variables
STAGE1_FRAMES = 20
STAGE2_FRAMES = 60
counter = 0
current_stage = 0 

# USB 웹캠을 위한 CAP_V4L2 옵션 및 해상도 설정
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Main Loop ---
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        img_h, img_w, _ = frame.shape
        
        target_stage = 0 

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_EYE])
                right_pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_EYE])
                avg_ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                # [1] Auto-Calibration Phase
                if not is_calibrated:
                    calibration_data.append(avg_ear)
                    cv2.putText(frame, "Keep eyes open (Calibrating...)", (30, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    if len(calibration_data) >= CALIBRATION_FRAMES:
                        normal_ear = sum(calibration_data) / len(calibration_data)
                        EAR_THRESHOLD = normal_ear * 0.75 
                        is_calibrated = True
                        print(f"[READY] Calibration Complete! Threshold: {EAR_THRESHOLD:.3f}")
                
                # [2] Drowsiness Detection Phase
                else:
                    if avg_ear < EAR_THRESHOLD:
                        counter += 1
                        
                        if counter >= STAGE2_FRAMES:
                            target_stage = 2
                            cv2.putText(frame, "STAGE 2: DANGER", (30, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        elif counter >= STAGE1_FRAMES:
                            target_stage = 1
                            cv2.putText(frame, "STAGE 1: WARNING", (30, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                    else:
                        counter = 0
                        target_stage = 0

                # [3] Hardware Control (새로운 클래스 적용)
                if is_calibrated and target_stage != current_stage:
                    if target_stage == 0:
                        ctrl.send_and_wait("OFF")
                    elif target_stage == 1:
                        ctrl.send_and_wait("LV1_WARN")
                    elif target_stage == 2:
                        ctrl.send_and_wait("LV2_DANGER")
                        
                    current_stage = target_stage

                # Status Overlay
                status_text = f"EAR: {avg_ear:.2f} | THR: {EAR_THRESHOLD:.2f} | Cnt: {counter}"
                cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Drowsiness Detection System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

except KeyboardInterrupt:
    print("\n[STOP] System forcefully closed by user.")

finally:
    print("Releasing resources...")
    if ctrl.is_connected:
        ctrl.send_and_wait("OFF") 
        ctrl.ser.close()
    cap.release()
    cv2.destroyAllWindows() 