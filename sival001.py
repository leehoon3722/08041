import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
import serial
import time

# --- [Hardware] 젯슨 나노 TX, RX (UART) 시리얼 포트 설정 ---
# /dev/ttyTHS1은 젯슨 나노의 핀 헤더(UART)를 의미합니다.
try:
    ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)
    print("[SUCCESS] UART(TX/RX) port connected: /dev/ttyTHS1")
except Exception as e:
    print(f"[WARNING] UART connection failed (Running software only): {e}")
    ser = None

# 요청하신 아스키 코드 패킷 전송 방식 적용
def send_uart_command(stage):
    if ser is None or not ser.is_open:
        return
        
    # 시작문자 '!' , 끝문자 '#' 규격에 맞춘 패킷 생성
    if stage == 0:
        command = "!VIB_OFF#\n"
    elif stage == 1:
        command = "!BUZ_ON#\n"
    elif stage == 2:
        command = "!VIB_ON#\n"
    else:
        return
        
    # ascii 인코딩으로 TX 송신
    ser.write(command.encode('ascii'))
    print(f"[UART TX] 전송 완료: {command.strip()}")

# --- [Vision] EAR Calculation ---
def calculate_ear(eye_landmarks):
    v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

# --- System Initialization ---
print("--- 3-Stage Drowsiness Detection System Started ---")
face_mesh = mp_face_mesh.FaceMesh()

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Calibration Variables
CALIBRATION_FRAMES = 50
calibration_data = []
EAR_THRESHOLD = 0
is_calibrated = False

# Stage Control Variables
STAGE1_FRAMES = 20  # Approx. 1 sec -> Stage 1 (Buzzer)
STAGE2_FRAMES = 60  # Approx. 3 sec -> Stage 2 (Vibration)
counter = 0
current_stage = 0 

cap = cv2.VideoCapture(1)

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
                            cv2.putText(frame, "STAGE 2: DANGER (VIB ON)", (30, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        elif counter >= STAGE1_FRAMES:
                            target_stage = 1
                            cv2.putText(frame, "STAGE 1: WARNING (BUZ ON)", (30, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                    else:
                        counter = 0
                        target_stage = 0

                # [3] TX 송신 제어 (상태가 변했을 때만 패킷 전송)
                if is_calibrated and target_stage != current_stage:
                    send_uart_command(target_stage)
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
    send_uart_command(0) # 프로그램 종료 시 안전하게 하드웨어 끄기
    if ser and ser.is_open:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()
