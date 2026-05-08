import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye_landmarks):
    v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (v1 + v2) / (2.0 * h)

mp_face_mesh = mp.solutions.face_mesh
# 3.6 버전 호환: refine_landmarks 옵션 제외
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.22  
CLOSED_FRAMES = 15
counter = 0

# CSI 카메라 또는 USB IR 카메라 선택 (보통 0번)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- 야간 이미지 전처리 (IR 카메라 최적화) ---
    # 1. 그레이스케일로 변환 후 대비 강화 (CLAHE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # 2. MediaPipe 처리를 위해 다시 RGB 형식으로 복사 (3채널 유지)
    rgb_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    results = face_mesh.process(rgb_frame)
    img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            left_pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_EYE])
            right_pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_EYE])
            
            avg_ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0
            
            status = "Eyes Open"
            color = (0, 255, 0)

            if avg_ear < EAR_THRESHOLD:
                counter += 1
                if counter >= CLOSED_FRAMES:
                    status = "DROWSY WARNING!"
                    color = (0, 0, 255)
            else:
                counter = 0

            # 결과 표시 (원본 이미지 위에 출력)
            cv2.putText(frame, "EAR: {:.2f}".format(avg_ear), (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, status, (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Night-Drowsiness-IR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
