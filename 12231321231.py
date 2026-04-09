import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    coords = []
    for i in eye_indices:
        lm = landmarks[i]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

cap = cv2.VideoCapture(0)
EAR_THRESHOLD = 0.22 

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            
            avg_ear = (left_ear + right_ear) / 2.0

            status = "Eyes Open"
            color = (0, 255, 0)
            if avg_ear < EAR_THRESHOLD:
                status = "Eyes Closed"
                color = (0, 0, 255)

            cv2.putText(image, f"EAR: {avg_ear:.2f}", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, status, (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Jetson Nano - Eye Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()