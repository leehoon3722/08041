import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from PIL import ImageFont, ImageDraw, Image

def gstreamer_pipeline(cap_w=1280, cap_h=720, disp_w=640, disp_h=360, fps=30, flip=0):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (cap_w, cap_h, fps, flip, disp_w, disp_h)
    )

def draw_korean(img, text, pos, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)

def calculate_ear(eye_landmarks):
    v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (v1 + v2) / (2.0 * h)

print("--- 시스템 가동 ---")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

CALIBRATION_FRAMES = 50
calibration_data = []
EAR_THRESHOLD = 0.0
is_calibrated = False
CLOSED_FRAMES = 20
counter = 0

# CSI 카메라 사용 시 gstreamer_pipeline 사용, USB 카메라 사용 시 0 입력
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 야간 대비 강화 (CLAHE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    rgb_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    results = face_mesh.process(rgb_frame)
    img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            l_pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_EYE])
            r_pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_EYE])
            
            avg_ear = (calculate_ear(l_pts) + calculate_ear(r_pts)) / 2.0
            
            if not is_calibrated:
                calibration_data.append(avg_ear)
                frame = draw_korean(frame, "정면을 보세요 (측정 중...)", (30, 100), 25, (0, 255, 255))
                if len(calibration_data) >= CALIBRATION_FRAMES:
                    EAR_THRESHOLD = (sum(calibration_data) / len(calibration_data)) * 0.75
                    is_calibrated = True
                    print("측정 완료! 임계값: {:.3f}".format(EAR_THRESHOLD))
            else:
                if avg_ear < EAR_THRESHOLD:
                    counter += 1
                    if counter >= CLOSED_FRAMES:
                        frame = draw_korean(frame, "위험! 졸음운전 주의!", (30, 150), 30, (255, 0, 0))
                else:
                    counter = 0
            
            cv2.putText(frame, "EAR: {:.2f} (TH: {:.2f})".format(avg_ear, EAR_THRESHOLD), 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Drowsiness System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
