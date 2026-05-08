import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import sys

def get_gst_string(sensor_id=0, w=1280, h=720, fps=30, flip=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1, format=(string)NV12 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, w, h, fps, flip, w, h)
    )

def get_ear(eye):
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h = dist.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h)

EAR_LIMIT = 0.23
FRAME_LIMIT = 15

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)

# Create video capture using the corrected GST pipeline
gst_pipeline = get_gst_string()
video_capture = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not video_capture.isOpened():
    print("Error: Unable to open camera.")
    sys.exit()

drowsy_counter = 0

while True:
    success, frame = video_capture.read()
    if not success:
        print("Error: Frame grab failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 0)

    for face in faces:
        shape = landmark_predictor(gray, face)
        coords = np.array([[p.x, p.y] for p in shape.parts()])

        left_ear = get_ear(coords[L_START:L_END])
        right_ear = get_ear(coords[R_START:R_END])
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_LIMIT:
            drowsy_counter += 1
            if drowsy_counter >= FRAME_LIMIT:
                cv2.putText(frame, "DROWSINESS WARNING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            drowsy_counter = 0

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
