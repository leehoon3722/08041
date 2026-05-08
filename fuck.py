import cv2
import sys

def get_gst_string():
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate=(fraction)30/1, format=(string)NV12 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def open_camera():
    pipeline = get_gst_string()
    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not video_capture.isOpened():
        print("Error: Unable to open camera.")
        return

    print("Camera started successfully. Press 'q' to exit.")

    while True:
        success, frame = video_capture.read()
        if not success:
            print("Error: Frame grab failed.")
            break

        cv2.imshow("Jetson Nano Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()
