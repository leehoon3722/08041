import cv2

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    # 1. 문자열 내의 (int), (fraction) 등의 표현이 간혹 문제를 일으킬 수 있어 표준 형식으로 수정했습니다.
    # 2. appsink에 drop=True를 추가하여 버퍼가 쌓여서 멈추는 현상을 방지했습니다.
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=True"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# CAP_GSTREAMER 플래그는 필수입니다.
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

if cap.isOpened():
    print("카메라 연결 성공! 창이 뜨는지 확인하세요.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break
        
        cv2.imshow("CSI Camera Test", frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
else:
    # 여기가 계속 뜬다면 하드웨어 데몬 문제입니다.
    print("카메라를 열 수 없습니다. 아래 해결법을 시도해 보세요.")
