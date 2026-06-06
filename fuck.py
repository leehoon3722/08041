import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
import socket
import time
import threading
import sys
import os

# ══════════════════════════════════════════
# 0. 파라미터 설정
# ══════════════════════════════════════════

# CSI 카메라
SENSOR_ID    = 0
CAP_WIDTH    = 640
CAP_HEIGHT   = 480
DISP_WIDTH   = 640
DISP_HEIGHT  = 480
FPS_TARGET   = 18
FLIP_METHOD  = 0

# 블루투스
ESP32_MAC_ADDR        = "08:3A:F2:B9:79:E2"
BT_CHANNEL            = 1
BT_TIMEOUT            = 10.0
BT_HEARTBEAT_INTERVAL = 1.0    
BT_ACK_RETRY          = 10     
BT_RECONNECT_INTERVAL = 5.0    
BT_ABNORMAL_THRESHOLD = 3      
BT_ABNORMAL_RESET_SEC = 30.0   

# EAR 캘리브레이션 및 졸음 판별 파라미터
CALIB_SEC         = 5.0   
CALIB_RATIO       = 0.75  
MIN_EAR_TH        = 0.18  
MAX_EAR_TH        = 0.35  

CLOSED_FRAMES_LV1 = 9   
CLOSED_FRAMES_LV2 = 27  

# 얼굴 없음 → 위험
FACE_MISSING_DANGER_SEC = 5.0

# 워치독
WATCHDOG_THRESHOLD      = 3      
WATCHDOG_RESET_SEC      = 30.0   
WATCHDOG_CAM_FAIL_LIMIT = 45     

# MediaPipe 눈 랜드마크
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ══════════════════════════════════════════
# 1. GStreamer / CSI 카메라
# ══════════════════════════════════════════
def check_opencv_gstreamer():
    info = cv2.getBuildInformation()
    if "GStreamer" in info and "YES" in info.split("GStreamer")[1].split("\n")[0]:
        print("✅ OpenCV GStreamer 지원 확인")
    else:
        print("🚨 [치명적 오류] OpenCV가 GStreamer를 지원하지 않습니다!")
        sys.exit(1)

def gstreamer_pipeline():
    return (
        f"nvarguscamerasrc sensor-id={SENSOR_ID} ! "
        f"video/x-raw(memory:NVMM), width=(int){CAP_WIDTH}, height=(int){CAP_HEIGHT}, "
        f"framerate=(fraction){FPS_TARGET}/1 ! "
        f"nvvidconv flip-method={FLIP_METHOD} ! "
        f"video/x-raw, width=(int){DISP_WIDTH}, height=(int){DISP_HEIGHT}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True"
    )

class CSICameraStream:
    def __init__(self):
        pipeline = gstreamer_pipeline()
        self.stream = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.stream.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
            sys.exit(1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped           = False
        self._lock             = threading.Lock()
        self._consecutive_fail = 0

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            if self.stream.isOpened():
                grabbed, frame = self.stream.read()
                with self._lock:
                    if grabbed and frame is not None:
                        self.grabbed           = grabbed
                        self.frame             = frame
                        self._consecutive_fail = 0
                    else:
                        self._consecutive_fail += 1

    def read(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    @property
    def consecutive_fail(self):
        with self._lock:
            return self._consecutive_fail

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        if self.stream.isOpened():
            self.stream.release()

# ══════════════════════════════════════════
# 2. 블루투스
# ══════════════════════════════════════════
sock         = None
sock_lock    = threading.Lock()
target_state = "OFF"
is_running   = True
bt_lock      = threading.Lock()

sim_mode      = False
sim_mode_lock = threading.Lock()
_sim_log_buf  = []

_bt_abnormal_count     = 0
_bt_last_abnormal_time = 0.0
_bt_abnormal_lock      = threading.Lock()

def _bt_record_abnormal(reason: str):
    global _bt_abnormal_count, _bt_last_abnormal_time
    with _bt_abnormal_lock:
        now = time.time()
        if _bt_last_abnormal_time > 0 and (now - _bt_last_abnormal_time) >= BT_ABNORMAL_RESET_SEC:
            _bt_abnormal_count = 0
        _bt_abnormal_count    += 1
        _bt_last_abnormal_time = now
        count = _bt_abnormal_count

    print(f"⚠️  [BT 워치독] 비정상 응답 {count}/{BT_ABNORMAL_THRESHOLD}회 — 원인: {reason}")
    if count >= BT_ABNORMAL_THRESHOLD:
        _bt_restart()

def _bt_record_normal():
    global _bt_abnormal_count, _bt_last_abnormal_time
    with _bt_abnormal_lock:
        _bt_abnormal_count     = 0
        _bt_last_abnormal_time = 0.0

def _bt_restart():
    global sock, is_running
    print(f"\n🔁 [BT 워치독] 비정상 응답 도달 → 프로세스 재시작\n")
    is_running = False
    with sock_lock:
        if sock:
            try: sock.send("!OFF#\n".encode("utf-8"))
            except: pass
            try: sock.close()
            except: pass
            sock = None
    cv2.destroyAllWindows()
    time.sleep(0.5)
    os.execv(sys.executable, [sys.executable] + sys.argv)

def _enter_sim_mode(reason: str):
    global sim_mode
    with sim_mode_lock:
        if sim_mode: return
        sim_mode = True
    print(f"\n⚠️  [시뮬레이션 모드] 블루투스 연결 실패 - 원인: {reason}\n")

def _exit_sim_mode():
    global sim_mode
    with sim_mode_lock:
        if not sim_mode: return
        sim_mode = False
    print(f"\n✅ [시뮬레이션 종료] 블루투스 재연결 성공\n")
    _sim_log_buf.clear()

def _sim_log(state: str):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] 상태={state}"
    _sim_log_buf.append(entry)
    print(f"📋 [SIM] {entry}")

def _create_socket():
    try:
        s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        s.settimeout(BT_TIMEOUT)
        s.connect((ESP32_MAC_ADDR, BT_CHANNEL))
        print(f"✅ 블루투스 연결 성공")
        return s
    except Exception as e:
        print(f"⚠️  블루투스 연결 실패: {e}")
        return None

def _init_bluetooth():
    global sock
    with sock_lock:
        sock = _create_socket()
    if sock is None:
        _enter_sim_mode("초기 연결 실패")

def _send_command(cmd_text: str) -> bool:
    global sock
    with sock_lock:
        if not sock: return False
        packet = f"!{cmd_text}#\n".encode("utf-8")
        for _ in range(BT_ACK_RETRY):
            if not is_running: return False
            try:
                sock.send(packet)
                time.sleep(0.1)
                try:
                    res = sock.recv(1024).decode("utf-8", errors="ignore").strip()
                    if "A" in res:
                        _bt_record_normal()
                        return True
                    elif res:
                        _bt_record_abnormal(f"비정상: '{res}'")
                    else:
                        _bt_record_abnormal("빈 응답")
                except socket.timeout:
                    _bt_record_abnormal("응답 타임아웃")
            except Exception as e:
                try: sock.close()
                except: pass
                sock = None
                return False
            time.sleep(0.1)
    return False

def bluetooth_thread():
    global sock, is_running
    current_state  = "OFF"
    last_heartbeat = time.time()
    last_reconnect = 0.0

    while is_running:
        with sock_lock:
            is_connected = sock is not None
        if not is_connected:
            now = time.time()
            if now - last_reconnect >= BT_RECONNECT_INTERVAL:
                new_sock = _create_socket()
                with sock_lock: sock = new_sock
                last_reconnect = now
                if sock:
                    _exit_sim_mode()
                    current_state = "FORCE_RESEND"
                else:
                    _enter_sim_mode("재연결 실패")
            time.sleep(0.5)
            continue

        with bt_lock:
            cmd = target_state

        if cmd != current_state or current_state == "FORCE_RESEND":
            success = _send_command(cmd)
            if success:
                print(f">> BT: {current_state} → {cmd}")
                current_state  = cmd
                last_heartbeat = time.time()

        now = time.time()
        if now - last_heartbeat >= BT_HEARTBEAT_INTERVAL:
            with sock_lock: s = sock
            if s:
                try:
                    s.send("!H#\n".encode("utf-8"))
                    time.sleep(0.05)
                    try:
                        res = s.recv(1024).decode("utf-8", errors="ignore").strip()
                        if "A" in res: _bt_record_normal()
                        elif res: _bt_record_abnormal(f"HB 비정상: '{res}'")
                    except socket.timeout: pass
                except Exception:
                    with sock_lock:
                        try: sock.close()
                        except: pass
                        sock = None
            last_heartbeat = now
        time.sleep(0.05)

# ══════════════════════════════════════════
# 3. EAR 계산
# ══════════════════════════════════════════
def calculate_ear(eye_pts: np.ndarray) -> float:
    v1 = dist.euclidean(eye_pts[1], eye_pts[5])
    v2 = dist.euclidean(eye_pts[2], eye_pts[4])
    h  = dist.euclidean(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

# ══════════════════════════════════════════
# 4. 신호 워치독
# ══════════════════════════════════════════
class SignalWatchdog:
    SIGNAL_TYPES = ("CAM_FAIL", "EAR_INVALID", "PROCESS_EXCEPT")

    def __init__(self):
        self._counts    = {s: 0 for s in self.SIGNAL_TYPES}
        self._last_err  = {s: 0.0 for s in self.SIGNAL_TYPES}
        self._lock      = threading.Lock()

    def record(self, signal_type: str):
        if signal_type not in self.SIGNAL_TYPES: return
        with self._lock:
            self._reset_if_expired(signal_type)
            self._counts[signal_type] += 1
            self._last_err[signal_type] = time.time()
            count = self._counts[signal_type]

        print(f"⚠️  [워치독] {signal_type} 오류 {count}/{WATCHDOG_THRESHOLD}회")
        if count >= WATCHDOG_THRESHOLD:
            self._restart(signal_type)

    def clear(self, signal_type: str):
        with self._lock:
            self._counts[signal_type]   = 0
            self._last_err[signal_type] = 0.0

    def counts(self) -> dict:
        with self._lock:
            return dict(self._counts)

    def _reset_if_expired(self, signal_type: str):
        last = self._last_err[signal_type]
        if last > 0 and (time.time() - last) >= WATCHDOG_RESET_SEC:
            self._counts[signal_type]   = 0
            self._last_err[signal_type] = 0.0

    def _restart(self, trigger: str):
        print(f"\n🔁 [워치독] {trigger} 오류 → 프로세스 재시작\n")
        global sock, is_running
        is_running = False
        with sock_lock:
            if sock:
                try: sock.send("!OFF#\n".encode("utf-8"))
                except: pass
                try: sock.close()
                except: pass
        cv2.destroyAllWindows()
        time.sleep(0.5)
        os.execv(sys.executable, [sys.executable] + sys.argv)

# ══════════════════════════════════════════
# 5. 화면 오버레이
# ══════════════════════════════════════════
STATE_COLORS = {
    "OFF"        : (50,  205, 50),
    "LV1_WARN"   : (0,   165, 255),
    "LV2_DANGER" : (0,   0,   255),
}

def draw_overlay(frame, state, fps, ear, threshold, closed_frames, bt_ok, wd_counts, is_calibrated, calib_remain):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS {int(fps)}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    with sim_mode_lock: is_sim = sim_mode
    if is_sim:
        blink_on = int(time.time() * 2) % 2 == 0
        cv2.putText(frame, "SIM MODE", (w - 160, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 200, 255) if blink_on else (0, 120, 180), 2)
    else:
        cv2.putText(frame, "BT:ON" if bt_ok else "BT:OFF", (w - 120, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (50, 205, 50) if bt_ok else (0, 0, 255), 2)

    if not is_calibrated:
        cv2.putText(frame, "CALIBRATING", (w // 2 - 120, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.putText(frame, f"Please look at the camera... {int(calib_remain)}s", 
                    (w // 2 - 220, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, state, (w // 2 - 120, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, STATE_COLORS.get(state, (255, 255, 255)), 3)

    bar_y = h - 60
    cv2.rectangle(frame, (0, bar_y - 5), (w, h), (20, 20, 20), -1)

    info_text = f"EAR: {ear:.3f} | THR: {threshold:.3f} | CLOSED FRAMES: {closed_frames}"
    cv2.putText(frame, info_text, (10, bar_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    wd_text_parts = []
    label_map = {"CAM_FAIL": "CAM", "EAR_INVALID": "EAR", "PROCESS_EXCEPT": "EXC"}
    for sig, cnt in wd_counts.items():
        if cnt > 0:
            wd_text_parts.append(f"{label_map[sig]}:{cnt}/{WATCHDOG_THRESHOLD}")
    if wd_text_parts:
        wd_text  = "WD " + "  ".join(wd_text_parts)
        wd_color = (0, 165, 255) if max(wd_counts.values()) < WATCHDOG_THRESHOLD else (0, 0, 255)
        cv2.putText(frame, wd_text, (10, bar_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, wd_color, 1)

def draw_no_face(frame, missing_sec):
    if missing_sec >= FACE_MISSING_DANGER_SEC:
        cv2.putText(frame, "NO FACE - DANGER!", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(frame, f"FACE MISSING... {int(FACE_MISSING_DANGER_SEC - missing_sec)}s",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

# ══════════════════════════════════════════
# 6. 메인 루프
# ══════════════════════════════════════════
def main():
    global target_state, is_running

    check_opencv_gstreamer()
    vs = CSICameraStream().start()
    time.sleep(2.0)
    print("✅ CSI 카메라 시작")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    watchdog = SignalWatchdog()

    state              = "OFF"
    prev_state         = "OFF"
    face_missing_start = 0.0
    closed_frames      = 0      
    prev_time          = time.time()

    is_calibrated      = False
    calib_start_time   = 0.0
    calib_ear_list     = []
    ear_threshold      = 0.25

    # ──────── 추가된 프레임 스킵용 변수 ────────
    frame_counter      = 0
    last_raw_ear       = 0.0
    last_face_found    = False

    print("── 시작 (R: 수동 캘리브레이션, Q: 종료) ──")

    try:
        while True:
            frame = vs.read()

            if frame is None or vs.consecutive_fail > WATCHDOG_CAM_FAIL_LIMIT:
                watchdog.record("CAM_FAIL")
                closed_frames = 0
                time.sleep(0.01)
                continue
            else:
                watchdog.clear("CAM_FAIL")

            curr_time = time.time()
            fps       = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            h, w, _ = frame.shape
            
            # ──────── 2프레임당 1번만 연산 수행 ────────
            frame_counter += 1
            process_this_frame = (frame_counter % 2 == 0)

            if process_this_frame:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    results = face_mesh.process(rgb)
                    last_face_found = False # 초기화 후 재확인
                    
                    if results.multi_face_landmarks:
                        largest_face, max_area = None, 0.0
                        
                        for face_lms in results.multi_face_landmarks:
                            xs   = [lm.x for lm in face_lms.landmark]
                            ys   = [lm.y for lm in face_lms.landmark]
                            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                            if area > max_area:
                                max_area, largest_face = area, face_lms

                        if largest_face:
                            lms = largest_face.landmark
                            left_pts  = np.array([(lms[i].x * w, lms[i].y * h, lms[i].z * w) for i in LEFT_EYE])
                            right_pts = np.array([(lms[i].x * w, lms[i].y * h, lms[i].z * w) for i in RIGHT_EYE])
                            
                            # 새 EAR 계산 및 캐싱
                            last_raw_ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                            if 0.0 < last_raw_ear < 1.0:
                                last_face_found = True
                                watchdog.clear("EAR_INVALID")
                                watchdog.clear("PROCESS_EXCEPT")
                            else:
                                watchdog.record("EAR_INVALID")
                except Exception as e:
                    print(f"⚠️ MediaPipe 처리 예외: {e}")
                    watchdog.record("PROCESS_EXCEPT")
                    last_face_found = False

            # ──────── 캐싱된 데이터를 기반으로 판별 및 UI 업데이트 (매 프레임) ────────
            calib_remain = 0.0

            if last_face_found:
                face_missing_start = 0.0
                
                # 캘리브레이션 진행
                if not is_calibrated:
                    if calib_start_time == 0.0:
                        calib_start_time = curr_time
                        calib_ear_list.clear()

                    elapsed = curr_time - calib_start_time
                    calib_remain = max(0.0, CALIB_SEC - elapsed)
                    
                    # 연산된 프레임에서만 리스트에 추가 (중복 방지)
                    if process_this_frame:
                        calib_ear_list.append(last_raw_ear)

                    if elapsed >= CALIB_SEC:
                        if calib_ear_list:
                            baseline = sum(calib_ear_list) / len(calib_ear_list)
                            calculated_th = baseline * CALIB_RATIO
                            ear_threshold = max(MIN_EAR_TH, min(MAX_EAR_TH, calculated_th))
                            print(f"✅ [캘리브레이션 완료] 기준 EAR: {baseline:.3f} → 설정된 임계값: {ear_threshold:.3f}")
                        
                        is_calibrated = True
                        closed_frames = 0

                # 일반 EAR 감지 로직
                else:
                    if last_raw_ear < ear_threshold:
                        closed_frames += 1
                    else:
                        closed_frames = 0

                    if closed_frames >= CLOSED_FRAMES_LV2:
                        state = "LV2_DANGER"
                    elif closed_frames >= CLOSED_FRAMES_LV1:
                        state = "LV1_WARN"
                    else:
                        state = "OFF"
                        
                with sock_lock:
                    bt_ok = sock is not None
                draw_overlay(frame, state, fps, last_raw_ear, ear_threshold, closed_frames, bt_ok, watchdog.counts(), is_calibrated, calib_remain)

            else:
                # 얼굴을 놓쳤을 때 캘리브레이션 중이었다면 타이머 갱신 (초기화는 안 함)
                if not is_calibrated and calib_start_time != 0.0:
                    calib_start_time += (curr_time - prev_time)

                closed_frames = 0
                if face_missing_start == 0.0:
                    face_missing_start = curr_time
                missing_sec = curr_time - face_missing_start

                state = "LV2_DANGER" if missing_sec >= FACE_MISSING_DANGER_SEC else "OFF"

                with sock_lock:
                    bt_ok = sock is not None

                draw_no_face(frame, missing_sec)
                draw_overlay(frame, state, fps, 0.0, ear_threshold, 0, bt_ok, watchdog.counts(), is_calibrated, 0.0)

            # 상태 변경 시 BT 전송
            if state != prev_state and is_calibrated:
                print(f"[상태 변경] {prev_state} → {state}")
                with sim_mode_lock: is_sim = sim_mode
                if is_sim:
                    _sim_log(state)
                else:
                    with bt_lock: target_state = state
                prev_state = state

            cv2.imshow("Jetson Drowsiness System (Lite + Auto_Calib)", frame)

            # 키보드 이벤트 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r") or key == ord("R"):
                print("🔄 [수동 재보정] R키 입력: 5초 캘리브레이션 재시작")
                is_calibrated = False
                calib_start_time = 0.0
                calib_ear_list.clear()
                closed_frames = 0
                state = "OFF"
                with bt_lock: target_state = "OFF"

    finally:
        is_running = False
        vs.stop()
        with sock_lock:
            if sock:
                try: sock.send("!OFF#\n".encode("utf-8"))
                except: pass
                try: sock.close()
                except: pass
        cv2.destroyAllWindows()
        print("── 종료 ──")

if __name__ == "__main__":
    print("=== 젯슨 나노 졸음운전 감지 시스템 시작 ===")
    _init_bluetooth()
    threading.Thread(target=bluetooth_thread, daemon=True).start()
    main()
