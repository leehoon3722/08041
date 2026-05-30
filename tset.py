"""
졸음 감지 시스템 — 젯슨 나노 최종 보완 및 버그 수정 버전
=========================================================
[수정 및 반영 사항]
  ① 화질 개선 : MediaPipe 처리 및 연산 해상도를 (640, 360)으로 상향
  ② 좌표 매핑 버그 수정 : MediaPipe 입력 영상 해상도와 랜드마크 픽셀 변환 해상도 일치 (눈 감지 정상화)
  ③ 하트비트 전송 보완 : 하트비트 패킷 유실 및 상태 꼬임 방지, 정상 전송 보증
  ④ 안전 장치 : 블루투스 자동 재연결, IQR 기반 캘리브레이션 필터, 신호 워치독 유지
"""

import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial import distance as dist
from collections import deque
import socket
import time
import threading
import sys
import os

# ══════════════════════════════════════════
# 0. 파라미터 설정
# ══════════════════════════════════════════

# CSI 카메라 및 처리 해상도
SENSOR_ID    = 0
CAP_WIDTH    = 1280
CAP_HEIGHT   = 720
DISP_WIDTH   = 640
DISP_HEIGHT  = 480
PROCESS_W    = 640  # 화질 향상을 위한 MediaPipe/연산용 가로 크기
PROCESS_H    = 360  # 화질 향상을 위한 MediaPipe/연산용 세로 크기
FPS_TARGET   = 30
FLIP_METHOD  = 0

# 블루투스
ESP32_MAC_ADDR        = "08:3A:F2:B9:79:E2"
BT_CHANNEL            = 1
BT_TIMEOUT            = 5.0
BT_HEARTBEAT_INTERVAL = 1.0    # 하트비트 주기 (초)
BT_ACK_RETRY          = 10     # ACK 재시도 횟수
BT_RECONNECT_INTERVAL = 5.0    # 재연결 시도 간격 (초)
BT_ABNORMAL_THRESHOLD = 3      # ESP32 비정상 응답 N회 → 프로세스 재시작
BT_ABNORMAL_RESET_SEC = 30.0   # 이 시간 동안 정상 응답이면 카운트 리셋

# EMA 스무딩
EMA_ALPHA          = 0.40

# PERCLOS
PERCLOS_WINDOW_SEC = 2.0
PERCLOS_LV1        = 0.15
PERCLOS_LV2        = 0.35
PERCLOS_EYE_RATIO  = 0.80

# 즉각 트리거
INSTANT_RATIO      = 0.60
INSTANT_LV1_FRAMES = 2
INSTANT_LV2_FRAMES = 8

# 캘리브레이션
CALIB_DURATION_SEC = 5.0
CALIB_IQR_FACTOR   = 1.5   # IQR × 이 값 이내만 유효 샘플로 인정

# 적응형 재보정
RECALIB_INTERVAL   = 120.0
RECALIB_WINDOW_SEC = 10.0
RECALIB_MAX_DRIFT  = 0.10

# 얼굴 없음 → 위험
FACE_MISSING_DANGER_SEC = 5.0

# 워치독 (잘못된 신호 감지 → 프로세스 재시작)
WATCHDOG_THRESHOLD      = 3      # 이 횟수 이상 잘못된 신호 → 재시작
WATCHDOG_RESET_SEC      = 30.0   # 이 시간 동안 오류 없으면 카운트 초기화
WATCHDOG_CAM_FAIL_LIMIT = 30     # 연속 카메라 드롭 프레임 수 기준

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
    """별도 스레드로 CSI 카메라 프레임 지속 수신 (버퍼 지연 방지)"""
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
            time.sleep(0.01)

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
# 2. 블루투스 통신 및 하트비트 스레드
# ══════════════════════════════════════════
sock         = None
sock_lock    = threading.Lock()
target_state = "OFF"
is_running   = True
bt_lock      = threading.Lock()

# 시뮬레이션 모드 관련
sim_mode      = False         # BT 연결 실패 시 True로 전환
sim_mode_lock = threading.Lock()
_sim_log_buf  = []            # 시뮬레이션 상태 변경 기록

# ESP32 비정상 응답 카운터
_bt_abnormal_count    = 0
_bt_last_abnormal_time = 0.0
_bt_abnormal_lock     = threading.Lock()

def _bt_record_abnormal(reason: str):
    global _bt_abnormal_count, _bt_last_abnormal_time
    with _bt_abnormal_lock:
        now = time.time()
        if _bt_last_abnormal_time > 0 and (now - _bt_last_abnormal_time) >= BT_ABNORMAL_RESET_SEC:
            print(f"[BT 워치독] 카운트 만료 리셋 ({_bt_abnormal_count} → 0)")
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
        if _bt_abnormal_count > 0:
            print(f"[BT 워치독] 정상 ACK 수신 — 카운트 리셋 ({_bt_abnormal_count} → 0)")
        _bt_abnormal_count     = 0
        _bt_last_abnormal_time = 0.0

def _bt_restart():
    global sock, is_running
    print(f"\n🔁 [BT 워치독] 비정상 응답 {BT_ABNORMAL_THRESHOLD}회 도달 → 프로세스 재시작\n")
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
    print(f"\n{'='*50}")
    print(f"⚠️  [시뮬레이션 모드] 블루투스 연결 실패 — ESP32 없이 동작")
    print(f"   원인: {reason}")
    print(f"   백그라운드에서 {BT_RECONNECT_INTERVAL}초마다 재연결 시도 중...")
    print(f"{'='*50}\n")

def _exit_sim_mode():
    global sim_mode
    with sim_mode_lock:
        if not sim_mode: return
        sim_mode = False
    print(f"\n{'='*50}")
    print(f"✅ [시뮬레이션 모드 종료] 블루투스 재연결 성공 — 실제 모드로 전환")
    print(f"{'='*50}\n")
    _sim_log_buf.clear()

def _sim_log(state: str, perclos: float):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] 상태={state}  PERCLOS={perclos:.1%}"
    _sim_log_buf.append(entry)
    print(f"📋 [SIM] {entry}")

def _create_socket():
    try:
        s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        s.settimeout(BT_TIMEOUT)
        s.connect((ESP32_MAC_ADDR, BT_CHANNEL))
        print(f"✅ 블루투스 연결 성공: {ESP32_MAC_ADDR}")
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
                time.sleep(0.05)
                try:
                    res = sock.recv(1024).decode("utf-8", errors="ignore").strip()
                    if "A" in res:
                        _bt_record_normal()
                        return True
                    elif res:
                        _bt_record_abnormal(f"비정상 응답: '{res}'")
                    else:
                        _bt_record_abnormal("빈 응답(empty)")
                except socket.timeout:
                    _bt_record_abnormal("응답 타임아웃")
            except Exception as e:
                print(f"BT 전송 에러 (재연결 필요): {e}")
                try: sock.close()
                except: pass
                sock = None
                return False
            time.sleep(0.05)
    return False

def bluetooth_thread():
    """자동 재연결 및 확실한 하트비트 신호(!H#) 주기적 송신"""
    global sock, is_running, target_state
    current_state  = "OFF"
    last_heartbeat = time.time()
    last_reconnect = 0.0

    while is_running:
        with sock_lock:
            is_connected = sock is not None
        
        # ── 재연결 체크 ──
        if not is_connected:
            now = time.time()
            if now - last_reconnect >= BT_RECONNECT_INTERVAL:
                print("🔄 블루투스 재연결 시도...")
                new_sock = _create_socket()
                with sock_lock:
                    sock = new_sock
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

        # ── 상태 변경 시 명령 우선 전송 ──
        if cmd != current_state or current_state == "FORCE_RESEND":
            success = _send_command(cmd)
            if success:
                print(f">> BT 상태 전송 완료: {current_state} → {cmd}")
                current_state  = cmd
                last_heartbeat = time.time()

        # ── 주기적 하트비트(!H#) 신호 전송 ──
        now = time.time()
        if now - last_heartbeat >= BT_HEARTBEAT_INTERVAL:
            with sock_lock:
                s = sock
            if s:
                try:
                    s.send("!H#\n".encode("utf-8"))
                    time.sleep(0.05)
                    try:
                        res = s.recv(1024).decode("utf-8", errors="ignore").strip()
                        if "A" in res:
                            _bt_record_normal()
                        elif res:
                            _bt_record_abnormal(f"하트비트 비정상 응답: '{res}'")
                    except socket.timeout:
                        # 하트비트 타임아웃은 유연하게 처리하되 로그만 가볍게 남김
                        pass
                except Exception:
                    with sock_lock:
                        try: sock.close()
                        except: pass
                        sock = None
            last_heartbeat = now

        time.sleep(0.05)

# ══════════════════════════════════════════
# 3. EAR 및 알고리즘 클래스들
# ══════════════════════════════════════════
def calculate_ear(eye_pts: np.ndarray) -> float:
    v1 = dist.euclidean(eye_pts[1], eye_pts[5])
    v2 = dist.euclidean(eye_pts[2], eye_pts[4])
    h  = dist.euclidean(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

class EMAFilter:
    def __init__(self, alpha: float):
        self.alpha  = alpha
        self._value = None

    def update(self, v: float) -> float:
        self._value = v if self._value is None else self.alpha * v + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self):
        return self._value if self._value is not None else 0.0

class PerclosCalculator:
    def __init__(self, fps: float, window_sec: float, eye_ratio: float):
        self.history   = deque(maxlen=int(fps * window_sec))
        self.eye_ratio = eye_ratio

    def update(self, ear: float, threshold: float) -> float:
        self.history.append(ear)
        if len(self.history) < 10:
            return 0.0
        closed = sum(1 for e in self.history if e < threshold * self.eye_ratio)
        return closed / len(self.history)

    def fast_recover(self, ear: float, n: int = 10):
        for _ in range(n):
            self.history.append(ear)

class AdaptiveThreshold:
    def __init__(self, calib_sec, recalib_interval, recalib_window_sec, fps, max_drift):
        self.calib_sec        = calib_sec
        self.recalib_interval = recalib_interval
        self.max_drift        = max_drift
        self.long_term_buf    = deque(maxlen=int(fps * recalib_window_sec))

        self.calib_buf     = []
        self.threshold     = None
        self.baseline      = None
        self.is_calibrated = False
        self._calib_start  = None
        self._last_recalib = None

    @staticmethod
    def _iqr_filter(data: list, factor: float) -> list:
        if len(data) < 4: return data
        arr = np.array(data)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr    = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        filtered = arr[(arr >= lo) & (arr <= hi)].tolist()
        removed  = len(data) - len(filtered)
        if removed:
            print(f"[캘리브레이션] IQR 필터: {removed}개 이상치(감은 눈 등) 제거 (유효 범위 {lo:.3f} ~ {hi:.3f})")
        return filtered if filtered else data

    def update(self, ear: float, perclos: float) -> bool:
        now = time.time()

        if not self.is_calibrated:
            if self._calib_start is None:
                self._calib_start = now
            if now - self._calib_start < self.calib_sec:
                if ear > 0:
                    self.calib_buf.append(ear)
                return False

            clean          = self._iqr_filter(self.calib_buf, CALIB_IQR_FACTOR)
            self.baseline  = float(np.mean(clean))
            self.threshold = self.baseline * 0.75
            self.is_calibrated = True
            self._last_recalib = now
            print(f"[캘리브레이션 완료] 샘플={len(clean)}, baseline={self.baseline:.4f}, threshold={self.threshold:.4f}")
            return True

        if perclos < 0.10 and ear > 0:
            self.long_term_buf.append(ear)

        if now - self._last_recalib >= self.recalib_interval:
            if len(self.long_term_buf) > 30:
                new_baseline = float(np.mean(self.long_term_buf))
                drift        = abs(new_baseline - self.baseline) / self.baseline
                if drift <= self.max_drift:
                    self.baseline  = new_baseline
                    self.threshold = self.baseline * 0.75
                    print(f"[재보정 완료] baseline={self.baseline:.4f}, threshold={self.threshold:.4f}")
            self._last_recalib = now

        return True

    def calib_remaining(self) -> float:
        if self._calib_start is None: return self.calib_sec
        return max(0.0, self.calib_sec - (time.time() - self._calib_start))

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
            print(f"[워치독] {signal_type} 카운트 만료 초기화")
            self._counts[signal_type]   = 0
            self._last_err[signal_type] = 0.0

    def _restart(self, trigger: str):
        print(f"\n🔁 [워치독] {trigger} 오류 임계값 도달 → 프로세스 자동 재시작\n")
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
# 5. 오버레이 및 메인 루프
# ══════════════════════════════════════════
STATE_COLORS = {
    "OFF"        : (50,  205, 50),
    "LV1_WARN"   : (0,   165, 255),
    "LV2_DANGER" : (0,   0,   255),
}

def draw_overlay(frame, state, fps, perclos, ear, threshold,
                 is_calibrated, calib_rem, bt_ok, wd_counts):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS {int(fps)}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    with sim_mode_lock:
        is_sim = sim_mode
    if is_sim:
        blink_on = int(time.time() * 2) % 2 == 0
        cv2.putText(frame, "SIM MODE", (w - 160, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255) if blink_on else (0, 120, 180), 2)
    else:
        cv2.putText(frame, "BT:ON" if bt_ok else "BT:OFF", (w - 120, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 205, 50) if bt_ok else (0, 0, 255), 2)

    if not is_calibrated:
        cv2.putText(frame, f"CALIBRATING... {calib_rem:.1f}s",
                    (w // 2 - 180, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2)
        return

    cv2.putText(frame, state, (w // 2 - 120, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, STATE_COLORS.get(state, (255, 255, 255)), 3)

    bar_y = h - 100
    cv2.rectangle(frame, (0, bar_y - 5), (w, h), (20, 20, 20), -1)

    pbar_w = int(min(perclos, 1.0) * (w - 20))
    cv2.rectangle(frame, (10, bar_y + 5), (10 + pbar_w, bar_y + 22), STATE_COLORS.get(state, (50, 205, 50)), -1)
    cv2.rectangle(frame, (10, bar_y + 5), (w - 10, bar_y + 22), (120, 120, 120), 1)
    for ratio, lv_color in [(PERCLOS_LV1, (0, 165, 255)), (PERCLOS_LV2, (0, 0, 255))]:
        lx = int(ratio * (w - 20)) + 10
        cv2.line(frame, (lx, bar_y + 3), (lx, bar_y + 24), lv_color, 2)
        
    cv2.putText(frame, f"PERCLOS {perclos:.1%}", (10, bar_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"EAR {ear:.3f}  THR {threshold:.3f}", (10, bar_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    wd_text_parts = []
    label_map = {"CAM_FAIL": "CAM", "EAR_INVALID": "EAR", "PROCESS_EXCEPT": "EXC"}
    for sig, cnt in wd_counts.items():
        if cnt > 0:
            wd_text_parts.append(f"{label_map[sig]}:{cnt}/{WATCHDOG_THRESHOLD}")
    if wd_text_parts:
        cv2.putText(frame, "WD " + "  ".join(wd_text_parts), (10, bar_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

def draw_no_face(frame, is_calibrated, missing_sec):
    if not is_calibrated:
        cv2.putText(frame, "WAITING FOR FACE...", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2)
        return
    if missing_sec >= FACE_MISSING_DANGER_SEC:
        cv2.putText(frame, "NO FACE — DANGER!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(frame, f"FACE MISSING... {int(FACE_MISSING_DANGER_SEC - missing_sec)}s",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

def main():
    global target_state, is_running

    check_opencv_gstreamer()
    vs = CSICameraStream().start()
    time.sleep(2.0)
    print("✅ CSI 카메라 파이프라인 활성화 완료")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ema          = EMAFilter(EMA_ALPHA)
    perclos_calc = PerclosCalculator(FPS_TARGET, PERCLOS_WINDOW_SEC, PERCLOS_EYE_RATIO)
    adaptive_thr = AdaptiveThreshold(CALIB_DURATION_SEC, RECALIB_INTERVAL, RECALIB_WINDOW_SEC, FPS_TARGET, RECALIB_MAX_DRIFT)
    watchdog     = SignalWatchdog()

    state                 = "OFF"
    prev_state            = "OFF"
    face_missing_start    = 0.0
    instant_closed_frames = 0
    prev_time             = time.time()

    print("── 시스템 메인 감시 루프 가동 (종료 키: q) ──")

    try:
        while True:
            frame = vs.read()

            # ── 카메라 프레임 드롭 체크 ──
            if frame is None or vs.consecutive_fail > WATCHDOG_CAM_FAIL_LIMIT:
                watchdog.record("CAM_FAIL")
                instant_closed_frames = 0
                time.sleep(0.01)
                continue
            else:
                watchdog.clear("CAM_FAIL")

            curr_time = time.time()
            fps       = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # ① 화질을 고해상도(640, 360)로 스케일 업 및 랜드마크 분석 이미지 생성
            small = cv2.resize(frame, (PROCESS_W, PROCESS_H))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            try:
                results = face_mesh.process(rgb)
            except Exception as e:
                print(f"⚠️  MediaPipe core 예외 발생: {e}")
                watchdog.record("PROCESS_EXCEPT")
                continue

            ema_ear = ema.value
            perclos = 0.0

            if results.multi_face_landmarks:
                face_missing_start = 0.0

                # 주 운전자 (가장 전면에 위치해 면적이 큰 얼굴) 탐색
                largest_face, max_area = None, 0.0
                for face_lms in results.multi_face_landmarks:
                    xs = [lm.x for lm in face_lms.landmark]
                    ys = [lm.y for lm in face_lms.landmark]
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    if area > max_area:
                        max_area, largest_face = area, face_lms

                if largest_face:
                    lms = largest_face.landmark
                    
                    # ⚠️ [핵심 버그 수정] 픽셀 복원 시 원본(w, h)이 아닌 MediaPipe 입력 해상도(PROCESS_W, PROCESS_H)를 곱해야 정확히 정렬됨
                    left_pts  = np.array([(lms[i].x * PROCESS_W, lms[i].y * PROCESS_H) for i in LEFT_EYE])
                    right_pts = np.array([(lms[i].x * PROCESS_W, lms[i].y * PROCESS_H) for i in RIGHT_EYE])
                    
                    raw_ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                    # ── EAR 값 물리적 바운더리 검사 ──
                    if raw_ear <= 0.0 or raw_ear >= 1.0:
                        watchdog.record("EAR_INVALID")
                    else:
                        watchdog.clear("EAR_INVALID")
                        watchdog.clear("PROCESS_EXCEPT")

                        ema_ear = ema.update(raw_ear)
                        is_cal  = adaptive_thr.update(ema_ear, perclos)

                        if is_cal and adaptive_thr.threshold:
                            perclos = perclos_calc.update(ema_ear, adaptive_thr.threshold)

                            # 즉각 트리거 조건 연산
                            if ema_ear < adaptive_thr.threshold * INSTANT_RATIO:
                                instant_closed_frames += 1
                            else:
                                if instant_closed_frames >= INSTANT_LV1_FRAMES:
                                    perclos_calc.fast_recover(ema_ear, n=10)
                                instant_closed_frames = 0

                            # 최종 상태 결정 판별식
                            if instant_closed_frames >= INSTANT_LV2_FRAMES or perclos >= PERCLOS_LV2:
                                state = "LV2_DANGER"
                            elif instant_closed_frames >= INSTANT_LV1_FRAMES or perclos >= PERCLOS_LV1:
                                state = "LV1_WARN"
                            else:
                                state = "OFF"
                        else:
                            state = "OFF"

                with sock_lock:
                    bt_ok = sock is not None

                draw_overlay(small, state, fps, perclos, ema_ear,
                             adaptive_thr.threshold or 0.0,
                             adaptive_thr.is_calibrated, adaptive_thr.calib_remaining(),
                             bt_ok, watchdog.counts())
            else:
                # 얼굴 유실 예외 처리
                instant_closed_frames = 0
                adaptive_thr.update(0.0, 0.0)

                if face_missing_start == 0.0:
                    face_missing_start = curr_time
                missing_sec = curr_time - face_missing_start

                state = ("LV2_DANGER" if adaptive_thr.is_calibrated and missing_sec >= FACE_MISSING_DANGER_SEC else "OFF")

                with sock_lock:
                    bt_ok = sock is not None

                draw_no_face(small, adaptive_thr.is_calibrated, missing_sec)
                draw_overlay(small, state, fps, 0.0, 0.0, 0.0,
                             adaptive_thr.is_calibrated, adaptive_thr.calib_remaining(),
                             bt_ok, watchdog.counts())

            # ── 상태 전송용 인터페이스 트리거 ──
            if state != prev_state:
                print(f"[상태 변경 트리거] {prev_state} → {state} (PERCLOS={perclos:.1%})")
                with sim_mode_lock:
                    is_sim = sim_mode
                if is_sim:
                    _sim_log(state, perclos)
                else:
                    with bt_lock:
                        target_state = state
                prev_state = state

            cv2.imshow("Jetson Drowsiness System", small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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
        print("── 졸음 감지 시스템이 정상 안전 종료되었습니다 ──")

if __name__ == "__main__":
    print("=== 젯슨 나노 졸음 감지 시스템 가동 ===")
    _init_bluetooth()
    threading.Thread(target=bluetooth_thread, daemon=True).start()
    main()
