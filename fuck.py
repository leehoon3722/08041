"""
졸음 감지 시스템 — 젯슨 나노 최종 버전
=========================================
[보완 사항]
  ① 블루투스 자동 재연결     : 끊어지면 BT_RECONNECT_INTERVAL초마다 재시도
  ② 캘리브레이션 이상값 필터 : IQR 기반으로 눈 감은 프레임 제거 후 baseline 계산
  ③ 신호 워치독              : 잘못된 신호 3회 이상 → 프로세스 자동 재시작

[EAR 알고리즘]
  - EMA 스무딩 / PERCLOS / 즉각 트리거 / 적응형 임계값
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

# CSI 카메라
SENSOR_ID    = 0
CAP_WIDTH    = 1280
CAP_HEIGHT   = 720
DISP_WIDTH   = 640
DISP_HEIGHT  = 480
FPS_TARGET   = 30
FLIP_METHOD  = 0

# 블루투스
FORCE_SIM_MODE        = True   # True: BT 시도 없이 즉시 시뮬레이션 모드 진입
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
CALIB_IQR_FACTOR   = 1.5   # ② IQR × 이 값 이내만 유효 샘플로 인정

# 적응형 재보정
RECALIB_INTERVAL   = 120.0
RECALIB_WINDOW_SEC = 10.0
RECALIB_MAX_DRIFT  = 0.10

# 얼굴 없음 → 위험
FACE_MISSING_DANGER_SEC = 5.0

# BT 통신 실패 재시작
# --restarted 인자가 있으면 이미 1회 재시작된 것 → 시뮬레이션 모드로 전환
BT_ALREADY_RESTARTED = "--restarted" in sys.argv

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
# 2. 블루투스 — ① 자동 재연결 포함
# ══════════════════════════════════════════
sock         = None
sock_lock    = threading.Lock()
target_state = "OFF"
is_running   = True
bt_lock      = threading.Lock()

# 시뮬레이션 모드
sim_mode      = False         # BT 연결 실패 시 True로 전환
sim_mode_lock = threading.Lock()
_sim_log_buf  = []            # 시뮬레이션 상태 변경 기록

# ESP32 비정상 응답 카운터
_bt_abnormal_count    = 0
_bt_last_abnormal_time = 0.0
_bt_abnormal_lock     = threading.Lock()

def _bt_record_abnormal(reason: str):
    """비정상 응답 1회 기록. BT_ABNORMAL_THRESHOLD 도달 시 프로세스 재시작."""
    global _bt_abnormal_count, _bt_last_abnormal_time
    with _bt_abnormal_lock:
        now = time.time()
        # 마지막 이상 신호 후 BT_ABNORMAL_RESET_SEC 경과 시 카운트 리셋
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
    """정상 ACK 수신 시 카운트 리셋."""
    global _bt_abnormal_count, _bt_last_abnormal_time
    with _bt_abnormal_lock:
        if _bt_abnormal_count > 0:
            print(f"[BT 워치독] 정상 ACK 수신 — 카운트 리셋 ({_bt_abnormal_count} → 0)")
        _bt_abnormal_count     = 0
        _bt_last_abnormal_time = 0.0

def _bt_restart():
    """BT 비정상 응답 누적 → 프로세스 재시작 (1회만). 이미 재시작된 경우 시뮬레이션 모드."""
    global sock, is_running
    if BT_ALREADY_RESTARTED:
        # 재시작 후에도 BT 실패 → 시뮬레이션 모드로 전환 (무한루프 방지)
        print("⚠️  [BT] 재시작 후에도 통신 실패 → 시뮬레이션 모드로 전환")
        _enter_sim_mode("재시작 후에도 BT 통신 실패")
        return
    print(f"\n🔁 [BT] 비정상 응답 {BT_ABNORMAL_THRESHOLD}회 → 프로세스 재시작\n")
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
    os.execv(sys.executable, [sys.executable] + sys.argv + ["--restarted"])

def _enter_sim_mode(reason: str):
    """BT 연결 불가 → 시뮬레이션 모드 진입."""
    global sim_mode
    with sim_mode_lock:
        if sim_mode:
            return
        sim_mode = True
    print(f"\n{'='*50}")
    print(f"⚠️  [시뮬레이션 모드] 블루투스 연결 실패 — ESP32 없이 동작")
    print(f"   원인: {reason}")
    print(f"   상태 변경은 콘솔에 출력됩니다.")
    print(f"   백그라운드에서 {BT_RECONNECT_INTERVAL}초마다 재연결 시도 중...")
    print(f"{'='*50}\n")

def _exit_sim_mode():
    """BT 재연결 성공 → 실제 모드로 복귀."""
    global sim_mode
    with sim_mode_lock:
        if not sim_mode:
            return
        sim_mode = False
    print(f"\n{'='*50}")
    print(f"✅ [시뮬레이션 모드 종료] 블루투스 재연결 성공 — 실제 모드로 전환")
    if _sim_log_buf:
        print(f"   시뮬레이션 중 상태 변경 기록 ({len(_sim_log_buf)}건):")
        for entry in _sim_log_buf[-10:]:   # 최근 10건만 출력
            print(f"     {entry}")
    print(f"{'='*50}\n")
    _sim_log_buf.clear()

def _sim_log(state: str, perclos: float):
    """시뮬레이션 모드에서 상태 변경을 콘솔 + 버퍼에 기록."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] 상태={state}  PERCLOS={perclos:.1%}"
    _sim_log_buf.append(entry)
    print(f"📋 [SIM] {entry}")

def _create_socket():
    """소켓 생성 + 연결 시도. 실패 시 None 반환."""
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
    if FORCE_SIM_MODE:
        _enter_sim_mode("강제 시뮬레이션 모드 (FORCE_SIM_MODE=True)")
        return
    with sock_lock:
        sock = _create_socket()
    if sock is None:
        _enter_sim_mode("초기 연결 실패")

def _send_command(cmd_text: str) -> bool:
    """
    명령 전송 + ESP32 응답 판별.
    - 'A'  포함 → 정상 ACK  → 카운트 리셋, True 반환
    - 'E'  포함 → ESP32 에러 응답 → 비정상 카운트 +1
    - 빈 응답/타임아웃 → 비정상 카운트 +1
    - 소켓 예외 → sock=None 후 False 반환
    """
    global sock
    with sock_lock:
        if not sock:
            return False
        packet = f"!{cmd_text}#\n".encode("utf-8")
        for _ in range(BT_ACK_RETRY):
            if not is_running:
                return False
            try:
                sock.send(packet)
                time.sleep(0.1)
                try:
                    res = sock.recv(1024).decode("utf-8", errors="ignore").strip()
                    if "A" in res:
                        # ── 정상 ACK ──
                        _bt_record_normal()
                        return True
                    elif res:
                        # ── ESP32가 응답은 했지만 ACK가 아님 (에러 코드 등) ──
                        _bt_record_abnormal(f"비정상 응답: '{res}'")
                    else:
                        # ── 빈 응답 ──
                        _bt_record_abnormal("빈 응답(empty)")
                except socket.timeout:
                    # ── 응답 없음 ──
                    _bt_record_abnormal("응답 타임아웃")
            except Exception as e:
                print(f"BT 전송 에러 (재연결 필요): {e}")
                try: sock.close()
                except: pass
                sock = None
                return False
            time.sleep(0.1)
    return False

def bluetooth_thread():
    """① 자동 재연결: sock이 None이면 BT_RECONNECT_INTERVAL마다 재시도"""
    global sock, is_running
    current_state  = "OFF"
    last_heartbeat = time.time()
    last_reconnect = 0.0

    while is_running:
        # ── ① 재연결 체크 ──
        with sock_lock:
            is_connected = sock is not None
        if not is_connected:
            now = time.time()
            if now - last_reconnect >= BT_RECONNECT_INTERVAL:
                if FORCE_SIM_MODE:
                    last_reconnect = now   # 재연결 시도 없이 대기만
                else:
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

        # ── 상태 변경 시 전송 ──
        if cmd != current_state or current_state == "FORCE_RESEND":
            success = _send_command(cmd)
            if success:
                print(f">> BT: {current_state} → {cmd}")
                current_state  = cmd
                last_heartbeat = time.time()

        # ── 하트비트 ──
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
                        # 타임아웃은 하트비트에서 허용 (카운트 안 함)
                    except socket.timeout:
                        pass
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
# 4. EMA 스무딩
# ══════════════════════════════════════════
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

# ══════════════════════════════════════════
# 5. PERCLOS 계산기
# ══════════════════════════════════════════
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
        """눈이 다시 열렸을 때 현재 EAR로 N프레임을 채워 히스토리를 빠르게 희석."""
        for _ in range(n):
            self.history.append(ear)

# ══════════════════════════════════════════
# 6. 적응형 임계값 — ② IQR 캘리브레이션 필터 포함
# ══════════════════════════════════════════
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
        """IQR 기반 이상치 제거 — 눈 감은 프레임의 낮은 EAR 값 제거"""
        if len(data) < 4:
            return data
        arr = np.array(data)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr    = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        filtered = arr[(arr >= lo) & (arr <= hi)].tolist()
        removed  = len(data) - len(filtered)
        if removed:
            print(f"[캘리브레이션] IQR 필터: {removed}개 이상치 제거 "
                  f"(유효 범위 {lo:.3f} ~ {hi:.3f})")
        return filtered if filtered else data

    def update(self, ear: float, perclos: float) -> bool:
        now = time.time()

        # ── 초기 캘리브레이션 수집 ──
        if not self.is_calibrated:
            if self._calib_start is None:
                self._calib_start = now
            if now - self._calib_start < self.calib_sec:
                if ear > 0:
                    self.calib_buf.append(ear)
                return False

            # ② IQR 필터 후 baseline 계산
            clean          = self._iqr_filter(self.calib_buf, CALIB_IQR_FACTOR)
            self.baseline  = float(np.mean(clean))
            self.threshold = self.baseline * 0.75
            self.is_calibrated = True
            self._last_recalib = now
            print(f"[캘리브레이션 완료] 샘플={len(clean)}, "
                  f"baseline={self.baseline:.4f}, threshold={self.threshold:.4f}")
            return True

        # ── 장기 버퍼 축적 (정상 상태일 때만) ──
        if perclos < 0.10 and ear > 0:
            self.long_term_buf.append(ear)

        # ── 주기적 재보정 ──
        if now - self._last_recalib >= self.recalib_interval:
            if len(self.long_term_buf) > 30:
                new_baseline = float(np.mean(self.long_term_buf))
                drift        = abs(new_baseline - self.baseline) / self.baseline
                if drift <= self.max_drift:
                    self.baseline  = new_baseline
                    self.threshold = self.baseline * 0.75
                    print(f"[재보정] baseline={self.baseline:.4f}, "
                          f"threshold={self.threshold:.4f}")
                else:
                    print(f"[재보정 스킵] drift={drift:.2%} > 허용 {self.max_drift:.2%}")
            self._last_recalib = now

        return True

    def calib_remaining(self) -> float:
        if self._calib_start is None:
            return self.calib_sec
        return max(0.0, self.calib_sec - (time.time() - self._calib_start))

    def reset(self):
        """새 탑승자를 위한 캘리브레이션 초기화 (R 키)."""
        self.calib_buf     = []
        self.long_term_buf.clear()
        self.threshold     = None
        self.baseline      = None
        self.is_calibrated = False
        self._calib_start  = None
        self._last_recalib = None
        print("[캘리브레이션 초기화] 새 탑승자 캘리브레이션 시작")



# ══════════════════════════════════════════
# 7. 화면 오버레이
# ══════════════════════════════════════════
STATE_COLORS = {
    "OFF"        : (50,  205, 50),
    "LV1_WARN"   : (0,   165, 255),
    "LV2_DANGER" : (0,   0,   255),
}

def draw_overlay(frame, state, fps, perclos, ear, threshold,
                 is_calibrated, calib_rem, bt_ok):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS {int(fps)}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    # 블루투스 / 시뮬레이션 상태 표시
    with sim_mode_lock:
        is_sim = sim_mode
    if is_sim:
        # 시뮬레이션 모드: 깜박이는 노란 배지
        blink_on = int(time.time() * 2) % 2 == 0
        cv2.putText(frame, "SIM MODE", (w - 160, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 200, 255) if blink_on else (0, 120, 180), 2)
    else:
        cv2.putText(frame, "BT:ON" if bt_ok else "BT:OFF", (w - 120, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (50, 205, 50) if bt_ok else (0, 0, 255), 2)

    if not is_calibrated:
        cv2.putText(frame, f"CALIBRATING... {calib_rem:.1f}s",
                    (w // 2 - 180, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2)
        return

    # 상태
    cv2.putText(frame, state, (w // 2 - 120, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, STATE_COLORS.get(state, (255, 255, 255)), 3)

    # 하단 정보 바
    bar_y = h - 100
    cv2.rectangle(frame, (0, bar_y - 5), (w, h), (20, 20, 20), -1)

    # PERCLOS 게이지
    pbar_w = int(min(perclos, 1.0) * (w - 20))
    cv2.rectangle(frame, (10, bar_y + 5),
                  (10 + pbar_w, bar_y + 22), STATE_COLORS.get(state, (50, 205, 50)), -1)
    cv2.rectangle(frame, (10, bar_y + 5), (w - 10, bar_y + 22), (120, 120, 120), 1)
    for ratio, lv_color in [(PERCLOS_LV1, (0, 165, 255)), (PERCLOS_LV2, (0, 0, 255))]:
        lx = int(ratio * (w - 20)) + 10
        cv2.line(frame, (lx, bar_y + 3), (lx, bar_y + 24), lv_color, 2)
    cv2.putText(frame, f"PERCLOS {perclos:.1%}",
                (10, bar_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # EAR / Threshold
    cv2.putText(frame, f"EAR {ear:.3f}  THR {threshold:.3f}",
                (10, bar_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # ③ 워치독 카운트 표시 (오류 1회 이상일 때만)
    wd_text_parts = []
    label_map = {"CAM_FAIL": "CAM", "EAR_INVALID": "EAR", "PROCESS_EXCEPT": "EXC"}
    for sig, cnt in wd_counts.items():
        if cnt > 0:
            wd_text_parts.append(f"{label_map[sig]}:{cnt}/{WATCHDOG_THRESHOLD}")
    if wd_text_parts:
        wd_text  = "WD " + "  ".join(wd_text_parts)
        wd_color = (0, 165, 255) if max(wd_counts.values()) < WATCHDOG_THRESHOLD else (0, 0, 255)
        cv2.putText(frame, wd_text, (10, bar_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, wd_color, 1)

    # R키 안내
    cv2.putText(frame, "R: recalibrate", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
    h, w = frame.shape[:2]
    if not is_calibrated:
        cv2.putText(frame, "WAITING FOR FACE...", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2)
        return
    if missing_sec >= FACE_MISSING_DANGER_SEC:
        cv2.putText(frame, "NO FACE — DANGER!", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(frame, f"FACE MISSING... {int(FACE_MISSING_DANGER_SEC - missing_sec)}s",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

# ══════════════════════════════════════════
# 9. 메인 루프
# ══════════════════════════════════════════
def main():
    global target_state, is_running

    check_opencv_gstreamer()

    vs = CSICameraStream().start()
    time.sleep(2.0)
    print("✅ CSI 카메라 시작")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    ema          = EMAFilter(EMA_ALPHA)
    perclos_calc = PerclosCalculator(FPS_TARGET, PERCLOS_WINDOW_SEC, PERCLOS_EYE_RATIO)
    adaptive_thr = AdaptiveThreshold(
        CALIB_DURATION_SEC, RECALIB_INTERVAL,
        RECALIB_WINDOW_SEC, FPS_TARGET, RECALIB_MAX_DRIFT,
    )

    state                 = "OFF"
    prev_state            = "OFF"
    face_missing_start    = 0.0
    instant_closed_frames = 0
    prev_time             = time.time()

    print("── 시작 (종료: q) ──")

    try:
        while True:
            frame = vs.read()
            if frame is None:
                instant_closed_frames = 0
                time.sleep(0.01)
                continue

            curr_time = time.time()
            fps       = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            h, w, _ = frame.shape
            small   = cv2.resize(frame, (640, 480))
            rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            try:
                results = face_mesh.process(rgb)
            except Exception as e:
                print(f"⚠️  MediaPipe 처리 예외: {e}")
                continue

            ema_ear = ema.value
            perclos = 0.0

            if results.multi_face_landmarks:
                face_missing_start = 0.0

                # 가장 큰 얼굴(운전자) 선택
                largest_face, max_area = None, 0.0
                for face_lms in results.multi_face_landmarks:
                    xs   = [lm.x for lm in face_lms.landmark]
                    ys   = [lm.y for lm in face_lms.landmark]
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    if area > max_area:
                        max_area, largest_face = area, face_lms

                if largest_face:
                    lms       = largest_face.landmark
                    left_pts  = np.array([(lms[i].x * w, lms[i].y * h) for i in LEFT_EYE])
                    right_pts = np.array([(lms[i].x * w, lms[i].y * h) for i in RIGHT_EYE])
                    raw_ear   = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                    # 물리적으로 불가능한 EAR은 프레임만 건너뜀 (재시작 아님)
                    if raw_ear <= 0.0 or raw_ear >= 1.0:
                        continue

                    watchdog.clear("PROCESS_EXCEPT")
                    ema_ear = ema.update(raw_ear)
                    is_cal  = adaptive_thr.update(ema_ear, perclos)

                    if is_cal and adaptive_thr.threshold:
                        perclos = perclos_calc.update(ema_ear, adaptive_thr.threshold)

                        # 즉각 트리거
                        if ema_ear < adaptive_thr.threshold * INSTANT_RATIO:
                            instant_closed_frames += 1
                        else:
                            # 경보 상태에서 눈이 다시 열렸으면 PERCLOS 히스토리 빠르게 희석
                            if instant_closed_frames >= INSTANT_LV1_FRAMES:
                                perclos_calc.fast_recover(ema_ear, n=10)
                            instant_closed_frames = 0

                        # 상태 판별
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

                draw_overlay(
                    frame, state, fps, perclos, ema_ear,
                    adaptive_thr.threshold or 0.0,
                    adaptive_thr.is_calibrated,
                    adaptive_thr.calib_remaining(),
                    bt_ok,
                )

            else:
                instant_closed_frames = 0
                adaptive_thr.update(0.0, 0.0)

                if face_missing_start == 0.0:
                    face_missing_start = curr_time
                missing_sec = curr_time - face_missing_start

                state = ("LV2_DANGER"
                         if adaptive_thr.is_calibrated and missing_sec >= FACE_MISSING_DANGER_SEC
                         else "OFF")

                with sock_lock:
                    bt_ok = sock is not None

                draw_no_face(frame, adaptive_thr.is_calibrated, missing_sec)
                draw_overlay(
                    frame, state, fps, 0.0, 0.0, 0.0,
                    adaptive_thr.is_calibrated,
                    adaptive_thr.calib_remaining(),
                    bt_ok,
                )

            # 블루투스 상태 동기화
            if state != prev_state:
                print(f"[상태 변경] {prev_state} → {state}  (PERCLOS={perclos:.1%})")
                with sim_mode_lock:
                    is_sim = sim_mode
                if is_sim:
                    _sim_log(state, perclos)   # 시뮬레이션: 콘솔 기록
                else:
                    with bt_lock:              # 실제 모드: ESP32 전송
                        target_state = state
                prev_state = state

            cv2.imshow("Jetson Drowsiness System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # 새 탑승자 → 캘리브레이션 전체 리셋
                adaptive_thr.reset()
                ema._value            = None
                perclos_calc.history.clear()
                instant_closed_frames = 0
                state                 = "OFF"
                prev_state            = "OFF"
                face_missing_start    = 0.0
                with bt_lock:
                    target_state = "OFF"
                print("🔄 [R키] 새 탑승자 캘리브레이션 시작 — 정면을 바라봐 주세요")

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

# ══════════════════════════════════════════
# 10. 진입점
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("=== 젯슨 나노 졸음 감지 시스템 (보완 버전) 시작 ===")
    _init_bluetooth()
    threading.Thread(target=bluetooth_thread, daemon=True).start()
    main()
