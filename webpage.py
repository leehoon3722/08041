"""
졸음 감지 시스템 — 젯슨 나노 GPU 가속 버전 (TensorRT 직접 실행)
==============================================================
[추론 파이프라인]
  MediaPipe CPU 추론 → TensorRT GPU 추론
  (onnxruntime · torch2trt · PyTorch · pycuda 모두 불필요 — JetPack 기본 구성만 사용)
  Stage 1: Ultra-Light 얼굴 검출 (version-RFB-320)   얼굴 bbox        (320×240 RGB 입력)
  Stage 2: 68점 랜드마크 (MobileNet-SE)              얼굴 정사각 크롭 (56×56 RGB 입력)
  · 첫 실행 때 ONNX → TensorRT FP16 엔진으로 변환해 ./trt_cache/ 에 저장 (모델당 수 분)
  · 다음 실행부터는 저장된 엔진을 바로 로드

[사전 준비]
  # 1) TensorRT 파이썬 모듈 확인 — JetPack에 이미 들어 있음 (pip 설치 필요 없음)
  python3 -c "import tensorrt; print(tensorrt.__version__)"
  # 2) 모델 다운로드
  wget https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx
  wget https://github.com/cunjian/pytorch_face_landmark/raw/master/onnx/landmark_detection_56_se_external.onnx

[얼굴 검출 엔진 변환이 실패하면]
  후처리(박스 디코딩)를 뺀 모델로 교체 → 디코딩은 CPU(numpy)가 처리
  wget https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320_without_postprocessing.onnx
  그리고 아래 FACE_DET_MODEL = "version-RFB-320_without_postprocessing.onnx"

[졸음 판정 — 시간 기준]
  EAR(눈 종횡비) → EMA 스무딩 → 운전자별 기준으로 '감김/뜸' 판정 (히스테리시스)
  · 연속 눈 감김 1초 이상 → LV1, 2초 이상 → LV2 (Euro NCAP 마이크로슬립 1~2초), 눈 뜨면 0.5초 뒤 해제
  · PERCLOS: 최근 30초 중 눈 감긴 시간 비율 15% 이상 → LV1, 30% 이상 → LV2 (누적 졸음 경향, 경고음은 2.5초만)
  · 얼굴이 5초 이상 안 보이면 LV2
  캘리브레이션 2단계: 정면 보고 눈 뜨기 3초 → 화면에 CLOSE EYES가 뜨면 눈 감기 2초
  → 뜬 눈·감은 눈 EAR의 중간을 감김 기준으로 (이 랜드마크 모델은 감아도 EAR이 0까지 안 내려감)

[머리 자세 → 전방 주시 태만]
  68점 중 6점(코끝·턱·눈꼬리·입꼬리) + solvePnP로 고개 방향 추정 → 캘리브레이션 때 잡은 '정면'과 비교
  · 한 번에 3초 이상 전방 이탈 → LV1, 6초 이상 → LV2
  · 최근 30초 중 이탈 누적 10초 이상 → LV1 (짧게 여러 번 보는 휴대폰 사용 패턴)
  · ESP32에는 기존과 같은 LV1_WARN / LV2_DANGER로 전달 → 펌웨어 수정 불필요

[경고음]
  LV1/LV2 경고음을 디스플레이 스피커(HDMI, aplay -D plughw:0,3)로 재생 — ESP32 부저 대신
  LV1: 낮은 음 '삑-삑' 반복, LV2: 높은 음 빠른 연속음, OFF가 되면 바로 멈춤

[카메라]
  USB 웹캠 (V4L2, /dev/video1, 640×480 MJPG 30fps 요청 → 실제 값은 실행 시 출력)
  · 적외선 카메라가 아니므로 어두운 곳(야간·터널)에선 얼굴 검출이 안 될 수 있음

[프레임 처리]
  카메라에서 새 프레임이 들어올 때만 처리 (같은 프레임 중복 처리 X)
  → 판정은 모두 '시간' 기준이라 FPS가 흔들려도 같은 기준으로 동작
  화면 왼쪽 위: FPS(실제 처리한 새 프레임 수) + 프레임당 처리 시간(ms)
"""

import sys
import os
import glob
import ctypes
import ctypes.util
import socket
import time
import threading
import shutil
import subprocess
import wave
import queue
from collections import deque

import cv2
import numpy as np
from scipy.spatial import distance as dist

try:
    import tensorrt as trt
except ImportError:
    print("🚨 [치명적 오류] tensorrt 파이썬 모듈을 찾을 수 없습니다.")
    print("   · JetPack 기본 python3(/usr/bin/python3)로 실행하세요.")
    print("   · 가상환경이라면 --system-site-packages 옵션으로 만들어야 tensorrt가 보입니다.")
    sys.exit(1)

# ══════════════════════════════════════════
# 0. 파라미터 설정
# ══════════════════════════════════════════

# USB 웹캠
CAM_INDEX    = 1        # /dev/video1 (웹캠). 다른 번호면 v4l2-ctl --list-devices 로 확인
CAM_WIDTH    = 640
CAM_HEIGHT   = 480
FPS_TARGET   = 30
CAM_FOURCC   = "MJPG"   # 대부분 웹캠은 MJPG여야 640×480 30fps가 나옴 (안 되면 "YUYV" 또는 None)
CAM_MIRROR   = True     # 화면을 거울처럼 좌우 반전 (판정에는 영향 없음)
CAM_KEEP_FPS = True     # 어두워도 노출 시간을 늘려 FPS를 떨어뜨리지 않게 (v4l2-ctl 사용)
CAM_EXPOSURE = None     # 수동 노출 (단위 0.1ms, 예: 250 = 25ms, 30fps면 333 이하). None = 자동 노출

# 블루투스
FORCE_SIM_MODE        = False  # True: BT 시도 없이 즉시 시뮬레이션 모드 진입
ESP32_MAC_ADDR        = "08:3A:F2:B9:8C:6A"
BT_CHANNEL            = 1
BT_TIMEOUT            = 5.0
BT_HEARTBEAT_INTERVAL = 1.0    # 하트비트 주기 (초)
BT_ACK_RETRY          = 10     # ACK 재시도 횟수
BT_RECONNECT_INTERVAL = 5.0    # 재연결 시도 간격 (초)
BT_ABNORMAL_THRESHOLD = 3      # ESP32 비정상 응답 N회 → 프로세스 재시작
BT_ABNORMAL_RESET_SEC = 30.0   # 이 시간 동안 정상 응답이면 카운트 리셋
BT_ACK_TIMEOUT        = 1.0    # 명령 1회당 ACK('A') 기다리는 시간 (초)
BT_HB_ACK_TIMEOUT     = 0.5    # 하트비트 ACK 기다리는 시간 (초) — 타임아웃은 카운트 안 함
BUZ_LV_ECHO_SEC       = 1.0    # LV1/LV2 명령 보낸 뒤 이 시간 안에 온 !BUZ# = ESP32가 LV 수신하고 보낸 것
BUZ_PLAY_ON_LV        = True   # LV 때 온 !BUZ#도 알림음 재생 (False: LV 경고음만 — 알림음이 끼어들지 않음)
BUZ_BANNER_SEC        = 3.5    # 핸들 이탈 !BUZ# 후 화면 'HANDS ON WHEEL' 표시 시간 (ESP32 반복 간격 3초보다 조금 길게)

# EMA 스무딩
EMA_ALPHA          = 0.40

# 졸음 판정 ① 연속 눈 감김 (시간 기준 — 급성, 마이크로슬립)
EYE_CLOSED_LV1_SEC = 1.0    # 연속 1초 이상 감음 → LV1 (Euro NCAP 마이크로슬립 1~2초)
EYE_CLOSED_LV2_SEC = 2.0    # 연속 2초 이상 감음 → LV2
ALARM_HOLD_SEC     = 0.5    # 눈을 다시 뜬 뒤 이 시간이 지나면 감김 경보 해제

# 졸음 판정 ② PERCLOS (누적 졸음 경향)
PERCLOS_WINDOW_SEC = 30.0   # 최근 30초 동안 (60초는 눈을 떠도 게이지가 너무 늦게 내려감)
PERCLOS_MIN_SEC    = 10.0   # 데이터가 10초 이상 쌓인 뒤부터 판정
PERCLOS_LV1        = 0.15   # 눈 감긴 시간 비율 15% 이상 → LV1
PERCLOS_LV2        = 0.30   # 30% 이상 → LV2
PERCLOS_SOUND_SEC  = 2.5    # PERCLOS 경보는 경고음을 이 시간만 울리고 멈춤 (화면·LED 경고는 유지)
                            # 연속 눈 감김·전방 주시 태만·얼굴 없음 경보는 지금처럼 계속 울림

# 캘리브레이션 — 2단계 (정면 보고 눈 뜨기 → 눈 감기)
CALIB_OPEN_SEC      = 3.0   # 뜬 눈 EAR 측정
CALIB_CLOSED_SEC    = 2.0   # 감은 눈 EAR 측정 (0이면 생략 → CLOSE_RATIO_DEFAULT 사용)
CALIB_SKIP_SEC      = 0.5   # 각 단계 시작 직후(안내 보고 반응하는 시간)는 버림
CALIB_IQR_FACTOR    = 1.5   # ② IQR × 이 값 이내만 유효 샘플로 인정
CLOSE_POINT         = 0.5   # 감김 기준 = 감은 눈 + 0.5 × (뜬 눈 − 감은 눈)  → 두 값의 중간
CLOSE_HYST          = 0.10  # 다시 뜸 기준 = 감김 기준 + 0.1 × (뜬 눈 − 감은 눈)  → 경계 깜빡임 방지
CLOSE_RATIO_DEFAULT = 0.75  # 감은 눈을 못 쟀을 때: 감김 기준 = 뜬 눈 × 0.75

# 적응형 재보정
RECALIB_INTERVAL   = 120.0
RECALIB_WINDOW_SEC = 10.0
RECALIB_MAX_DRIFT  = 0.10

# 얼굴 없음 → 위험
FACE_MISSING_DANGER_SEC = 5.0

# 머리 자세 → 전방 주시 태만 (기준: 캘리브레이션 때 잡은 '정면' 각도)
HEAD_POSE_ENABLE    = True
HEAD_YAW_OFF_DEG    = 30.0   # 좌우로 이 각도 넘게 돌리면 전방 이탈
HEAD_PITCH_DOWN_DEG = 20.0   # 아래로 숙임 (휴대폰·센터콘솔)
HEAD_PITCH_UP_DEG   = 25.0   # 위로 젖힘
HEAD_HYSTERESIS_DEG = 5.0    # 복귀는 (임계값 - 이 값) 안쪽으로 들어와야 인정 → 경계에서 깜빡임 방지
HEAD_EMA_ALPHA      = 0.4    # 각도 스무딩 (랜드마크 떨림 완화)
OFFROAD_LV1_SEC     = 3.0    # 한 번에 3초 이상 이탈 → LV1 (Euro NCAP '장시간 주의분산' 3~4초)
OFFROAD_LV2_SEC     = 6.0    # 6초 이상 → LV2
VATS_WINDOW_SEC     = 30.0   # 최근 30초 중
VATS_LV1_SEC        = 10.0   # 이탈 누적 10초 이상 → LV1 (짧게 여러 번 보는 패턴)
DRAW_HEAD_AXIS      = False   # 코끝에서 얼굴 방향 선 표시 (머리 자세 확인용, 끄려면 False)

# 경고음 — 디스플레이 스피커(HDMI)로 재생 (ESP32 부저 대신)
SOUND_ENABLE        = True
SOUND_DEVICE        = "plughw:0,3"   # aplay -l 의 card 0, device 3 (HDMI 0). 기본 출력 장치를 쓰려면 "default"
SOUND_DIR           = "./sounds"     # 경고음 wav를 처음 실행 때 자동 생성해 저장
SOUND_VOLUME        = 0.8            # 0.0 ~ 1.0
LV1_TONE_HZ         = 880            # LV1: 낮은 음 '삑-삑' 1초 간격
LV2_TONE_HZ         = 1760           # LV2: 높은 음 빠른 연속 '삑삑삑'
# 전시용 웹 UI (Flask) — 같은 폴더의 exhibit_ui.html을 http://localhost:5000 으로 띄움
WEB_UI_ENABLE       = True
WEB_PORT            = 5000
WEB_STREAM_FPS      = 15             # 웹으로 보내는 영상 fps (판정은 카메라 30fps 그대로). 무거우면 10
WEB_STREAM_WIDTH    = 640            # 웹 영상 가로 크기 (더 줄이면 압축 부담 감소, 예: 480)
WEB_JPEG_QUALITY    = 75
WEB_UI_AUTO_OPEN    = True           # 프로그램 실행 시 두 번째 모니터에 UI 자동으로 띄움 (start_exhibit_ui.sh 사용)
WEB_UI_MONITOR      = ""             # 띄울 모니터 이름 (xrandr 이름, 예: "DP-0"). 빈칸이면 자동 선택
BUZ_SOUND_FILE      = None           # ESP32 !BUZ# 알림음. None이면 '띵-동-띵-동' 자동 생성,
                                     # 녹음한 음성 등을 쓰려면 wav 경로 (예: "./sounds/handsoff_voice.wav")

# BT 통신 실패 재시작
# --restarted 인자가 있으면 이미 1회 재시작된 것 → 시뮬레이션 모드로 전환
BT_ALREADY_RESTARTED = "--restarted" in sys.argv

# 모델 경로 (ONNX) — 첫 실행 시 TensorRT 엔진으로 변환해 TRT_CACHE_DIR에 저장
FACE_DET_MODEL  = "version-RFB-320.onnx"                    # 얼굴 bbox 검출
LANDMARK_MODEL  = "landmark_detection_56_se_external.onnx"  # 68점 랜드마크 (56×56 입력)

# TensorRT 설정
TRT_CACHE_DIR   = "./trt_cache"   # 변환된 엔진(.engine) 저장 위치
TRT_FP16        = True            # FP16 추론 (젯슨 나노 속도 향상)
TRT_WORKSPACE   = 1 << 28         # 엔진 빌드 작업 메모리 256MB

# 화면에 눈 랜드마크 12점 표시 (랜드마크가 눈에 제대로 붙는지 확인용, 끄려면 False)
DRAW_EYE_POINTS = False

# 68점 랜드마크 눈 인덱스
LEFT_EYE  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

# ══════════════════════════════════════════
# 1. USB 웹캠 (V4L2)
# ══════════════════════════════════════════
def _apply_v4l2_controls():
    """웹캠 노출 설정 — 자동 노출이 어두운 곳에서 FPS를 30→15로 낮추는 것 방지.
    설정 이름이 웹캠·커널마다 달라서 지원되는 이름을 찾아 적용, 없으면 조용히 넘어감."""
    if not (CAM_KEEP_FPS or CAM_EXPOSURE is not None):
        return
    if shutil.which("v4l2-ctl") is None:
        print("⚠️  [웹캠] v4l2-ctl이 없어 노출 설정을 건너뜁니다 (sudo apt install v4l-utils)")
        return
    dev = f"/dev/video{CAM_INDEX}"
    try:
        out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-ctrls"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=3).stdout.decode(errors="ignore")
    except Exception:
        return
    names = {line.split()[0] for line in out.splitlines() if line.strip()}
    pick = lambda *cands: next((c for c in cands if c in names), None)
    sets = []
    if CAM_KEEP_FPS:
        n = pick("exposure_auto_priority", "exposure_dynamic_framerate")
        if n:
            sets.append(f"{n}=0")
    if CAM_EXPOSURE is not None:
        auto_n = pick("exposure_auto", "auto_exposure")
        abs_n  = pick("exposure_absolute", "exposure_time_absolute")
        if auto_n and abs_n:
            sets += [f"{auto_n}=1", f"{abs_n}={int(CAM_EXPOSURE)}"]   # 1 = 수동 모드
    for kv in sets:
        r = subprocess.run(["v4l2-ctl", "-d", dev, "-c", kv], stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=3)
        msg = "적용" if r.returncode == 0 else "실패: " + r.stderr.decode(errors="ignore").strip()
        print(f"[웹캠] {kv} {msg}")


class CameraStream:
    """별도 스레드로 웹캠 프레임 지속 수신 (버퍼 지연 방지).
    프레임마다 번호(frame_id)를 매겨, read(last_id)가 '새 프레임'만 넘겨주도록 함."""
    def __init__(self):
        self.stream = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(CAM_INDEX)          # V4L2 백엔드가 없는 빌드 대비
        if not self.stream.isOpened():
            print(f"❌ 웹캠을 열 수 없습니다 (CAM_INDEX={CAM_INDEX}). ls /dev/video* 로 번호를 확인하세요.")
            sys.exit(1)
        if CAM_FOURCC:
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM_FOURCC))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.stream.set(cv2.CAP_PROP_FPS,          FPS_TARGET)
        # ※ CAP_PROP_BUFFERSIZE=1 은 쓰지 않음: 웹캠은 버퍼가 1장이면 한 장 처리하는 동안
        #   다음 프레임을 받을 자리가 없어 한 장씩 건너뜀 → 30fps가 15fps로 떨어짐.
        #   지연은 별도 스레드가 계속 최신 프레임으로 덮어써서 생기지 않음.
        _apply_v4l2_controls()                                  # 카메라를 연 뒤에 적용해야 유지됨
        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed or self.frame is None:
            print("❌ 웹캠에서 프레임을 읽지 못했습니다. 다른 프로그램이 카메라를 쓰고 있는지 확인하세요.")
            sys.exit(1)
        self.height, self.width = self.frame.shape[:2]
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        print(f"✅ 웹캠 {self.width}×{self.height} @ {fps:.0f}fps (요청 {CAM_WIDTH}×{CAM_HEIGHT} @ {FPS_TARGET})")
        self.frame_id          = 1
        self.cam_fps           = 0.0      # 실제로 카메라에서 들어오는 속도 (화면 표시용)
        self._last_t           = None
        self.stopped           = False
        self._lock             = threading.Lock()
        self._new_frame        = threading.Condition(self._lock)
        self._consecutive_fail = 0

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        # 이 스레드는 '받아서 넘기기'만 — 좌우 반전 등은 read()에서 (다음 프레임을 놓치지 않게)
        while not self.stopped:
            if self.stream.isOpened():
                grabbed, frame = self.stream.read()
                with self._new_frame:
                    if grabbed and frame is not None:
                        now = time.time()
                        if self._last_t is not None:
                            inst = 1.0 / max(now - self._last_t, 1e-3)
                            self.cam_fps = inst if self.cam_fps == 0 else 0.9 * self.cam_fps + 0.1 * inst
                        self._last_t           = now
                        self.grabbed           = grabbed
                        self.frame             = frame
                        self.frame_id         += 1
                        self._consecutive_fail = 0
                        self._new_frame.notify_all()
                    else:
                        self._consecutive_fail += 1

    def read(self, last_id=None, timeout=0.1):
        """새 프레임(frame_id != last_id)이 올 때까지 최대 timeout초 기다린 뒤
        (frame_id, 프레임 복사본) 반환. 그동안 새 프레임이 없으면 (frame_id, None)."""
        with self._new_frame:
            if last_id is not None:
                self._new_frame.wait_for(
                    lambda: self.frame_id != last_id or self.stopped, timeout)
                if self.frame_id == last_id:
                    return self.frame_id, None
            if self.frame is None:
                return self.frame_id, None
            # 반전은 새 배열을 만들므로 복사를 겸함
            return self.frame_id, (cv2.flip(self.frame, 1) if CAM_MIRROR else self.frame.copy())

    @property
    def consecutive_fail(self):
        with self._lock:
            return self._consecutive_fail

    def stop(self):
        self.stopped = True
        with self._new_frame:
            self._new_frame.notify_all()
        time.sleep(0.1)
        if self.stream.isOpened():
            self.stream.release()

# ══════════════════════════════════════════
# 2. 블루투스 — ① 자동 재연결 포함
# ══════════════════════════════════════════
sock         = None
sock_lock    = threading.RLock()   # RLock: _send_command 안에서 _bt_restart가 다시 잡아도 멈추지 않게
target_state = "OFF"
is_running   = True
bt_lock      = threading.Lock()

# 시뮬레이션 모드
sim_mode      = False         # BT 연결 실패 시 True로 전환
sim_mode_lock = threading.Lock()
_sim_log_buf  = []            # 시뮬레이션 상태 변경 기록

# ESP32 → Jetson 수신 처리 (수신은 bt_reader_thread 하나만 함)
_ack_q          = queue.Queue()      # ESP32가 보낸 명령 응답 줄 ('A' 등)
sock_write_lock = threading.Lock()   # 여러 스레드가 동시에 send 하지 않도록
_last_lv_cmd_t  = 0.0                # 마지막으로 LV1/LV2 명령을 보낸 시각
buz_last_time   = 0.0                # 마지막 핸들 이탈 !BUZ 수신 시각 (화면 배너용)

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
            try: _bt_write(sock, "!OFF#\n")
            except: pass
            try: sock.close()
            except: pass
            sock = None
    if _alarm_sound is not None:
        _alarm_sound.stop()
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

def _bt_write(s, text: str) -> bool:
    """소켓에 한 줄 전송 (스레드 간 send 직렬화). 실패하면 False."""
    if s is None:
        return False
    with sock_write_lock:
        s.send(text.encode("utf-8"))
    return True

def _drain_acks():
    while True:
        try: _ack_q.get_nowait()
        except queue.Empty: return

def _wait_ack(timeout: float):
    """응답 줄 하나 기다림. 시간 초과면 None."""
    try:
        return _ack_q.get(timeout=timeout)
    except queue.Empty:
        return None

def _drop_socket(s, why: str):
    """끊긴 소켓 정리 → bluetooth_thread가 재연결."""
    global sock
    if s is None:
        return
    print(f"⚠️  [BT] 연결 끊김: {why}")
    try: s.close()
    except Exception: pass
    if sock is s:
        sock = None

def _send_command(cmd_text: str) -> bool:
    """
    명령 전송 + ESP32 응답 판별 (응답은 bt_reader_thread가 _ack_q에 넣어 줌).
    - 'A'  포함 → 정상 ACK  → 카운트 리셋, True 반환
    - 그 외 응답 → 비정상 카운트 +1
    - BT_ACK_TIMEOUT 동안 응답 없음 → 비정상 카운트 +1
    - 소켓 예외 → 소켓 정리 후 False 반환
    """
    with sock_lock:
        s = sock
        if not s:
            return False
        packet = f"!{cmd_text}#\n"
        for _ in range(BT_ACK_RETRY):
            if not is_running:
                return False
            try:
                _drain_acks()                        # 늦게 도착한 이전 응답 버림
                if cmd_text in ("LV1_WARN", "LV2_DANGER"):
                    global _last_lv_cmd_t            # 보낸 시각 기록 — ESP32가 'A' 바로 뒤에 보내는 !BUZ# 구분용
                    _last_lv_cmd_t = time.time()
                _bt_write(s, packet)
            except Exception as e:
                _drop_socket(s, f"전송 에러 {e}")
                return False
            res = _wait_ack(BT_ACK_TIMEOUT)
            if res is None:
                if sock is not s:                    # 기다리는 동안 끊김
                    return False
                _bt_record_abnormal("응답 타임아웃")
            elif "A" in res:
                _bt_record_normal()
                return True
            else:
                _bt_record_abnormal(f"비정상 응답: '{res}'")
            time.sleep(0.1)
    return False

# ──────────────────────────────────────────
# ESP32 → Jetson 이벤트: !BUZ# → 디스플레이 스피커 알림음 (ESP32 코드는 그대로)
#   ESP32가 !BUZ#를 보내는 경우 두 가지:
#     ① 핸들 이탈 (1.5초 이상 손 뗌, 계속되면 3초마다)  → 알림음 + 화면 'HANDS ON WHEEL!'
#     ② Jetson의 LV1/LV2 명령을 받았을 때 ("A\n" 바로 뒤에 옴) → BUZ_PLAY_ON_LV면 알림음
#   Jetson은 !BUZ#에 응답을 보내지 않음:
#     ESP32는 한 줄 읽고 남은 수신 데이터를 비우므로, 응답을 보내면 바로 뒤의 명령·하트비트가
#     지워질 수 있음 → ACK 누락 → 비정상 카운트. 그래서 명령은 지금처럼 'A'를 받은 뒤에만 다음 것을 보냄.
# ──────────────────────────────────────────
def _on_buz(body: str):
    global buz_last_time
    now = time.time()
    lv_echo = now - _last_lv_cmd_t < BUZ_LV_ECHO_SEC
    if lv_echo:
        print("🔔 [ESP32] !BUZ# 수신 (LV 경고 수신 응답)"
              + (" → 알림음" if BUZ_PLAY_ON_LV else " → 무시 (LV 경고음 재생 중)"))
        if not BUZ_PLAY_ON_LV:
            return
    else:
        if _web_ui is not None and now - buz_last_time > BUZ_BANNER_SEC:
            _web_ui.event("hands", "핸들에서 손을 뗌")
        buz_last_time = now
        print("🔔 [ESP32] !BUZ# 수신 (핸들 이탈) → 디스플레이 스피커 알림음")
    if _alarm_sound is not None:
        _alarm_sound.play_buz()

def _handle_rx_line(line: str, s):
    """ESP32에서 온 한 줄: !...# 이면 이벤트, 그 외('A' 등)는 명령 응답."""
    if not line:
        return
    if line.startswith("!") and line.endswith("#"):
        body = line[1:-1]
        if body.startswith("BUZ"):
            _on_buz(body)
        else:
            print(f"[BT] 알 수 없는 ESP32 메시지: {line}")
        return
    _ack_q.put(line)

def bt_reader_thread():
    """ESP32에서 오는 데이터를 혼자 다 읽어서 줄 단위로 나눔 → 응답은 _ack_q, 이벤트는 바로 처리."""
    cur, buf = None, b""
    while is_running:
        s = sock
        if s is None:
            cur, buf = None, b""
            time.sleep(0.1)
            continue
        if s is not cur:                             # 새 연결 → 버퍼 비움
            cur, buf = s, b""
        try:
            data = s.recv(1024)
        except socket.timeout:
            continue
        except OSError as e:
            if sock is s:
                _drop_socket(s, f"수신 에러 {e}")
            continue
        if not data:
            _drop_socket(s, "ESP32가 연결을 닫음")
            continue
        buf += data
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            _handle_rx_line(raw.decode("utf-8", errors="ignore").strip(), s)
        if len(buf) > 4096:                          # 줄바꿈 없는 쓰레기 데이터 방지
            buf = b""

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
                        _drain_acks()
                        _bt_write(s, "!H#\n")
                        res = _wait_ack(BT_HB_ACK_TIMEOUT)
                        if res is not None:
                            if "A" in res:
                                _bt_record_normal()
                            else:
                                _bt_record_abnormal(f"하트비트 비정상 응답: '{res}'")
                        # 타임아웃은 하트비트에서 허용 (카운트 안 함)
                    except Exception as e:
                        _drop_socket(s, f"하트비트 전송 에러 {e}")
            last_heartbeat = now

        time.sleep(0.05)

# ══════════════════════════════════════════
# 3. GPU 추론 — TensorRT 엔진 (얼굴 검출 + 랜드마크)
# ══════════════════════════════════════════
# 젯슨 나노(JetPack 4.x = TensorRT 7.1 ~ 8.2) 기준 '바인딩' API 사용.
# GPU 메모리는 pycuda 대신 libcudart를 ctypes로 직접 호출 → 추가 설치 없음.

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class _CudaRT:
    """CUDA 런타임(libcudart) 최소 래퍼: 메모리 할당 / 복사만 사용."""
    H2D, D2H = 1, 2   # cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost

    def __init__(self):
        candidates = []
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        candidates.append("libcudart.so")
        candidates += sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"), reverse=True)
        candidates += sorted(glob.glob("/usr/local/cuda*/targets/*/lib/libcudart.so*"), reverse=True)

        lib = None
        for name in candidates:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("libcudart.so(CUDA 런타임)를 찾지 못했습니다. "
                               "JetPack CUDA가 /usr/local/cuda 에 설치돼 있는지 확인하세요.")

        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMalloc.restype  = ctypes.c_int
        lib.cudaFree.argtypes   = [ctypes.c_void_p]
        lib.cudaFree.restype    = ctypes.c_int
        lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.cudaMemcpy.restype  = ctypes.c_int
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype  = ctypes.c_char_p
        self._lib = lib

    def _check(self, err, what):
        if err != 0:
            msg = self._lib.cudaGetErrorString(err)
            raise RuntimeError(f"{what} 실패: {msg.decode() if msg else err} (code {err})")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr.value

    def free(self, ptr):
        self._lib.cudaFree(ptr)

    def htod(self, dev_ptr, host_arr):
        self._check(self._lib.cudaMemcpy(dev_ptr, host_arr.ctypes.data,
                                         host_arr.nbytes, self.H2D), "cudaMemcpy(H→D)")

    def dtoh(self, host_arr, dev_ptr):
        self._check(self._lib.cudaMemcpy(host_arr.ctypes.data, dev_ptr,
                                         host_arr.nbytes, self.D2H), "cudaMemcpy(D→H)")


_CUDA = None

def _cuda():
    global _CUDA
    if _CUDA is None:
        _CUDA = _CudaRT()
    return _CUDA


def _engine_path(onnx_path: str) -> str:
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    return os.path.join(TRT_CACHE_DIR, stem + ("_fp16" if TRT_FP16 else "_fp32") + ".engine")


def _build_engine(onnx_path: str, engine_path: str):
    """ONNX → TensorRT 엔진 빌드 후 파일로 저장 (첫 실행 1회)."""
    print(f"🔧 [{os.path.basename(onnx_path)}] TensorRT 엔진 빌드 중... "
          f"(첫 실행만, 수 분 걸릴 수 있음 / INT64 관련 경고는 무시해도 됨)")
    t0 = time.time()
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX 파싱 실패:\n  " + "\n  ".join(errors))

    # 동적 배치(-1) 입력은 배치 1로 고정 → 최적화 프로파일 없이 정적 엔진으로 빌드
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape  = tuple(tensor.shape)
        if any(d < 0 for d in shape[1:]):
            raise RuntimeError(f"입력 '{tensor.name}' {shape}: 배치 외 차원이 동적이라 자동 고정 불가")
        if shape[0] < 0:
            tensor.shape = (1,) + shape[1:]

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):          # TensorRT 8.4+
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, TRT_WORKSPACE)
    else:                                                 # TensorRT 7.x ~ 8.2 (젯슨 나노)
        config.max_workspace_size = TRT_WORKSPACE
    if TRT_FP16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("⚠️  이 GPU는 FP16 가속을 지원하지 않아 FP32로 빌드합니다.")

    if hasattr(builder, "build_serialized_network"):      # TensorRT 8.x
        plan = builder.build_serialized_network(network, config)
    else:                                                 # TensorRT 7.x
        engine = builder.build_engine(network, config)
        plan   = engine.serialize() if engine is not None else None
    if plan is None:
        raise RuntimeError("TensorRT 엔진 빌드 실패 (위쪽 TensorRT 로그 확인)")

    with open(engine_path, "wb") as f:
        f.write(plan)
    print(f"✅ 엔진 저장: {engine_path} ({time.time() - t0:.0f}초)")


def _load_engine(runtime, engine_path: str):
    if not os.path.exists(engine_path):
        return None
    try:
        with open(engine_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print(f"⚠️  엔진 로드 예외: {e}")
        return None


class TRTModel:
    """TensorRT 엔진 1개 = 입력 1개 / 출력 N개. infer(blob) → {출력이름: ndarray}"""

    def __init__(self, onnx_path: str):
        self.name   = os.path.basename(onnx_path)
        engine_path = _engine_path(onnx_path)

        self.runtime = trt.Runtime(TRT_LOGGER)            # 엔진보다 오래 살아 있어야 함
        engine = _load_engine(self.runtime, engine_path)
        if engine is None:
            if os.path.exists(engine_path):
                print(f"⚠️  [{self.name}] 캐시 엔진을 읽지 못함 (TensorRT 버전 변경?) → 다시 빌드")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX 모델 없음: {onnx_path}")
            _build_engine(onnx_path, engine_path)
            engine = _load_engine(self.runtime, engine_path)
            if engine is None:
                raise RuntimeError(f"[{self.name}] 빌드한 엔진을 로드하지 못했습니다.")
        self.engine  = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"[{self.name}] 실행 컨텍스트 생성 실패 (GPU 메모리 부족?)")

        # trtexec 등으로 만든 '동적 배치' 엔진이면 배치 1로 지정
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                shape = tuple(engine.get_binding_shape(i))
                if any(d < 0 for d in shape):
                    fixed = tuple(1 if d < 0 else d for d in shape)
                    if not self.context.set_binding_shape(i, fixed):
                        raise RuntimeError(f"[{self.name}] 입력 shape {fixed} 지정 실패")

        # 바인딩별 호스트/GPU 버퍼 할당
        cuda = _cuda()
        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(engine.num_bindings):
            shape = tuple(self.context.get_binding_shape(i))
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
            host  = np.zeros(shape, dtype=dtype)
            dev   = cuda.malloc(host.nbytes)
            self.bindings.append(int(dev))
            item = (engine.get_binding_name(i), host, dev)
            (self.inputs if engine.binding_is_input(i) else self.outputs).append(item)
        if len(self.inputs) != 1:
            raise RuntimeError(f"[{self.name}] 입력이 1개인 모델만 지원합니다 (현재 {len(self.inputs)}개)")

        print(f"✅ [{self.name}] TensorRT 엔진 로드 — 입력 {self.input_shape}, "
              f"출력 {[h.shape for _, h, _ in self.outputs]}")

    @property
    def input_shape(self):
        return self.inputs[0][1].shape

    @property
    def output_shapes(self):
        return {name: host.shape for name, host, _ in self.outputs}

    def infer(self, blob: np.ndarray) -> dict:
        _, host_in, dev_in = self.inputs[0]
        if blob.shape != host_in.shape:
            raise ValueError(f"[{self.name}] 입력 shape 불일치: 모델 {host_in.shape}, 입력 {blob.shape}")
        np.copyto(host_in, blob)                  # dtype 변환 포함 (float64 → float32)
        cuda = _cuda()
        cuda.htod(dev_in, host_in)
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError(f"[{self.name}] TensorRT 추론 실행 실패")
        outs = {}
        for name, host_out, dev_out in self.outputs:
            cuda.dtoh(host_out, dev_out)
            outs[name] = host_out.copy()
        return outs


class FaceDetector:
    """
    Ultra-Light-Fast-Generic-Face-Detector (version-RFB-320) TensorRT 엔진.
    입력: BGR 프레임 → 320×240 RGB, (x-127)/128
    출력: [(x1, y1, x2, y2, score), ...] 원본 해상도 기준 좌표
    """

    INPUT_W, INPUT_H = 320, 240
    CONF_THR   = 0.7
    IOU_THR    = 0.4
    CENTER_VAR = 0.1     # *_without_postprocessing 모델 디코딩용
    SIZE_VAR   = 0.2

    def __init__(self):
        self.model = TRTModel(FACE_DET_MODEL)
        if self.model.input_shape != (1, 3, self.INPUT_H, self.INPUT_W):
            raise RuntimeError(f"얼굴 검출 입력 shape가 예상과 다름: {self.model.input_shape}")

        # 출력 2개를 마지막 차원으로 구분: scores [1,N,2], boxes [1,N,4]
        self._score_key = self._box_key = None
        for name, shape in self.model.output_shapes.items():
            if shape[-1] == 2:
                self._score_key = name
            elif shape[-1] == 4:
                self._box_key = name
        if self._score_key is None or self._box_key is None:
            raise RuntimeError(f"얼굴 검출 출력 형식이 예상과 다름: {self.model.output_shapes}")

        # 후처리 없는 모델이면 앵커 디코딩을 CPU(numpy)에서 수행
        self.decode_on_cpu = "without_postprocessing" in os.path.basename(FACE_DET_MODEL)
        self.priors = None
        if self.decode_on_cpu:
            self.priors = self._make_priors(self.INPUT_W, self.INPUT_H)
            n_out = self.model.output_shapes[self._box_key][1]
            if len(self.priors) != n_out:
                raise RuntimeError(f"앵커 수 불일치: priors {len(self.priors)} vs 모델 {n_out}")
            print("   (얼굴 박스 디코딩: CPU)")

    @staticmethod
    def _make_priors(w, h):
        """Linzaer fd_config와 동일한 앵커 (320×240 → 4420개)."""
        strides   = [8, 16, 32, 64]
        min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        priors = []
        for stride, sizes in zip(strides, min_boxes):
            fw, fh = int(np.ceil(w / stride)), int(np.ceil(h / stride))
            for j in range(fh):
                for i in range(fw):
                    cx, cy = (i + 0.5) / fw, (j + 0.5) / fh
                    for s in sizes:
                        priors.append([cx, cy, s / w, s / h])
        return np.clip(np.array(priors, dtype=np.float32), 0.0, 1.0)

    def _decode(self, loc, priors):
        center = loc[:, :2] * self.CENTER_VAR * priors[:, 2:] + priors[:, :2]
        size   = np.exp(loc[:, 2:] * self.SIZE_VAR) * priors[:, 2:]
        return np.concatenate([center - size / 2, center + size / 2], axis=1)

    def detect(self, frame: np.ndarray):
        """얼굴 bbox 리스트 반환 (없으면 빈 리스트)."""
        h, w = frame.shape[:2]
        blob = cv2.resize(frame, (self.INPUT_W, self.INPUT_H))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (blob - 127.0) / 128.0                 # ultraface 정규화
        blob = blob.transpose(2, 0, 1)[np.newaxis]    # NCHW

        outs   = self.model.infer(blob)
        scores = outs[self._score_key].reshape(-1, 2)[:, 1]
        boxes  = outs[self._box_key].reshape(-1, 4)    # (x1,y1,x2,y2) 0~1 정규화 좌표

        # 4420개 앵커를 파이썬 for문 대신 numpy로 한 번에 거름 (나노 CPU 부담 감소)
        keep = np.flatnonzero(scores >= self.CONF_THR)
        if keep.size == 0:
            return []
        scores = scores[keep]
        boxes  = boxes[keep]
        if self.decode_on_cpu:
            boxes = self._decode(boxes, self.priors[keep])
        boxes = boxes * np.array([w, h, w, h], dtype=np.float32)

        # NMS
        rects = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
        idxs  = cv2.dnn.NMSBoxes(rects, scores.tolist(), self.CONF_THR, self.IOU_THR)
        idxs  = np.array(idxs, dtype=np.int64).reshape(-1)
        return [(int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3]),
                 float(scores[i])) for i in idxs]


class LandmarkDetector:
    """
    68점 랜드마크 (cunjian/pytorch_face_landmark, MobileNet-SE) TensorRT 엔진.
    전처리는 원저자 예제(test_camera_light_onnx.py)와 동일:
      얼굴 박스 → 긴 변×1.1 정사각형 크롭(프레임 밖은 검은색 패딩) → 56×56
      → RGB → /255 → ImageNet mean/std 정규화
    출력: (68, 2) 원본 프레임 픽셀 좌표
    """

    CROP_SCALE = 1.1
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self):
        self.model = TRTModel(LANDMARK_MODEL)
        _, c, self.in_h, self.in_w = self.model.input_shape
        if c != 3:
            raise RuntimeError(f"랜드마크 입력 채널이 3이 아님: {self.model.input_shape}")
        out_shape = list(self.model.output_shapes.values())[0]
        if int(np.prod(out_shape)) != 136:
            raise RuntimeError(f"랜드마크 출력이 68점(136)이 아님: {out_shape}")

    def predict(self, frame: np.ndarray, bbox):
        """(68, 2) 랜드마크 좌표 반환. 실패 시 None."""
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        fh, fw = frame.shape[:2]

        # 박스 중심 기준 정사각형 (긴 변 × 1.1)
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        size = int(max(bw, bh) * self.CROP_SCALE)
        if size < 8:
            return None
        sx1 = x1 + bw // 2 - size // 2
        sy1 = y1 + bh // 2 - size // 2
        sx2, sy2 = sx1 + size, sy1 + size

        # 프레임 안쪽만 자르고 나머지는 검은색 패딩 → 항상 size×size
        cx1, cy1 = max(0, sx1), max(0, sy1)
        cx2, cy2 = min(fw, sx2), min(fh, sy2)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        if (cx1, cy1, cx2, cy2) != (sx1, sy1, sx2, sy2):
            crop = cv2.copyMakeBorder(crop, int(cy1 - sy1), int(sy2 - cy2),
                                      int(cx1 - sx1), int(sx2 - cx2),
                                      cv2.BORDER_CONSTANT, value=0)

        face = cv2.resize(crop, (self.in_w, self.in_h))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        face = (face - self.MEAN) / self.STD
        blob = face.transpose(2, 0, 1)[np.newaxis]     # (1, 3, 56, 56)

        out = list(self.model.infer(blob).values())[0]  # (1, 136), 크롭 기준 0~1
        pts = out.reshape(-1, 2).astype(np.float32)

        # 크롭 좌표 → 원본 프레임 좌표 역투영
        pts[:, 0] = pts[:, 0] * size + sx1
        pts[:, 1] = pts[:, 1] * size + sy1
        return pts


# ══════════════════════════════════════════
# 4. EAR 계산
# ══════════════════════════════════════════
def calculate_ear(eye_pts: np.ndarray) -> float:
    v1 = dist.euclidean(eye_pts[1], eye_pts[5])
    v2 = dist.euclidean(eye_pts[2], eye_pts[4])
    h  = dist.euclidean(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

# ══════════════════════════════════════════
# 5. EMA 스무딩
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
# 6. 졸음 판정 — 시간 기준 (연속 눈 감김 + 장기 PERCLOS)
# ══════════════════════════════════════════
class DrowsinessJudge:
    """
    졸음 단계(0/1/2) 판정. '프레임 수'가 아니라 '시간'으로 판단 → FPS가 흔들려도 같은 기준.
      ① 연속 눈 감김 (급성, 마이크로슬립): EYE_CLOSED_LV1_SEC 이상 → 1, EYE_CLOSED_LV2_SEC 이상 → 2
         눈을 다시 뜨면 ALARM_HOLD_SEC 뒤 해제 (뜬 직후 바로 끊겼다 켜지는 것 방지)
      ② PERCLOS (누적 졸음 경향): 최근 PERCLOS_WINDOW_SEC 중 눈 감긴 시간 비율
         ≥ PERCLOS_LV1 → 1, ≥ PERCLOS_LV2 → 2  (데이터가 PERCLOS_MIN_SEC 이상 쌓인 뒤부터)
    감김/뜸은 기준 두 개(감김 기준 < 뜸 기준) 사이를 넘을 때만 바뀜 → 경계값 근처에서 깜빡임 방지.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.closed       = False
        self.closed_since = None
        self.closed_dur   = 0.0        # 지금 이어지고 있는 눈 감김 시간 (초)
        self.perclos      = 0.0
        self.level        = 0
        self.lv_close     = 0          # 연속 눈 감김 단계
        self.lv_perc      = 0          # PERCLOS 단계
        self.cause        = ""         # 화면 표시용 원인
        self._hist        = deque()    # (시각, 시간 조각, 감김 여부) — 최근 PERCLOS_WINDOW_SEC
        self._t_total     = 0.0
        self._t_closed    = 0.0
        self._last_t      = None
        self._event_max   = 0          # 이번 감김에서 올라간 최고 단계 (뜬 뒤 잠깐 유지)
        self._open_since  = None

    def update(self, ear: float, thr_close: float, thr_open: float, now: float) -> int:
        # 감김/뜸 판정 (히스테리시스)
        if self.closed:
            if ear > thr_open:
                self.closed = False
        elif ear < thr_close:
            self.closed = True

        # PERCLOS: 최근 창 안에서 눈 감긴 '시간' 비율
        dt = 0.0 if self._last_t is None else min(max(now - self._last_t, 0.0), 0.5)
        self._last_t = now
        if dt > 0:
            self._hist.append((now, dt, self.closed))
            self._t_total += dt
            if self.closed:
                self._t_closed += dt
        while self._hist and self._hist[0][0] < now - PERCLOS_WINDOW_SEC:
            _, d0, c0 = self._hist.popleft()
            self._t_total -= d0
            if c0:
                self._t_closed -= d0
        self._t_closed = max(self._t_closed, 0.0)
        self.perclos = (self._t_closed / self._t_total) if self._t_total >= PERCLOS_MIN_SEC else 0.0

        # ① 연속 눈 감김
        if self.closed:
            if self.closed_since is None:
                self.closed_since = now
            self.closed_dur  = now - self.closed_since
            self._open_since = None
            lv = (2 if self.closed_dur >= EYE_CLOSED_LV2_SEC
                  else 1 if self.closed_dur >= EYE_CLOSED_LV1_SEC else 0)
            self._event_max = max(self._event_max, lv)
        else:
            self.closed_since, self.closed_dur = None, 0.0
            if self._event_max > 0:                      # 뜬 뒤 ALARM_HOLD_SEC 동안만 유지
                if self._open_since is None:
                    self._open_since = now
                if now - self._open_since >= ALARM_HOLD_SEC:
                    self._event_max = 0
        lv_close = self._event_max

        # ② PERCLOS
        lv_perc = 2 if self.perclos >= PERCLOS_LV2 else 1 if self.perclos >= PERCLOS_LV1 else 0

        self.lv_close, self.lv_perc = lv_close, lv_perc
        self.level = max(lv_close, lv_perc)
        if lv_close > 0 and lv_close >= lv_perc:
            self.cause = f"EYES CLOSED {self.closed_dur:.1f}s" if self.closed else "EYES CLOSED"
        elif lv_perc > 0:
            self.cause = f"PERCLOS {self.perclos:.0%}"
        else:
            self.cause = ""
        return self.level

# ══════════════════════════════════════════
# 6-1. 적응형 임계값 — 2단계 캘리브레이션 (뜬 눈 → 감은 눈) + 주기적 재보정
# ══════════════════════════════════════════
class AdaptiveThreshold:
    """
    운전자별 '눈 감김' 기준 EAR.
      1) OPEN 단계 (CALIB_OPEN_SEC): 정면 보고 눈 뜨기 → 뜬 눈 EAR (② IQR 이상치 제거 후 평균)
      2) CLOSED 단계 (CALIB_CLOSED_SEC): 눈 감기 → 감은 눈 EAR (하위 30% 값 — 조금 일찍 떠도 영향 적게)
      → 감김 기준 = 감은 눈 + CLOSE_POINT × (뜬 눈 − 감은 눈)
        뜸 기준   = 감김 기준 + CLOSE_HYST × (뜬 눈 − 감은 눈)
    이 랜드마크 모델은 눈을 감아도 EAR이 뜬 눈의 55~60% 정도까지만 내려감
    → '뜬 눈 × 고정 비율' 하나로는 뜬 눈과 감은 눈 사이 여유가 거의 없어서 감은 눈 값을 직접 잼.
    감은 눈 단계를 생략(CALIB_CLOSED_SEC=0)하거나 차이가 너무 작으면 감김 기준 = 뜬 눈 × CLOSE_RATIO_DEFAULT.
    """

    def __init__(self, recalib_interval, recalib_window_sec, fps, max_drift):
        self.recalib_interval = recalib_interval
        self.max_drift        = max_drift
        self.long_term_buf    = deque(maxlen=int(fps * recalib_window_sec))
        self.reset(verbose=False)

    def reset(self, verbose: bool = True):
        """새 탑승자를 위한 캘리브레이션 초기화 (R 키)."""
        self.phase         = "open"     # "open" → "closed" → "done"
        self._phase_start  = None
        self.open_buf      = []
        self.closed_buf    = []
        self.baseline      = None       # 뜬 눈 EAR
        self.closed_level  = None       # 감은 눈 EAR (못 재면 None)
        self.threshold     = None       # 감김 기준
        self.thr_open      = None       # 다시 뜸 기준
        self.is_calibrated = False
        self._last_recalib = None
        self.long_term_buf.clear()
        if verbose:
            print("[캘리브레이션 초기화] 새 탑승자 캘리브레이션 시작")

    @staticmethod
    def _iqr_filter(data: list, factor: float) -> list:
        """IQR 기반 이상치 제거 — 깜빡임 등 눈 감은 프레임의 낮은 EAR 값 제거"""
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

    def _set_threshold(self):
        b = self.baseline
        if self.closed_level is not None:
            gap = b - self.closed_level
            self.threshold = self.closed_level + CLOSE_POINT * gap
        else:
            gap = b * (1.0 - CLOSE_RATIO_DEFAULT) / (1.0 - CLOSE_POINT)   # 기본 비율에 해당하는 차이
            self.threshold = b * CLOSE_RATIO_DEFAULT
        self.thr_open = self.threshold + CLOSE_HYST * gap

    def _finish(self, now: float):
        self._set_threshold()
        self.phase, self.is_calibrated, self._last_recalib = "done", True, now
        c = f"{self.closed_level:.3f}" if self.closed_level is not None else "측정 안 함(기본 비율)"
        print(f"[캘리브레이션 완료] 뜬 눈 EAR={self.baseline:.3f}, 감은 눈 EAR={c} "
              f"→ 감김 기준 {self.threshold:.3f} / 뜸 기준 {self.thr_open:.3f}")

    @staticmethod
    def _phase_rules(dur: float):
        """단계 길이에 맞춘 (앞부분 버리는 시간, 최소 샘플 수) — 단계를 짧게 설정해도 끝없이 재시작하지 않게."""
        skip = min(CALIB_SKIP_SEC, 0.3 * dur)
        return skip, max(3, int(0.3 * (dur - skip) * FPS_TARGET))

    def update(self, ear: float, perclos: float = 0.0, on_road: bool = True) -> bool:
        now = time.time()

        # ── 캘리브레이션 ──
        if not self.is_calibrated:
            if self._phase_start is None:
                self._phase_start = now
            el = now - self._phase_start
            if self.phase == "open":
                skip, min_n = self._phase_rules(CALIB_OPEN_SEC)
                if el < CALIB_OPEN_SEC:
                    if ear > 0 and el >= skip:
                        self.open_buf.append(ear)
                    return False
                if len(self.open_buf) < min_n:                    # 얼굴이 거의 안 보였음
                    print("⚠️  [캘리브레이션] 얼굴이 잘 안 보여 다시 시작합니다 — 정면을 봐 주세요")
                    self.open_buf, self._phase_start = [], now
                    return False
                self.baseline = float(np.mean(self._iqr_filter(self.open_buf, CALIB_IQR_FACTOR)))
                if CALIB_CLOSED_SEC > 0:
                    self.phase, self._phase_start = "closed", now
                    print(f"[캘리브레이션] 뜬 눈 EAR={self.baseline:.3f} → "
                          f"이제 눈을 {CALIB_CLOSED_SEC:.0f}초 정도 감아 주세요")
                    return False
                self._finish(now)
                return True
            # closed 단계
            skip, min_n = self._phase_rules(CALIB_CLOSED_SEC)
            if el < CALIB_CLOSED_SEC:
                if ear > 0 and el >= skip:
                    self.closed_buf.append(ear)
                return False
            if len(self.closed_buf) >= min_n:
                c = float(np.percentile(self.closed_buf, 30))
                if self.baseline - c >= 0.15 * self.baseline:     # 눈을 제대로 감았음
                    self.closed_level = c
                else:
                    print(f"⚠️  [캘리브레이션] 눈 감기 단계 EAR({c:.3f})이 뜬 눈과 거의 같음 "
                          f"→ 기본 비율 사용 (R로 다시 가능)")
            else:
                print("⚠️  [캘리브레이션] 눈 감기 단계에서 얼굴이 안 보임 → 기본 비율 사용")
            self._finish(now)
            return True

        # ── 장기 버퍼 축적 (정상 + 정면 주시 + 눈 뜬 상태일 때만) ──
        # 아래를 보면 윗눈꺼풀이 따라 내려와 EAR이 낮아지므로, 고개를 돌린 동안은 제외
        if on_road and perclos < 0.10 and ear > self.thr_open:
            self.long_term_buf.append(ear)

        # ── 주기적 재보정: 뜬 눈 값만 갱신 (감은 눈 값은 캘리브레이션 값 유지) ──
        if now - self._last_recalib >= self.recalib_interval:
            if len(self.long_term_buf) > 30:
                new_baseline = float(np.mean(self.long_term_buf))
                drift        = abs(new_baseline - self.baseline) / self.baseline
                if drift <= self.max_drift:
                    self.baseline = new_baseline
                    self._set_threshold()
                    print(f"[재보정] 뜬 눈 EAR={self.baseline:.3f} → 감김 기준 {self.threshold:.3f}")
                else:
                    print(f"[재보정 스킵] drift={drift:.2%} > 허용 {self.max_drift:.2%} "
                          f"(R로 다시 캘리브레이션 가능)")
            self._last_recalib = now

        return True

    def calib_remaining(self) -> float:
        """현재 캘리브레이션 단계의 남은 시간 (초)."""
        dur = CALIB_CLOSED_SEC if self.phase == "closed" else CALIB_OPEN_SEC
        if self._phase_start is None:
            return dur
        return max(0.0, dur - (time.time() - self._phase_start))


# ══════════════════════════════════════════
# 6-2. 머리 자세(Head Pose) → 전방 주시 태만
# ══════════════════════════════════════════
# 일반 성인 얼굴 3D 모델 6점 (learnopencv 값, 단위 임의).
# 카메라 좌표계(x: 오른쪽, y: 아래, z: 카메라에서 멀어지는 방향)로 바꿔 둠 → 정면이면 각도 ≈ 0
HEAD_MODEL_3D = np.array([
    (   0.0,    0.0,   0.0),   # 30 코끝
    (   0.0,  330.0,  65.0),   # 8  턱끝
    (-225.0, -170.0, 135.0),   # 36 화면 왼쪽 눈 바깥 꼬리
    ( 225.0, -170.0, 135.0),   # 45 화면 오른쪽 눈 바깥 꼬리
    (-150.0,  150.0, 125.0),   # 48 입꼬리 (화면 왼쪽)
    ( 150.0,  150.0, 125.0),   # 54 입꼬리 (화면 오른쪽)
], dtype=np.float64)
HEAD_LM_IDX   = [30, 8, 36, 45, 48, 54]
HEAD_EYE_SPAN = 450.0          # 3D 모델의 양쪽 눈꼬리 사이 거리 (첫 프레임 거리 추정용)

SEVERITY    = {"OFF": 0, "LV1_WARN": 1, "LV2_DANGER": 2}
LEVEL_STATE = ("OFF", "LV1_WARN", "LV2_DANGER")


class HeadPoseEstimator:
    """
    68점 랜드마크 중 6점 + cv2.solvePnP → 머리 각도 (yaw, pitch, roll) [도].
      yaw  : 좌우 회전 (부호는 화면 기준, 판정에는 크기만 사용)
      pitch: + 아래로 숙임 / - 위로 젖힘
    · 초점거리는 '영상 가로폭' 근사값. 판정은 캘리브레이션 때의 정면과의 '차이'만 쓰므로 정확할 필요 없음
    · 직전 프레임 결과를 초기값으로 써서 빠르고 안정적으로 수렴
    """

    def __init__(self, frame_w: int, frame_h: int):
        fx = fy = float(frame_w)                         # f ≈ 영상 가로폭 (웹캠은 정사각 픽셀)
        self.K    = np.array([[fx, 0.0, frame_w / 2.0],
                              [0.0, fy, frame_h / 2.0],
                              [0.0, 0.0, 1.0]], dtype=np.float64)
        self.dist = np.zeros((4, 1), dtype=np.float64)
        self.rvec = None
        self.tvec = None

    def reset(self):
        self.rvec = None
        self.tvec = None

    def estimate(self, pts: np.ndarray):
        """(yaw, pitch, roll) 반환. 실패하면 None."""
        img  = np.ascontiguousarray(pts[HEAD_LM_IDX], dtype=np.float64).reshape(-1, 1, 2)
        span = float(np.linalg.norm(img[3, 0] - img[2, 0]))      # 양쪽 눈꼬리 간격 (px)
        if not np.isfinite(span) or span < 5.0:                  # 랜드마크 비정상 → 건너뜀
            self.reset()
            return None
        if self.rvec is None:
            # 첫 프레임: 눈꼬리 간격으로 거리, 코끝 위치로 좌우·상하 위치 초기값
            z    = self.K[0, 0] * HEAD_EYE_SPAN / span
            u, v = img[0, 0]
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.array([[(u - self.K[0, 2]) * z / self.K[0, 0]],
                             [(v - self.K[1, 2]) * z / self.K[1, 1]],
                             [z]], dtype=np.float64)
        else:
            rvec, tvec = self.rvec.copy(), self.tvec.copy()

        try:
            ok, rvec, tvec = cv2.solvePnP(HEAD_MODEL_3D, img, self.K, self.dist,
                                          rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:                          # 퇴화된 입력 (점이 겹침 등)
            ok = False
        if not ok or not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0:
            self.reset()
            return None
        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = cv2.RQDecomp3x3(R)[0]
        if abs(yaw) > 90 or abs(pitch) > 90:      # 뒤집힌 해 → 버리고 다음 프레임에 새로 시작
            self.reset()
            return None
        self.rvec, self.tvec = rvec, tvec
        return float(yaw), float(pitch), float(roll)

    def nose_line(self, length: float = 300.0):
        """화면 표시용: 코끝 → 얼굴이 향하는 방향 선분의 두 끝점."""
        p3 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -length]], dtype=np.float64)
        p2 = cv2.projectPoints(p3, self.rvec, self.tvec, self.K, self.dist)[0].reshape(-1, 2)
        return (int(p2[0, 0]), int(p2[0, 1])), (int(p2[1, 0]), int(p2[1, 1]))


class DistractionMonitor:
    """
    캘리브레이션 때 잡은 '정면' 대비 머리 방향으로 전방 주시 태만 단계(0/1/2) 판정.
      · 한 번에 OFFROAD_LV1_SEC 이상 이탈 → 1, OFFROAD_LV2_SEC 이상 → 2
      · 최근 VATS_WINDOW_SEC 동안 이탈 누적 VATS_LV1_SEC 이상 → 1
    사이드미러 확인처럼 짧은 이탈은 시간 조건에서 걸러짐.
    """

    def __init__(self):
        self.reset(verbose=False)

    def reset(self, verbose: bool = True):
        self.calib     = []             # 캘리브레이션 중 (yaw, pitch) 샘플
        self.base      = None           # 정면 기준 (yaw, pitch)
        self._yaw      = EMAFilter(HEAD_EMA_ALPHA)
        self._pitch    = EMAFilter(HEAD_EMA_ALPHA)
        self.off_road  = False
        self.off_since = None
        self.off_dur   = 0.0
        self._hist     = deque()        # (시각, 이탈 시간 조각) — 최근 VATS_WINDOW_SEC
        self.vats_sum  = 0.0
        self._last_t   = None
        self.dyaw      = 0.0
        self.dpitch    = 0.0
        self.level     = 0
        if verbose:
            print("[머리 자세] 정면 기준 초기화 — 캘리브레이션 동안 다시 잡음")

    def add_calib(self, yaw: float, pitch: float):
        self.calib.append((yaw, pitch))

    def _ensure_base(self, yaw: float, pitch: float):
        if self.base is not None:
            return
        if self.calib:
            arr = np.array(self.calib)
            self.base = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
        else:
            self.base = (yaw, pitch)
        print(f"[머리 자세 기준] 정면 yaw={self.base[0]:+.1f}°, pitch={self.base[1]:+.1f}° "
              f"(샘플 {len(self.calib)}개)")

    def _account(self, now: float, off: bool) -> int:
        dt = 0.0 if self._last_t is None else min(max(now - self._last_t, 0.0), 0.5)
        self._last_t = now
        if off:
            if not self.off_road:
                self.off_since = now
            if dt > 0:
                self._hist.append((now, dt))
                self.vats_sum += dt
        else:
            self.off_since = None
        self.off_road = off
        while self._hist and self._hist[0][0] < now - VATS_WINDOW_SEC:
            self.vats_sum -= self._hist.popleft()[1]
        self.vats_sum = max(self.vats_sum, 0.0)

        self.off_dur = (now - self.off_since) if off else 0.0
        if self.off_dur >= OFFROAD_LV2_SEC:
            self.level = 2
        elif self.off_dur >= OFFROAD_LV1_SEC or self.vats_sum >= VATS_LV1_SEC:
            self.level = 1
        else:
            self.level = 0
        return self.level

    def update(self, yaw: float, pitch: float, now: float) -> int:
        """얼굴이 보이는 프레임 (캘리브레이션 이후)."""
        self._ensure_base(yaw, pitch)
        y = self._yaw.update(yaw)
        p = self._pitch.update(pitch)
        self.dyaw, self.dpitch = y - self.base[0], p - self.base[1]
        m = HEAD_HYSTERESIS_DEG if self.off_road else 0.0
        off = (abs(self.dyaw) > HEAD_YAW_OFF_DEG - m
               or self.dpitch > HEAD_PITCH_DOWN_DEG - m
               or -self.dpitch > HEAD_PITCH_UP_DEG - m)
        return self._account(now, off)

    def update_missing(self, now: float) -> int:
        """얼굴을 놓친 프레임: 이탈 중이었다면 더 돌아간 것으로 보고 계속 이탈로 셈."""
        if self.base is None:
            return 0
        return self._account(now, self.off_road)

    def cause(self) -> str:
        if self.level == 0:
            return ""
        if self.off_dur >= OFFROAD_LV1_SEC:
            return f"EYES OFF ROAD {self.off_dur:.1f}s"
        return f"DISTRACTION {self.vats_sum:.0f}s/{VATS_WINDOW_SEC:.0f}s"


def _worse(state: str, d_level: int, calibrated: bool = True) -> str:
    """졸음 판정과 전방 주시 태만 판정 중 더 위험한 단계 (캘리브레이션 전에는 머리 판정 무시)."""
    if not calibrated or d_level <= 0:
        return state
    d_state = LEVEL_STATE[d_level]
    return d_state if SEVERITY[d_state] > SEVERITY.get(state, 0) else state


def _cause_text(drowsy_cause: str, d_level: int, distract) -> str:
    """화면 하단 오른쪽에 띄울 경보 원인 (예: 'EYES CLOSED 1.4s', 'PERCLOS 17%', 'EYES OFF ROAD 3.2s')."""
    parts = []
    if drowsy_cause:
        parts.append(drowsy_cause)
    if d_level > 0:
        parts.append(distract.cause())
    return " + ".join(p for p in parts if p)


# ══════════════════════════════════════════
# 6-3. 경고음 — 디스플레이 스피커 (aplay, 추가 설치 불필요)
# ══════════════════════════════════════════
class AlarmSound:
    """
    상태(LV1/LV2)에 맞는 경고음을 별도 스레드에서 반복 재생. 메인 루프는 set_level()만 호출.
      · wav 파일은 처음 실행 때 numpy로 만들어 SOUND_DIR에 저장 (LV1: 1초짜리, LV2: 0.5초짜리 반복 단위)
      · 단계가 바뀌면 재생 중인 소리를 즉시 끊고 새 소리로 전환, OFF면 정지
      · aplay가 없거나 재생이 계속 실패하면 경고만 출력하고 꺼짐 (본 기능엔 영향 없음)
    """
    RATE = 44100

    def __init__(self):
        self.enabled = SOUND_ENABLE and shutil.which("aplay") is not None
        self._level  = 0
        self._cond   = threading.Condition()
        self._proc   = None
        self._stop   = False
        self._fails  = 0
        self.files   = {}
        self._buz_pending = False
        self._playing     = 0              # 지금 재생 중: 0 없음, 1/2 단계음, "buz" 알림음
        if SOUND_ENABLE and not self.enabled:
            print("⚠️  [경고음] aplay를 찾지 못해 경고음 없이 동작합니다 (sudo apt install alsa-utils)")
        if self.enabled:
            os.makedirs(SOUND_DIR, exist_ok=True)
            self.files[1] = self._make_wav("lv1.wav", LV1_TONE_HZ, [(0.18, True), (0.12, False), (0.18, True), (0.52, False)])
            self.files[2] = self._make_wav("lv2.wav", LV2_TONE_HZ, [(0.09, True), (0.04, False)] * 4)
            if BUZ_SOUND_FILE and os.path.isfile(BUZ_SOUND_FILE):
                self.files["buz"] = BUZ_SOUND_FILE
            else:
                if BUZ_SOUND_FILE:
                    print(f"⚠️  [경고음] BUZ_SOUND_FILE({BUZ_SOUND_FILE}) 없음 → 기본 알림음 사용")
                self.files["buz"] = self._make_wav_seq("buz.wav", [(1320, 0.22), (0, 0.06), (990, 0.30), (0, 0.12),
                                                                  (1320, 0.22), (0, 0.06), (990, 0.30)])
            print(f"✅ [경고음] 디스플레이 스피커 사용 ({SOUND_DEVICE})")

    def _make_wav(self, name, hz, pattern):
        """pattern: [(초, 소리 여부), ...] — 끝에 클릭음이 안 나게 앞뒤 5ms 페이드."""
        chunks = []
        for sec, on in pattern:
            n = int(self.RATE * sec)
            if on:
                t = np.arange(n) / self.RATE
                tone = np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(2 * np.pi * 2 * hz * t)   # 배음 섞어 또렷하게
                f = min(int(0.005 * self.RATE), n // 2)
                env = np.ones(n); env[:f] = np.linspace(0, 1, f); env[n - f:] = np.linspace(1, 0, f)
                chunks.append(tone / 1.3 * env)
            else:
                chunks.append(np.zeros(n))
        pcm = (np.concatenate(chunks) * SOUND_VOLUME * 32767).astype(np.int16)
        path = os.path.join(SOUND_DIR, name)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(self.RATE)
            w.writeframes(pcm.tobytes())
        return path

    def _make_wav_seq(self, name, notes):
        """notes: [(주파수 Hz, 초), ...] — 주파수 0은 쉼. 단계음과 구별되게 두 음을 번갈아."""
        chunks = []
        for hz, sec in notes:
            n = int(self.RATE * sec)
            if hz:
                t = np.arange(n) / self.RATE
                tone = np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(2 * np.pi * 2 * hz * t)
                f = min(int(0.008 * self.RATE), n // 2)
                env = np.ones(n); env[:f] = np.linspace(0, 1, f); env[n - f:] = np.linspace(1, 0, f)
                chunks.append(tone / 1.3 * env)
            else:
                chunks.append(np.zeros(n))
        pcm = (np.concatenate(chunks) * SOUND_VOLUME * 32767).astype(np.int16)
        path = os.path.join(SOUND_DIR, name)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(self.RATE)
            w.writeframes(pcm.tobytes())
        return path

    def play_buz(self):
        """ESP32 !BUZ# → 알림음 1회. 단계음이 울리는 중이면 잠깐 끊고 알림음 → 끝나면 단계음 다시.
        (HDMI 출력 장치는 한 번에 소리 하나만 낼 수 있어서 겹치지 않고 순서대로)"""
        if not self.enabled:
            return
        with self._cond:
            if self._playing == "buz":
                return                         # 이미 알림음 재생 중
            self._buz_pending = True
            self._kill()
            self._cond.notify_all()

    def start(self):
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()
        return self

    def set_level(self, level: int):
        """0=정지, 1=LV1 경고음, 2=LV2 경고음."""
        if not self.enabled:
            return
        with self._cond:
            if level == self._level:
                return
            self._level = level
            if self._playing != "buz":         # 알림음은 끝까지 재생, 끝나면 새 단계음
                self._kill()                   # 바뀌면 지금 소리를 바로 끊음
            self._cond.notify_all()

    def _kill(self):
        p = self._proc
        if p is not None and p.poll() is None:
            try: p.terminate()
            except Exception: pass

    def _loop(self):
        while True:
            with self._cond:
                while self._level == 0 and not self._buz_pending and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                if self._buz_pending:
                    self._buz_pending = False
                    lv = "buz"
                else:
                    lv = self._level
                self._playing = lv
                try:
                    self._proc = subprocess.Popen(["aplay", "-q", "-D", SOUND_DEVICE, self.files[lv]],
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                except OSError as e:
                    print(f"⚠️  [경고음] 재생 실패: {e} → 경고음 끔")
                    self.enabled = False
                    return
                proc = self._proc
            _, err = proc.communicate()        # 소리 하나 끝날 때까지 (끊기면 바로 반환)
            with self._cond:
                self._playing = 0
            if proc.returncode not in (0, -15) and (lv == "buz" or self._level == lv):
                self._fails += 1
                msg = err.decode(errors="ignore").strip().splitlines()
                print(f"⚠️  [경고음] aplay 오류: {msg[-1] if msg else proc.returncode}")
                if self._fails >= 3:
                    print(f"⚠️  [경고음] 계속 실패해서 끔 — SOUND_DEVICE({SOUND_DEVICE}) 확인 (aplay -l)")
                    self.enabled = False
                    return
                time.sleep(0.5)
            elif proc.returncode == 0:
                self._fails = 0

    def stop(self):
        if not self.enabled:
            return
        with self._cond:
            self._stop, self._level = True, 0
            self._kill()
            self._cond.notify_all()


_alarm_sound = None    # _bt_restart()에서 재시작 전에 소리를 끄기 위한 참조


# ══════════════════════════════════════════
# 6-4. 전시용 웹 UI — Flask (영상 MJPEG + 상태 JSON), 화면은 exhibit_ui.html
# ══════════════════════════════════════════
class WebUI:
    """
    메인 루프는 update()로 '깨끗한 프레임(오버레이 없는 영상)'과 상태만 넘김 → 무거운 일은 여기 스레드에서.
      · JPEG 압축은 보는 사람이 있을 때만, WEB_STREAM_FPS로 한 번만 해서 모든 접속자가 같이 씀
      · /            → exhibit_ui.html
      · /stream.mjpg → 카메라 영상
      · /api/state   → 상태 JSON (페이지가 0.2초마다 가져감)
    Flask가 없으면 경고만 출력하고 꺼짐 (본 기능엔 영향 없음).
    """

    def __init__(self):
        self.enabled   = False
        self._lock     = threading.Lock()
        self._cond     = threading.Condition(self._lock)
        self._frame    = None          # 최신 깨끗한 프레임 (BGR)
        self._frame_id = 0
        self._jpeg     = None
        self._jpeg_id  = 0
        self._clients  = 0
        self._last_enc = 0.0
        self._state    = {"ready": False}
        self._events   = deque(maxlen=40)   # (번호, 시각, 종류, 글)
        self._event_no = 0
        if not WEB_UI_ENABLE:
            return
        try:
            import flask  # noqa: F401
        except ImportError:
            print("⚠️  [웹 UI] Flask가 없어 웹 화면 없이 동작합니다 (pip3 install flask==2.0.3)")
            return
        self.html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exhibit_ui.html")
        if not os.path.isfile(self.html_path):
            print(f"⚠️  [웹 UI] {self.html_path} 없음 → 웹 화면 끔 (파이썬 파일과 같은 폴더에 두세요)")
            return
        self.enabled = True

    # ── 메인 루프에서 호출 ──
    def wants_frame(self) -> bool:
        """영상을 보는 사람이 있고, 다음 전송 시각이 됐을 때만 True → 그때만 frame.copy()."""
        return (self.enabled and self._clients > 0
                and time.time() - self._last_enc >= 1.0 / WEB_STREAM_FPS)

    def update(self, frame=None, **state):
        if not self.enabled:
            return
        with self._cond:
            if frame is not None:
                self._frame = frame
                self._frame_id += 1
                self._last_enc = time.time()
                self._cond.notify_all()
            state["ready"] = True
            state["t"] = time.time()
            self._state = state

    def event(self, kind: str, text: str):
        """화면 오른쪽 아래 기록에 남길 일 (상태 변경, 핸들 이탈 등)."""
        if not self.enabled:
            return
        with self._lock:
            self._event_no += 1
            self._events.append({"no": self._event_no, "t": time.time(), "kind": kind, "text": text})

    # ── 서버 ──
    def start(self):
        if not self.enabled:
            return self
        threading.Thread(target=self._encode_loop, daemon=True).start()
        threading.Thread(target=self._serve, daemon=True).start()
        ips = _local_ips()
        print(f"✅ [웹 UI] http://localhost:{WEB_PORT}"
              + (f"  (같은 네트워크: {', '.join(f'http://{ip}:{WEB_PORT}' for ip in ips)})" if ips else ""))
        if WEB_UI_AUTO_OPEN:
            self._open_kiosk()
        return self

    def _open_kiosk(self):
        """start_exhibit_ui.sh를 백그라운드로 실행 → 두 번째 모니터에 Chromium 전체 화면."""
        sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "start_exhibit_ui.sh")
        if not os.environ.get("DISPLAY"):
            print("⚠️  [웹 UI] 화면(DISPLAY)이 없어 자동으로 띄우지 않음 — 다른 기기 브라우저로 접속하세요")
            return
        if not os.path.isfile(sh):
            print(f"⚠️  [웹 UI] {sh} 없음 → 자동으로 띄우지 않음")
            return
        try:
            subprocess.Popen(["bash", sh] + ([WEB_UI_MONITOR] if WEB_UI_MONITOR else []))
        except OSError as e:
            print(f"⚠️  [웹 UI] 자동 실행 실패: {e}")

    def _encode_loop(self):
        last = 0
        params = [int(cv2.IMWRITE_JPEG_QUALITY), WEB_JPEG_QUALITY]
        while is_running:
            with self._cond:
                while self._frame_id == last and is_running:
                    self._cond.wait(0.5)
                frame, last = self._frame, self._frame_id
            if frame is None:
                continue
            if WEB_STREAM_WIDTH and frame.shape[1] > WEB_STREAM_WIDTH:
                hh = int(frame.shape[0] * WEB_STREAM_WIDTH / frame.shape[1])
                frame = cv2.resize(frame, (WEB_STREAM_WIDTH, hh), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, params)
            if ok:
                with self._cond:
                    self._jpeg, self._jpeg_id = buf.tobytes(), self._jpeg_id + 1
                    self._cond.notify_all()

    def _mjpeg(self):
        with self._lock:
            self._clients += 1
        last = 0
        try:
            while is_running:
                with self._cond:
                    while self._jpeg_id == last and is_running:
                        if not self._cond.wait(2.0):
                            break
                    jpg, last = self._jpeg, self._jpeg_id
                if jpg is None:
                    time.sleep(0.05)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        finally:
            with self._lock:
                self._clients -= 1

    def _serve(self):
        import logging
        from flask import Flask, Response, jsonify, request
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app = Flask(__name__)

        @app.route("/")
        def index():
            with open(self.html_path, "rb") as f:
                return Response(f.read(), mimetype="text/html; charset=utf-8")

        @app.route("/stream.mjpg")
        def stream():
            return Response(self._mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame",
                            headers={"Cache-Control": "no-store"})

        @app.route("/api/state")
        def api_state():
            since = request.args.get("since", default=0, type=int)
            with self._lock:
                st = dict(self._state)
                st["events"] = [e for e in self._events if e["no"] > since]
            st["now"] = time.time()
            return jsonify(st)

        try:
            app.run(host="0.0.0.0", port=WEB_PORT, threaded=True, use_reloader=False)
        except OSError as e:
            print(f"⚠️  [웹 UI] 서버 시작 실패: {e} (포트 {WEB_PORT}를 다른 프로그램이 쓰는 중?)")
            self.enabled = False


def _local_ips():
    """같은 네트워크에서 접속할 주소 안내용."""
    ips = []
    try:
        out = subprocess.run(["hostname", "-I"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=2).stdout.decode()
        ips = [ip for ip in out.split() if ":" not in ip and not ip.startswith("172.17.")]
    except Exception:
        pass
    return ips[:2]


_web_ui = None


# ══════════════════════════════════════════
# 7. 화면 오버레이
# ══════════════════════════════════════════
STATE_COLORS = {
    "OFF"        : (50,  205, 50),
    "LV1_WARN"   : (0,   165, 255),
    "LV2_DANGER" : (0,   0,   255),
}

def _draw_buz_banner(frame):
    """ESP32에서 !BUZ#가 들어온 뒤 BUZ_BANNER_SEC 동안 화면 아래쪽에 'HANDS ON WHEEL!' (깜빡임)."""
    if buz_last_time <= 0 or time.time() - buz_last_time > BUZ_BANNER_SEC:
        return
    if int(time.time() * 3) % 2:
        return
    h, w = frame.shape[:2]
    text = "HANDS ON WHEEL!"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    x, y = (w - tw) // 2, h - 110
    cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 200), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)


def draw_overlay(frame, state, fps, perclos, ear, threshold,
                 is_calibrated, calib_rem, bt_ok,
                 proc_ms=None, head_info=None, cause="", calib_phase="open", closed_dur=0.0,
                 cam_fps=None):
    h, w = frame.shape[:2]
    _draw_buz_banner(frame)
    # 상단 바 반투명 처리 — 바 영역(56줄)만 계산
    # (예전: 프레임 전체 복사 + 전체 합성. 결과 픽셀은 동일: 0.6×20 + 0.4×원본)
    frame[:56] = cv2.convertScaleAbs(frame[:56], alpha=0.4, beta=12.0)

    fps_text = f"FPS {int(fps)}"
    cv2.putText(frame, fps_text, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    if proc_ms is not None:   # 프레임당 처리 시간 → 33ms(30fps)에서 이 값을 뺀 만큼이 여유
        tw = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
        extra = f"  cam {cam_fps:.0f}" if cam_fps else ""     # cam = 카메라가 실제로 주는 fps
        cv2.putText(frame, f"{proc_ms:.0f}ms{extra}", (18 + tw, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # 블루투스 / 시뮬레이션 상태 표시
    with sim_mode_lock:
        is_sim = sim_mode
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
        # 2단계 캘리브레이션 안내: 정면 보고 눈 뜨기 → 눈 감기
        if calib_phase == "closed":
            msg, col = f"CLOSE EYES {calib_rem:.1f}s", (0, 160, 255)
        else:
            msg, col = f"LOOK AHEAD {calib_rem:.1f}s", (0, 220, 220)
        cv2.putText(frame, msg, (w // 2 - 130, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        return

    # 상태
    cv2.putText(frame, state, (w // 2 - 120, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, STATE_COLORS.get(state, (255, 255, 255)), 3)

    # 하단 정보 바
    bar_y = h - 80
    cv2.rectangle(frame, (0, bar_y - 5), (w, h), (20, 20, 20), -1)

    # PERCLOS 게이지
    pbar_w = int(min(perclos, 1.0) * (w - 20))
    cv2.rectangle(frame, (10, bar_y + 5),
                  (10 + pbar_w, bar_y + 22), STATE_COLORS.get(state, (50, 205, 50)), -1)
    cv2.rectangle(frame, (10, bar_y + 5), (w - 10, bar_y + 22), (120, 120, 120), 1)
    for ratio, lv_color in [(PERCLOS_LV1, (0, 165, 255)), (PERCLOS_LV2, (0, 0, 255))]:
        lx = int(ratio * (w - 20)) + 10
        cv2.line(frame, (lx, bar_y + 3), (lx, bar_y + 24), lv_color, 2)
    cv2.putText(frame, f"PERCLOS({PERCLOS_WINDOW_SEC:.0f}s) {perclos:.1%}",
                (10, bar_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # EAR / 감김 기준 / 지금 이어지는 눈 감김 시간
    ear_text = f"EAR {ear:.3f}  THR {threshold:.3f}"
    if closed_dur > 0:
        ear_text += f"  CLOSED {closed_dur:.1f}s"
    cv2.putText(frame, ear_text,
                (10, bar_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # R키 안내
    cv2.putText(frame, "R: recalibrate  B: buz test", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    # 하단 바 오른쪽: 경보 원인 / 머리 각도(정면 대비) / 최근 30초 이탈 누적
    if cause:
        cv2.putText(frame, cause, (w // 2 + 10, bar_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, STATE_COLORS.get(state, (255, 255, 255)), 2)
    if head_info is not None:
        dyaw, dpitch, vats = head_info
        cv2.putText(frame, f"HEAD Y{round(dyaw):+d} P{round(dpitch):+d}  OFF {vats:.0f}/{VATS_WINDOW_SEC:.0f}s",
                    (w // 2 + 10, bar_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def draw_no_face(frame, is_calibrated: bool, missing_sec: float):
    """얼굴 미감지 시 화면 중앙에 상태 메시지 출력."""
    if not is_calibrated:
        cv2.putText(frame, "WAITING FOR FACE...", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2)
        return
    if missing_sec >= FACE_MISSING_DANGER_SEC:
        cv2.putText(frame, "NO FACE — DANGER!", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        remain = int(FACE_MISSING_DANGER_SEC - missing_sec)
        cv2.putText(frame, f"FACE MISSING... {remain}s", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

# ══════════════════════════════════════════
# 9. 메인 루프
# ══════════════════════════════════════════
def main():
    global target_state, is_running, _alarm_sound, _web_ui

    # 모델 파일 사전 점검 (ONNX 또는 이미 변환된 엔진 중 하나는 있어야 함)
    for model_path in (FACE_DET_MODEL, LANDMARK_MODEL):
        if not os.path.exists(model_path) and not os.path.exists(_engine_path(model_path)):
            print(f"\n❌ 모델 파일 없음: {model_path}")
            print("아래 명령어로 다운로드 후 재실행하세요:")
            print("  wget https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB"
                  "/raw/master/models/onnx/version-RFB-320.onnx")
            print("  wget https://github.com/cunjian/pytorch_face_landmark"
                  "/raw/master/onnx/landmark_detection_56_se_external.onnx")
            sys.exit(1)

    # 엔진 준비 (첫 실행은 빌드에 수 분 걸리므로 카메라보다 먼저)
    os.makedirs(TRT_CACHE_DIR, exist_ok=True)
    print(f"TensorRT {trt.__version__}")
    try:
        face_detector   = FaceDetector()
        landmark_detect = LandmarkDetector()
    except Exception as e:
        print(f"\n❌ TensorRT 엔진 준비 실패: {e}")
        print("   얼굴 검출 모델 변환이 실패했다면 version-RFB-320_without_postprocessing.onnx 로")
        print("   바꿔 보세요 (파일 맨 위 설명 참고).")
        sys.exit(1)
    print("✅ TensorRT 추론 엔진 준비 완료")

    vs = CameraStream().start()
    time.sleep(1.0)                               # 웹캠 자동 노출이 자리 잡을 시간

    ema          = EMAFilter(EMA_ALPHA)
    judge        = DrowsinessJudge()          # 졸음 판정 (연속 눈 감김 + PERCLOS, 시간 기준)
    adaptive_thr = AdaptiveThreshold(
        RECALIB_INTERVAL, RECALIB_WINDOW_SEC, FPS_TARGET, RECALIB_MAX_DRIFT,
    )
    print(f"캘리브레이션: 정면 보고 눈 뜨기 {CALIB_OPEN_SEC:.0f}초"
          + (f" → 화면에 CLOSE EYES가 뜨면 눈 감기 {CALIB_CLOSED_SEC:.0f}초" if CALIB_CLOSED_SEC > 0 else ""))
    head_est = HeadPoseEstimator(vs.width, vs.height) if HEAD_POSE_ENABLE else None
    distract = DistractionMonitor()
    sound    = AlarmSound().start()                              # 디스플레이 스피커 경고음
    _alarm_sound = sound
    web      = WebUI().start()                                   # 전시용 웹 UI (두 번째 모니터·노트북)
    _web_ui  = web
    fps_ema  = EMAFilter(0.1)     # 화면 표시용 (순간값은 출렁여서 평균)
    proc_ema = EMAFilter(0.1)     # 프레임당 처리 시간(ms)

    state                 = "OFF"
    prev_state            = "OFF"
    face_missing_start    = 0.0
    prev_time             = None
    was_calibrated        = False
    perc_announced        = 0       # 경고음으로 이미 알린 PERCLOS 단계
    perc_sound_until      = 0.0     # PERCLOS 경고음을 이 시각까지만
    last_frame_id         = -1

    print("── 시작 (종료: q) ──")

    try:
        while True:
            # 새 프레임만 처리 — 같은 프레임을 두 번 처리하면 '프레임 수' 기준 판정이 빨라짐
            frame_id, frame = vs.read(last_frame_id)
            if frame is None:
                if frame_id <= 0:            # 카메라에서 아직 한 장도 못 받음
                    time.sleep(0.01)
                continue
            last_frame_id = frame_id

            curr_time = time.time()
            if prev_time is not None:
                fps_ema.update(1.0 / max(curr_time - prev_time, 1e-6))
            prev_time = curr_time
            fps       = fps_ema.value

            h, w, _ = frame.shape
            web_frame = frame.copy() if web.wants_frame() else None   # 웹 UI용 오버레이 없는 영상

            try:
                bboxes = face_detector.detect(frame)
            except Exception as e:
                print(f"⚠️  얼굴 검출 예외: {e}")
                continue

            ema_ear      = ema.value
            perclos      = 0.0
            d_level      = distract.level     # 전방 주시 태만 단계 (0/1/2)
            face_box     = None               # 웹 UI 얼굴 상자 (0~1 비율)
            missing_sec  = 0.0
            acute_lv     = 0                  # 계속 울릴 경보 단계 (눈 감김·주시 태만·얼굴 없음)
            perc_lv      = 0                  # PERCLOS 단계 (경고음은 짧게 한 번)
            drowsy_cause = ""                 # 이번 프레임 졸음 경보 원인 (화면 표시용)
            head         = None

            if bboxes:
                face_missing_start = 0.0

                # 가장 큰 얼굴(운전자) 선택
                bbox = max(bboxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                face_box = [round(float(bbox[0]) / w, 4), round(float(bbox[1]) / h, 4),
                            round(float(bbox[2]) / w, 4), round(float(bbox[3]) / h, 4)]

                try:
                    landmarks = landmark_detect.predict(frame, bbox)
                except Exception as e:
                    print(f"⚠️  랜드마크 예외: {e}")
                    landmarks = None

                if landmarks is not None:
                    # ── 머리 자세 → 전방 주시 태만 ──
                    if head_est is not None:
                        head = head_est.estimate(landmarks)
                    if head is not None:
                        if adaptive_thr.is_calibrated:
                            d_level = distract.update(head[0], head[1], curr_time)
                        else:
                            distract.add_calib(head[0], head[1])   # 캘리브레이션 동안 정면 수집

                    left_pts  = landmarks[LEFT_EYE]
                    right_pts = landmarks[RIGHT_EYE]
                    raw_ear   = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                    # ── 측정이 끝난 뒤 표시 ──
                    if head is not None and DRAW_HEAD_AXIS:
                        p0, p1 = head_est.nose_line()
                        cv2.line(frame, p0, p1, (255, 200, 0), 2)
                    if DRAW_EYE_POINTS:
                        for px, py in np.vstack([left_pts, right_pts]):
                            cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 255), -1)

                    if raw_ear <= 0.0 or raw_ear >= 1.0:
                        continue

                    ema_ear = ema.update(raw_ear)
                    # 고개를 돌린 동안의 EAR은 장기 재보정 버퍼에 넣지 않음
                    is_cal  = adaptive_thr.update(ema_ear, judge.perclos, on_road=not distract.off_road)

                    if is_cal and adaptive_thr.threshold:
                        # 졸음 판정 (시간 기준): 연속 눈 감김 1초 LV1 / 2초 LV2 + 최근 30초 PERCLOS
                        lv = judge.update(ema_ear, adaptive_thr.threshold, adaptive_thr.thr_open, curr_time)
                        perclos      = judge.perclos
                        state        = LEVEL_STATE[lv]
                        drowsy_cause = judge.cause
                        acute_lv     = judge.lv_close
                        perc_lv      = judge.lv_perc
                    else:
                        state = "OFF"

                # 졸음 판정과 전방 주시 태만 중 더 위험한 단계로
                state = _worse(state, d_level, adaptive_thr.is_calibrated)
                if adaptive_thr.is_calibrated:
                    acute_lv = max(acute_lv, d_level)

                bt_ok = sock is not None      # 락 안 잡음 — BT 재시도 중에도 화면이 멈추지 않게

                draw_overlay(
                    frame, state, fps, perclos, ema_ear,
                    adaptive_thr.threshold or 0.0,
                    adaptive_thr.is_calibrated,
                    adaptive_thr.calib_remaining(),
                    bt_ok,
                    proc_ms=proc_ema.value,
                    cam_fps=getattr(vs, "cam_fps", None),
                    head_info=((distract.dyaw, distract.dpitch, distract.vats_sum)
                               if head is not None and distract.base is not None else None),
                    cause=_cause_text(drowsy_cause, d_level, distract),
                    calib_phase=adaptive_thr.phase,
                    closed_dur=judge.closed_dur if adaptive_thr.is_calibrated else 0.0,
                )

            else:
                adaptive_thr.update(0.0, 0.0)
                if head_est is not None:
                    head_est.reset()
                if adaptive_thr.is_calibrated:
                    d_level = distract.update_missing(curr_time)   # 고개를 돌리다 놓쳤으면 이탈 시간 계속

                if face_missing_start == 0.0:
                    face_missing_start = curr_time
                missing_sec = curr_time - face_missing_start

                state = ("LV2_DANGER"
                         if adaptive_thr.is_calibrated and missing_sec >= FACE_MISSING_DANGER_SEC
                         else "OFF")
                state = _worse(state, d_level, adaptive_thr.is_calibrated)
                acute_lv = SEVERITY.get(state, 0)

                bt_ok = sock is not None      # 락 안 잡음 — BT 재시도 중에도 화면이 멈추지 않게

                draw_no_face(frame, adaptive_thr.is_calibrated, missing_sec)
                perclos = judge.perclos
                draw_overlay(
                    frame, state, fps, perclos, 0.0, adaptive_thr.threshold or 0.0,
                    adaptive_thr.is_calibrated,
                    adaptive_thr.calib_remaining(),
                    bt_ok,
                    proc_ms=proc_ema.value,
                    cam_fps=getattr(vs, "cam_fps", None),
                    cause=_cause_text("", d_level, distract),
                    calib_phase=adaptive_thr.phase,
                )

            # ── 디스플레이 스피커 경고음 (BT 모드와 무관) ──
            #   눈 감김·주시 태만·얼굴 없음 → 그 상태인 동안 계속
            #   PERCLOS만으로 올라간 경보 → 단계가 올라갈 때 PERCLOS_SOUND_SEC만 울리고 화면 표시만 유지
            if perc_lv > perc_announced:
                perc_sound_until = curr_time + PERCLOS_SOUND_SEC
            perc_announced = perc_lv            # 내려갔다가 다시 오르면 다시 알림
            sound_lv = acute_lv
            if curr_time < perc_sound_until:
                sound_lv = max(sound_lv, perc_lv)
            sound.set_level(sound_lv)

            # ── 전시용 웹 UI에 상태 전달 ──
            if web.enabled:
                calibrated = adaptive_thr.is_calibrated
                causes = []
                if calibrated:
                    if bboxes and judge.lv_close > 0:
                        causes.append("eyes")
                    if bboxes and judge.lv_perc > 0:
                        causes.append("perclos")
                    if d_level > 0:
                        causes.append("offroad" if distract.off_dur >= OFFROAD_LV1_SEC else "distract")
                    if not bboxes and missing_sec >= FACE_MISSING_DANGER_SEC:
                        causes.append("noface")
                if calibrated and not was_calibrated:
                    web.event("calib", "캘리브레이션 완료")
                was_calibrated = calibrated
                with sim_mode_lock:
                    sim_now = sim_mode
                web.update(
                    web_frame,
                    level=SEVERITY.get(state, 0), state=state, causes=causes,
                    calibrated=calibrated, calib_phase=adaptive_thr.phase,
                    calib_rem=round(adaptive_thr.calib_remaining(), 2),
                    calib_open_sec=CALIB_OPEN_SEC, calib_closed_sec=CALIB_CLOSED_SEC,
                    face=bool(bboxes), face_box=face_box, missing_sec=round(missing_sec, 1),
                    face_missing_sec=FACE_MISSING_DANGER_SEC,
                    ear=round(float(ema_ear or 0.0), 4),
                    thr=round(float(adaptive_thr.threshold or 0.0), 4),
                    thr_open=round(float(adaptive_thr.thr_open or 0.0), 4),
                    baseline=round(float(adaptive_thr.baseline or 0.0), 4),
                    closed=bool(judge.closed) if bboxes else False,
                    closed_dur=round(judge.closed_dur, 2) if calibrated and bboxes else 0.0,
                    eye_lv1_sec=EYE_CLOSED_LV1_SEC, eye_lv2_sec=EYE_CLOSED_LV2_SEC,
                    perclos=round(float(judge.perclos), 4), perclos_window=PERCLOS_WINDOW_SEC,
                    perclos_lv1=PERCLOS_LV1, perclos_lv2=PERCLOS_LV2,
                    head=distract.base is not None and head is not None,
                    yaw=round(float(distract.dyaw), 1), pitch=round(float(distract.dpitch), 1),
                    off_road=bool(distract.off_road), off_dur=round(distract.off_dur, 1),
                    vats=round(distract.vats_sum, 1), vats_window=VATS_WINDOW_SEC,
                    fps=round(fps or 0.0, 1), cam_fps=round(float(getattr(vs, "cam_fps", 0.0) or 0.0), 1),
                    proc_ms=round(proc_ema.value or 0.0, 1),
                    bt=sock is not None, sim=sim_now,
                    hands_off=buz_last_time > 0 and time.time() - buz_last_time < BUZ_BANNER_SEC,
                    sound=sound_lv,
                )

            # 블루투스 상태 동기화
            if state != prev_state:
                print(f"[상태 변경] {prev_state} → {state}  (PERCLOS={perclos:.1%})")
                web.event(state, {"OFF": "정상으로 복귀", "LV1_WARN": "주의 단계",
                                  "LV2_DANGER": "위험 단계"}.get(state, state))
                with sim_mode_lock:
                    is_sim = sim_mode
                if is_sim:
                    _sim_log(state, perclos)   # 시뮬레이션: 콘솔 기록
                else:
                    with bt_lock:              # 실제 모드: ESP32 전송
                        target_state = state
                prev_state = state

            proc_ema.update((time.time() - curr_time) * 1000.0)   # 화면 출력 전까지 처리 시간
            cv2.imshow("Jetson Drowsiness System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                print("🔔 [B키] ESP32 !BUZ# 알림음 테스트")
                if time.time() - buz_last_time > BUZ_BANNER_SEC:
                    web.event("hands", "핸들에서 손을 뗌 (B키 테스트)")
                globals()["buz_last_time"] = time.time()
                sound.play_buz()
            elif key == ord("r"):
                # 새 탑승자 → 캘리브레이션 전체 리셋
                adaptive_thr.reset()
                ema._value            = None
                judge.reset()
                state                 = "OFF"
                prev_state            = "OFF"
                face_missing_start    = 0.0
                perc_announced, perc_sound_until = 0, 0.0
                web.event("calib", "새 탑승자 — 캘리브레이션 다시 시작")
                distract.reset()                  # 머리 자세 '정면'도 다시 잡음
                if head_est is not None:
                    head_est.reset()
                with bt_lock:
                    target_state = "OFF"
                sound.set_level(0)
                print(f"🔄 [R키] 새 탑승자 캘리브레이션 시작 — 정면 보고 눈 뜨기 {CALIB_OPEN_SEC:.0f}초"
                      + (f" → CLOSE EYES가 뜨면 {CALIB_CLOSED_SEC:.0f}초 감기" if CALIB_CLOSED_SEC > 0 else ""))

    finally:
        is_running = False
        if _alarm_sound is not None:
            _alarm_sound.stop()
        if WEB_UI_AUTO_OPEN and web.enabled:            # 자동으로 띄운 UI 창도 같이 닫기
            subprocess.run(["pkill", "-f", "user-data-dir=/tmp/exhibit-ui"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vs.stop()
        with sock_lock:
            if sock:
                try: _bt_write(sock, "!OFF#\n")
                except: pass
                try: sock.close()
                except: pass
        cv2.destroyAllWindows()
        print("── 종료 ──")

# ══════════════════════════════════════════
# 10. 진입점
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("=== 젯슨 나노 졸음 감지 시스템 (TensorRT 버전) 시작 ===")
    _init_bluetooth()
    threading.Thread(target=bluetooth_thread, daemon=True).start()
    threading.Thread(target=bt_reader_thread, daemon=True).start()
    main()
