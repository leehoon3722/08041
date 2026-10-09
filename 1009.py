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

[EAR 알고리즘]
  EMA 스무딩 / PERCLOS / 즉각 트리거 / 적응형 임계값
"""

import sys
import os
import glob
import ctypes
import ctypes.util
import socket
import time
import threading
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

# 모델 경로 (ONNX) — 첫 실행 시 TensorRT 엔진으로 변환해 TRT_CACHE_DIR에 저장
FACE_DET_MODEL  = "version-RFB-320.onnx"                    # 얼굴 bbox 검출
LANDMARK_MODEL  = "landmark_detection_56_se_external.onnx"  # 68점 랜드마크 (56×56 입력)

# TensorRT 설정
TRT_CACHE_DIR   = "./trt_cache"   # 변환된 엔진(.engine) 저장 위치
TRT_FP16        = True            # FP16 추론 (젯슨 나노 속도 향상)
TRT_WORKSPACE   = 1 << 28         # 엔진 빌드 작업 메모리 256MB

# 화면에 눈 랜드마크 12점 표시 (랜드마크가 눈에 제대로 붙는지 확인용, 끄려면 False)
DRAW_EYE_POINTS = True

# 68점 랜드마크 눈 인덱스
LEFT_EYE  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

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
# 6. PERCLOS 계산기
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
    cv2.putText(frame, f"PERCLOS {perclos:.1%}",
                (10, bar_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # EAR / Threshold
    cv2.putText(frame, f"EAR {ear:.3f}  THR {threshold:.3f}",
                (10, bar_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # R키 안내
    cv2.putText(frame, "R: recalibrate", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)


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
    global target_state, is_running

    check_opencv_gstreamer()

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

    vs = CSICameraStream().start()
    time.sleep(2.0)
    print("✅ CSI 카메라 시작")

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

            try:
                bboxes = face_detector.detect(frame)
            except Exception as e:
                print(f"⚠️  얼굴 검출 예외: {e}")
                continue

            ema_ear = ema.value
            perclos = 0.0

            if bboxes:
                face_missing_start = 0.0

                # 가장 큰 얼굴(운전자) 선택
                bbox = max(bboxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

                try:
                    landmarks = landmark_detect.predict(frame, bbox)
                except Exception as e:
                    print(f"⚠️  랜드마크 예외: {e}")
                    landmarks = None

                if landmarks is not None:
                    left_pts  = landmarks[LEFT_EYE]
                    right_pts = landmarks[RIGHT_EYE]
                    raw_ear   = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                    if DRAW_EYE_POINTS:
                        for px, py in np.vstack([left_pts, right_pts]):
                            cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 255), -1)

                    if raw_ear <= 0.0 or raw_ear >= 1.0:
                        continue

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
    print("=== 젯슨 나노 졸음 감지 시스템 (TensorRT 버전) 시작 ===")
    _init_bluetooth()
    threading.Thread(target=bluetooth_thread, daemon=True).start()
    main()
