#!/bin/bash
# 전시용 웹 UI를 전체 화면(키오스크)으로 띄움.
#   UI_LAYOUT=mirror (기본): 터치스크린에 UI(재측정 버튼 포함) 하나를 띄우고, 두 번째 모니터는 같은 화면(미러링)
#   UI_LAYOUT=dual         : 터치스크린과 두 번째 모니터에 UI를 따로 띄움 (브라우저 2개 → 더 무거움)
#   보통은 python3 webpage.py 가 자동으로 이 스크립트를 실행함 (WEB_UI_AUTO_OPEN=True)
#   직접 실행할 때: 프로그램을 켠 상태에서 '다른 터미널'에서
#       bash start_exhibit_ui.sh            (모니터 자동 선택)
#       bash start_exhibit_ui.sh DP-0       (두 번째 모니터 이름 직접 지정 — xrandr 로 확인)
#   두 모니터가 같은 화면(미러링)이면 옆으로 펼쳐서 띄우고, UI 창이 모두 닫히면 원래대로 돌려놓음.
# 끄기: 프로그램 종료(q) 하면 같이 닫힘, 또는 각 UI 화면에서 Alt+F4

URL="http://localhost:5000"
PROFILE=/tmp/exhibit-ui            # 창마다 /tmp/exhibit-ui-touch, /tmp/exhibit-ui-main 사용

# ── 설정 ──
UI_LAYOUT="${UI_LAYOUT:-mirror}"   # mirror: 두 모니터 같은 화면 / dual: 따로따로
UI_MIRROR_SCALE=1           # mirror에서 두 번째 모니터 해상도가 터치스크린과 다르면 터치스크린 화면을 꽉 차게 늘려서 보여 줌
UI_TOUCH="${UI_TOUCH:-1}"   # (dual) 1: 터치스크린에도 UI 띄움 / 0: 두 번째 모니터에만
UI_RES="1280x720"           # (dual) 두 번째 모니터 해상도 (작을수록 가벼움). 원래대로: ""
UI_GPU=1                    # 1: 브라우저 GPU 사용 / 0: GPU 끔 — 둘 다 해 보고 빠른 쪽
UI_NICE=10                  # 브라우저 우선순위 낮춤 → 졸음 판정 프로그램이 CPU를 먼저 씀 (0~19)
TOUCH_DEVICE=""             # 터치스크린 입력 장치 이름 (xinput list 로 확인). 빈칸이면 이름으로 자동 찾기

# 예전 실행에서 남은 UI 창(연결 실패 화면 등)이 있으면 닫고 새로 띄움
if pgrep -f "user-data-dir=$PROFILE" >/dev/null; then
  echo "[웹 UI] 남아 있던 UI 창 닫고 새로 띄움"
  pkill -f "user-data-dir=$PROFILE"
  sleep 2
fi

# 프로그램의 웹 서버가 뜰 때까지 대기 (학교·회사망 프록시 설정이 있어도 localhost는 직접 접속)
READY=0
for i in $(seq 1 120); do
  if python3 -c "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('$URL/api/state', timeout=1)" 2>/dev/null; then
    READY=1; break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  echo "❌ [웹 UI] 웹 서버($URL)에 접속이 안 됨 — 프로그램 콘솔에 '[웹 UI]' 오류가 있는지 확인하세요"
  exit 1
fi

# 연결된 모니터 목록 (첫 번째 = primary가 있으면 primary = 터치스크린)
mapfile -t OUTS < <(xrandr --query | awk '/ connected primary/ {print $1}'; xrandr --query | awk '/ connected/ && !/ connected primary/ {print $1}')
MAIN="${OUTS[0]}"
OUT="${1:-${OUTS[1]}}"
if [ "${#OUTS[@]}" -lt 2 ] && [ -z "$1" ]; then
  echo "⚠️  [웹 UI] 연결된 모니터가 ${#OUTS[@]}개뿐 — 두 번째 모니터 케이블/전원을 확인하세요"
  OUT=""
fi

geom() {   # 모니터 위치 "가로x세로+X+Y"
  xrandr --query | awk -v o="$1" '$1==o {for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+x[0-9]+\+[0-9]+\+[0-9]+$/) {print $i; exit}}'
}

modes_of() {   # 모니터가 지원하는 해상도 목록
  xrandr --query | awk -v o="$1" '$1==o {f=1; next} /^[^ ]/ {f=0} f {print $1}'
}

MIRRORED=0
if [ "$UI_LAYOUT" == "mirror" ] && [ -n "$OUT" ] && [ "$OUT" != "$MAIN" ]; then
  # 두 번째 모니터를 터치스크린과 같은 화면으로 (예전 실행에서 옆으로 펼쳐 둔 것도 되돌림)
  G_OUT=$(geom "$OUT"); G_MAIN=$(geom "$MAIN"); M="${G_MAIN%%+*}"
  if [ "${G_OUT#*+}" != "${G_MAIN#*+}" ] || [ "${G_OUT%%+*}" != "$M" ]; then
    if modes_of "$OUT" | grep -qx "$M"; then
      xrandr --output "$OUT" --mode "$M" --same-as "$MAIN" && echo "[웹 UI] $OUT → $MAIN 과 같은 화면 ($M)"
    elif [ "$UI_MIRROR_SCALE" == "1" ] && xrandr --output "$OUT" --auto --same-as "$MAIN" --scale-from "$M" 2>/dev/null; then
      echo "[웹 UI] $OUT → $MAIN 과 같은 화면 ($M 화면을 모니터 크기에 맞춰 늘림)"
    else
      xrandr --output "$OUT" --auto --same-as "$MAIN"
      echo "[웹 UI] $OUT → $MAIN 과 같은 화면 (해상도가 달라 두 번째 모니터엔 왼쪽 위 ${M} 영역에 UI가 보일 수 있음)"
    fi
    sleep 2
  fi
fi
if [ "$UI_LAYOUT" == "dual" ] && [ -n "$OUT" ] && [ "$OUT" != "$MAIN" ]; then
  G_OUT=$(geom "$OUT"); G_MAIN=$(geom "$MAIN")
  # 같은 위치(+X+Y)면 미러링 상태 → 옆으로 펼침
  if [ -z "$G_OUT" ] || [ "${G_OUT#*+}" == "${G_MAIN#*+}" ]; then
    echo "[웹 UI] $OUT 가 $MAIN 과 같은 화면이라 오른쪽으로 펼침"
    xrandr --output "$OUT" --auto --right-of "$MAIN"
    MIRRORED=1
    sleep 2
  fi
  if [ -n "$UI_RES" ]; then
    if xrandr --query | awk -v o="$OUT" '$1==o {f=1; next} /^[^ ]/ {f=0} f' | grep -q " $UI_RES"; then
      xrandr --output "$OUT" --mode "$UI_RES" --right-of "$MAIN" && [ "$MIRRORED" == "0" ] && MIRRORED=2
      sleep 2
    else
      echo "[웹 UI] $OUT 가 $UI_RES 를 지원하지 않아 해상도 유지"
    fi
  fi
fi

# ── 터치 입력을 터치스크린에만 연결 ──
# 모니터를 옆으로 펼치면 터치 좌표가 두 화면 전체(예: 1024+1280 폭)로 늘어나서,
# 터치스크린 오른쪽을 누르면 두 번째 모니터 쪽이 눌림 → 재측정 버튼이 안 먹는 원인
map_touch() {
  command -v xinput >/dev/null || { echo "⚠️  [터치] xinput 없음 — sudo apt install xinput"; return; }
  local ids
  if [ -n "$TOUCH_DEVICE" ]; then
    ids=$(xinput list | grep -F "$TOUCH_DEVICE" | grep -o 'id=[0-9]*' | cut -d= -f2)
  else
    ids=$(xinput list | grep -iE 'touch|ilitek|waveshare|egalax|goodix|ft5|qdtech|wch|usb2iic|hid 222a|multitouch' \
          | grep -iv 'keyboard' | grep -o 'id=[0-9]*' | cut -d= -f2)
  fi
  if [ -z "$ids" ]; then
    [ "$1" != "quiet" ] && echo "⚠️  [터치] 터치스크린 장치를 못 찾음 — 'xinput list' 결과에서 이름을 찾아 TOUCH_DEVICE에 넣어 주세요"
    return
  fi
  for id in $ids; do
    if xinput map-to-output "$id" "$MAIN" 2>/dev/null && [ "$1" != "quiet" ]; then
      echo "[터치] $(xinput list --name-only "$id" 2>/dev/null) (id=$id) → $MAIN 에만 연결"
    fi
  done
}
TOUCH_LOOP=""
if [ -n "$OUT" ] && [ "$OUT" != "$MAIN" ]; then
  map_touch
  # 터치 케이블을 나중에 꽂거나 다시 꽂아도 적용되게 5초마다 다시 연결
  ( while true; do sleep 5; map_touch quiet; done ) &
  TOUCH_LOOP=$!
fi

BROWSER=$(command -v chromium-browser || command -v chromium || command -v google-chrome)
if [ -z "$BROWSER" ]; then
  echo "❌ [웹 UI] Chromium이 없습니다: sudo apt install chromium-browser"
  echo "   (다른 브라우저로 $URL 에 직접 접속해도 됩니다)"
  exit 1
fi
GPU_FLAGS=""
[ "$UI_GPU" == "0" ] && GPU_FLAGS="--disable-gpu --disable-gpu-compositing"

launch() {   # launch <모니터> <주소> <창 이름>
  local G X Y
  G=$(geom "$1")
  X=$(echo "$G" | cut -d+ -f2); Y=$(echo "$G" | cut -d+ -f3)
  echo "[웹 UI] $3 → 모니터: ${1:-기본}  위치: ${X:-0},${Y:-0}  (${G:-정보 없음})"
  nice -n "$UI_NICE" "$BROWSER" --kiosk --app="$2" $GPU_FLAGS \
    --window-position="${X:-0},${Y:-0}" \
    --user-data-dir="$PROFILE-$3" \
    --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
    --disable-translate --no-first-run --check-for-update-interval=31536000 \
    --disable-smooth-scrolling --disable-background-networking --disable-extensions \
    --disable-pinch --overscroll-history-navigation=0 \
    --no-proxy-server \
    >/dev/null 2>&1 &
}

PIDS=()
if [ "$UI_LAYOUT" == "dual" ] && [ -n "$OUT" ] && [ "$OUT" != "$MAIN" ]; then
  launch "$OUT" "$URL/" main;  PIDS+=($!)
  sleep 3                       # 두 창을 동시에 띄우면 젯슨이 버벅여서 차례로
fi
if [ "$UI_LAYOUT" != "dual" ] || [ "$UI_TOUCH" == "1" ] || [ -z "$OUT" ]; then
  launch "$MAIN" "$URL/?touch=1" touch;  PIDS+=($!)
fi

# 모든 UI 창이 닫힐 때까지 기다린 뒤 모니터 설정 원래대로
for p in "${PIDS[@]}"; do wait "$p"; done
[ -n "$TOUCH_LOOP" ] && kill "$TOUCH_LOOP" 2>/dev/null

if [ "$MIRRORED" == "1" ]; then
  echo "[웹 UI] 창 닫힘 → $OUT 를 다시 $MAIN 과 같은 화면으로"
  xrandr --output "$OUT" --auto --same-as "$MAIN"
elif [ "$MIRRORED" == "2" ]; then
  echo "[웹 UI] 창 닫힘 → $OUT 해상도 원래대로"
  xrandr --output "$OUT" --auto --right-of "$MAIN"
fi
