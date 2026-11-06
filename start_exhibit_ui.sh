#!/bin/bash
# 전시용 웹 UI를 두 번째 모니터에 전체 화면(키오스크)으로 띄움.
#   보통은 python3 drowsiness_jetson_trt.py 가 자동으로 이 스크립트를 실행함 (WEB_UI_AUTO_OPEN=True)
#   직접 실행할 때: 프로그램을 켠 상태에서 '다른 터미널'에서
#       bash start_exhibit_ui.sh            (두 번째 모니터 자동 선택)
#       bash start_exhibit_ui.sh DP-0       (모니터 이름 직접 지정 — xrandr 로 확인)
#   두 모니터가 같은 화면(미러링)이면 잠깐 옆으로 펼쳐서 띄우고, UI 창이 닫히면 다시 같은 화면으로 돌려놓음.
# 끄기: 프로그램 종료(q) 하면 같이 닫힘, 또는 UI 화면에서 Alt+F4

URL="http://localhost:5000"
PROFILE=/tmp/exhibit-ui

# 이미 떠 있으면 하나 더 띄우지 않음
if pgrep -f "user-data-dir=$PROFILE" >/dev/null; then
  echo "[웹 UI] 이미 실행 중"
  exit 0
fi

# 프로그램의 웹 서버가 뜰 때까지 대기
for i in $(seq 1 120); do
  python3 -c "import urllib.request; urllib.request.urlopen('$URL/api/state', timeout=1)" 2>/dev/null && break
  sleep 1
done

# 연결된 모니터 목록 (첫 번째 = primary가 있으면 primary)
mapfile -t OUTS < <(xrandr --query | awk '/ connected primary/ {print $1}'; xrandr --query | awk '/ connected/ && !/ connected primary/ {print $1}')
if [ "${#OUTS[@]}" -lt 2 ] && [ -z "$1" ]; then
  echo "⚠️  [웹 UI] 연결된 모니터가 ${#OUTS[@]}개뿐 — 두 번째 모니터 케이블/전원을 확인하세요 (지금 화면에 띄움)"
fi
MAIN="${OUTS[0]}"
OUT="${1:-${OUTS[1]}}"

geom() {   # 모니터 위치 "가로x세로+X+Y"
  xrandr --query | awk -v o="$1" '$1==o {for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+x[0-9]+\+[0-9]+\+[0-9]+$/) {print $i; exit}}'
}

MIRRORED=0
if [ -n "$OUT" ] && [ "$OUT" != "$MAIN" ]; then
  G_OUT=$(geom "$OUT"); G_MAIN=$(geom "$MAIN")
  # 같은 위치(+X+Y)면 미러링 상태 → 옆으로 펼침
  if [ -z "$G_OUT" ] || [ "${G_OUT#*+}" == "${G_MAIN#*+}" ]; then
    echo "[웹 UI] $OUT 가 $MAIN 과 같은 화면이라 오른쪽으로 펼침"
    xrandr --output "$OUT" --auto --right-of "$MAIN"
    MIRRORED=1
    sleep 2
  fi
fi

G=$(geom "$OUT")
X=$(echo "$G" | cut -d+ -f2); Y=$(echo "$G" | cut -d+ -f3)
echo "[웹 UI] 모니터: ${OUT:-기본}  위치: ${X:-0},${Y:-0}  (${G:-정보 없음})"

BROWSER=$(command -v chromium-browser || command -v chromium || command -v google-chrome)
if [ -z "$BROWSER" ]; then
  echo "❌ [웹 UI] Chromium이 없습니다: sudo apt install chromium-browser"
  echo "   (다른 브라우저로 $URL 에 직접 접속해도 됩니다)"
  exit 1
fi

"$BROWSER" --kiosk --app="$URL" \
  --window-position="${X:-0},${Y:-0}" \
  --user-data-dir="$PROFILE" \
  --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
  --disable-translate --no-first-run --check-for-update-interval=31536000 \
  >/dev/null 2>&1
# ↑ UI 창이 닫힐 때까지 여기서 기다림

if [ "$MIRRORED" == "1" ]; then
  echo "[웹 UI] 창 닫힘 → $OUT 를 다시 $MAIN 과 같은 화면으로"
  xrandr --output "$OUT" --same-as "$MAIN"
fi
