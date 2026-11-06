#!/bin/bash
# 전시용 웹 UI를 두 번째 모니터에 전체 화면(키오스크)으로 띄움.
#   1) 먼저 졸음 감지 프로그램 실행:  python3 drowsiness_jetson_trt.py
#   2) 다른 터미널에서:               ./start_exhibit_ui.sh            (두 번째 모니터 자동 선택)
#                                    ./start_exhibit_ui.sh DP-0       (모니터 이름 직접 지정, xrandr 로 확인)
# 끄기: Alt+F4 (또는 pkill -f exhibit-ui)

URL="http://localhost:5000"

# 프로그램의 웹 서버가 뜰 때까지 대기 (엔진 로딩에 시간이 걸릴 수 있음)
echo "웹 UI 서버 기다리는 중... ($URL)"
for i in $(seq 1 120); do
  python3 -c "import urllib.request; urllib.request.urlopen('$URL/api/state', timeout=1)" 2>/dev/null && break
  sleep 1
done

# 대상 모니터: 인자로 받거나, primary가 아닌 연결된 모니터
OUT="$1"
if [ -z "$OUT" ]; then
  OUT=$(xrandr --query | awk '/ connected/ && !/ primary/ {print $1}' | tail -n 1)
fi
GEOM=$(xrandr --query | awk -v o="$OUT" '$1==o {for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+x[0-9]+\+[0-9]+\+[0-9]+$/) print $i}')
X=$(echo "$GEOM" | cut -d+ -f2)
Y=$(echo "$GEOM" | cut -d+ -f3)
echo "모니터: ${OUT:-기본}  위치: ${X:-0},${Y:-0}  (${GEOM:-정보 없음})"

BROWSER=$(command -v chromium-browser || command -v chromium || command -v google-chrome)
if [ -z "$BROWSER" ]; then
  echo "❌ Chromium이 없습니다: sudo apt install chromium-browser"
  exit 1
fi

"$BROWSER" --kiosk --app="$URL" \
  --window-position="${X:-0},${Y:-0}" \
  --user-data-dir=/tmp/exhibit-ui \
  --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
  --disable-translate --no-first-run --check-for-update-interval=31536000 \
  >/dev/null 2>&1 &
echo "✅ 웹 UI 실행 (끄기: Alt+F4)"
