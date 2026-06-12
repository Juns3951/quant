#!/usr/bin/env bash
# Mac/Linux 실행 스크립트 - 더블클릭하거나 터미널에서 ./run.command 로 실행하세요.
set -e

# 스크립트가 있는 폴더로 이동
cd "$(dirname "$0")"

echo "================================================"
echo "  Long-Term Quant 앱 준비 중..."
echo "================================================"

# python 명령 찾기 (python3 우선)
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[오류] Python이 설치되어 있지 않습니다."
  echo "https://www.python.org/downloads/ 에서 Python 3.11+ 를 설치한 뒤 다시 실행하세요."
  read -p "엔터를 누르면 종료합니다..."
  exit 1
fi

# 가상환경 생성 (최초 1회)
if [ ! -d ".venv" ]; then
  echo "최초 실행: 가상환경을 만드는 중입니다 (1~2분 소요)..."
  "$PY" -m venv .venv
fi

# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치 (이미 설치돼 있으면 빠르게 건너뜀)
echo "필요한 패키지를 확인/설치하는 중입니다..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements-webapp.txt

# 앱 실행
echo ""
echo "앱을 시작합니다. 브라우저가 자동으로 열립니다."
echo "(열리지 않으면 http://127.0.0.1:8000 으로 접속하세요)"
echo ""
python webapp.py
