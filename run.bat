@echo off
REM Windows 실행 스크립트 - 더블클릭하면 실행됩니다.
chcp 65001 >nul
cd /d "%~dp0"

echo ================================================
echo   Long-Term Quant 앱 준비 중...
echo ================================================

REM python 명령 찾기
where python >nul 2>&1
if %errorlevel%==0 (
  set PY=python
) else (
  where py >nul 2>&1
  if %errorlevel%==0 (
    set PY=py
  ) else (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 Python 3.11+ 를 설치한 뒤 다시 실행하세요.
    echo 설치 시 "Add Python to PATH" 체크를 꼭 하세요.
    pause
    exit /b 1
  )
)

REM 가상환경 생성 (최초 1회)
if not exist ".venv" (
  echo 최초 실행: 가상환경을 만드는 중입니다 ^(1~2분 소요^)...
  %PY% -m venv .venv
)

REM 가상환경 활성화
call .venv\Scripts\activate.bat

REM 패키지 설치
echo 필요한 패키지를 확인/설치하는 중입니다...
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements-webapp.txt

REM 앱 실행
echo.
echo 앱을 시작합니다. 브라우저가 자동으로 열립니다.
echo ^(열리지 않으면 http://127.0.0.1:8000 으로 접속하세요^)
echo.
python webapp.py

pause
