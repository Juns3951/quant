# Telegram Stock Analyzer Bot

텔레그램에 티커를 보내면 첨부 텍스트의 분석 프레임워크를 바탕으로 주가를 분석해 답변하는 파이썬 봇입니다.

구현된 분석 기준:

- 시장 국면 분류: 강한 상승/하락 추세, 박스권, 변동성 급증
- 핵심 지표: EMA10/20/50/200, ADX, RSI, MACD, Bollinger Bands, ATR, VWAP20, 거래량 평균
- 3단계 필터: 추세 확인 -> 모멘텀 확인 -> 거래량/유동성 확인
- 가짜 돌파 필터: 2거래일 종가 안착, RSI 다이버전스, ATR 확장, VWAP 이격
- 매도/리스크: ATR 기반 초기 손절, 추적 손절, 과매수/추세 둔화 경고
- 킬스위치: 벤치마크 200일 EMA 이탈, VIX 40 초과, 5일 변동성 급등

## 준비

1. Telegram에서 `@BotFather`에게 `/newbot`을 보내 봇을 만들고 토큰을 받습니다.
2. Python 3.10 이상을 설치합니다.
3. 이 폴더로 이동합니다.

```powershell
cd C:\Users\juns3\Documents\Codex\2026-05-31\files-mentioned-by-the-user-txt\outputs\telegram-stock-analyzer
```

4. 가상환경을 만들고 패키지를 설치합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. `.env.example`을 `.env`로 복사한 뒤 `TELEGRAM_BOT_TOKEN`을 입력합니다.

```powershell
Copy-Item .env.example .env
notepad .env
```

## 실행

```powershell
python bot.py
```

텔레그램에서 봇에게 아래처럼 보내면 됩니다.

```text
AAPL
TSLA 2y
/analyze NVDA 1y
005930.KS
035420.KQ
```

## 선택 설정

`.env`에서 아래 값을 바꿀 수 있습니다.

```dotenv
DEFAULT_PERIOD=1y
BENCHMARK_TICKER=SPY
ALLOWED_CHAT_IDS=
```

`ALLOWED_CHAT_IDS`에 숫자 chat id를 쉼표로 넣으면 해당 채팅만 봇을 사용할 수 있습니다. 비워두면 봇에 접근 가능한 모든 채팅에서 사용할 수 있습니다.

## GitHub Actions로 실행

로컬 PC를 켜두지 않고 GitHub Actions가 텔레그램 메시지를 확인하고 답장하게 만들 수 있습니다. 단, GitHub Actions는 상시 실행 서버가 아니므로 실시간 응답이 아니라 5분 안팎의 폴링 방식입니다.

1. 이 폴더의 파일들을 GitHub 저장소 루트에 올립니다.
2. GitHub 저장소에서 `Settings` -> `Secrets and variables` -> `Actions`로 이동합니다.
3. `Secrets`에 아래 값을 추가합니다.

```text
TELEGRAM_BOT_TOKEN=BotFather에서 받은 봇 토큰
ALLOWED_CHAT_IDS=내 텔레그램 chat id
```

4. `Variables`에는 선택값을 추가할 수 있습니다.

```text
DEFAULT_PERIOD=1y
BENCHMARK_TICKER=SPY
```

5. `Actions` 탭에서 `Telegram Stock Analyzer` 워크플로를 활성화하고 `Run workflow`로 한 번 수동 실행합니다.
6. 이후 텔레그램에서 봇에게 `AAPL`, `TSLA 2y`, `005930.KS`처럼 보내면 Actions가 정기 실행될 때 답장합니다.

주의: 로컬에서 `python bot.py`를 동시에 실행하지 마세요. Telegram `getUpdates` 폴링은 한쪽이 메시지를 먼저 가져가면 다른 쪽이 같은 메시지를 못 받을 수 있습니다.

### Telegram 토큰과 방 번호 확인

- `TELEGRAM_BOT_TOKEN`: Telegram의 `@BotFather`에게 `/newbot`으로 봇을 만든 뒤 받은 토큰입니다. GitHub에는 코드나 `.env`가 아니라 반드시 `Repository secrets`에 넣습니다.
- `ALLOWED_CHAT_IDS`: 봇에게 아무 메시지나 보낸 뒤 브라우저에서 `https://api.telegram.org/bot<토큰>/getUpdates`를 열면 `chat.id` 숫자가 나옵니다. 개인 채팅은 보통 양수, 그룹/채널은 보통 음수입니다.

그룹방에서 쓸 때는 봇을 그룹에 초대하고 `/analyze AAPL`처럼 명령어로 호출하는 편이 안전합니다. 일반 텍스트 `AAPL`까지 그룹에서 읽게 하려면 `@BotFather`의 `/setprivacy`에서 해당 봇의 privacy mode를 끄세요.

`이 봇을 사용할 권한이 없는 채팅입니다`가 뜨면 토큰 문제가 아니라 `ALLOWED_CHAT_IDS` 값이 현재 채팅방의 `chat.id`와 다르다는 뜻입니다. 답장에 표시되는 `현재 chat id`를 그대로 GitHub Actions secret의 `ALLOWED_CHAT_IDS`에 넣고 다시 실행하세요.

가장 확실한 확인 방법은 봇에게 `/chatid` 또는 `/id`를 보내는 것입니다. 이 명령은 `ALLOWED_CHAT_IDS`가 틀려도 현재 채팅방의 id를 답장하도록 되어 있습니다.

## 테스트

인터넷 없이 분석 로직만 간단히 확인하려면 다음을 실행하세요.

```powershell
python test_smoke.py
```

## 주의

이 봇은 기술적 지표 기반의 참고용 분석 도구입니다. 실제 매수/매도 주문을 자동 실행하지 않으며, 투자 조언이나 수익 보장을 의미하지 않습니다. yfinance 데이터는 지연되거나 누락될 수 있으므로 실제 주문 전 증권사/거래소 시세를 반드시 확인하세요.
