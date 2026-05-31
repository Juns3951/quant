# Telegram Long-Term Quant Analyzer

텔레그램에 티커를 보내면 장기 투자 관점의 퀀트 분석을 GitHub Actions 또는 로컬 봇으로 돌려 답장하는 앱입니다. 별도 Streamlit 대시보드 `longterm_app.py`로 1986년 이후 장기 백테스트도 확인할 수 있습니다.

## 핵심 전략

- 장기 추세 필터: 50일 EMA와 200일 SMA
- 매수/보유: EMA50이 SMA200 위에 있는 장기 강세장
- 매도/현금화: EMA50이 SMA200 아래로 내려가는 데드크로스
- 노이즈 차단 손절: ATR14의 3.0x~4.0x 장기 추적 손절
- 성과 분석: CAGR, MDD, 단순 보유 대비 누적 성과
- 장기 켈리 비중: `f* = (mu - r) / sigma^2`

## 텔레그램 봇 실행

1. Telegram에서 `@BotFather`에게 `/newbot`을 보내 봇을 만들고 토큰을 받습니다.
2. Python 3.10 이상을 준비합니다.
3. 이 폴더에서 패키지를 설치합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. `.env.example`을 `.env`로 복사한 뒤 토큰을 입력합니다.

```powershell
Copy-Item .env.example .env
notepad .env
```

5. 실행합니다.

```powershell
python bot.py
```

텔레그램 예시:

```text
AAPL
TSLA
/analyze NVDA
005930.KS
/chatid
```

한국 종목은 yfinance 표기처럼 `.KS` 또는 `.KQ`를 붙여 보내세요.

## GitHub Actions로 실행

로컬 PC를 켜두지 않고 GitHub Actions가 텔레그램 메시지를 확인하고 답장하게 만들 수 있습니다. 단, Actions는 상시 서버가 아니므로 보통 5분 안팎의 폴링 방식입니다.

정상 파일 구조:

```text
repository-root/
  .github/
    workflows/
      telegram-stock-analyzer.yml
  bot.py
  github_action_poll.py
  longterm_app.py
  request_parser.py
  stock_analyzer.py
  requirements.txt
```

GitHub 저장소에서 `Settings` -> `Secrets and variables` -> `Actions`로 이동한 뒤 `Secrets`에 추가합니다.

```text
TELEGRAM_BOT_TOKEN=BotFather에서 받은 봇 토큰
ALLOWED_CHAT_IDS=내 텔레그램 chat id
```

선택값은 `Variables`에 넣을 수 있습니다.

```text
DEFAULT_PERIOD=max
BENCHMARK_TICKER=SPY
```

`Actions` 탭에서 `Telegram Stock Analyzer` 워크플로를 `Run workflow`로 한 번 실행하세요. 이후 텔레그램에 티커를 보내면 다음 스케줄 실행 때 답장합니다.

## Chat ID 확인

가장 확실한 방법은 봇에게 아래 명령을 보내는 것입니다.

```text
/chatid
```

GitHub Actions 방식이면 `Run workflow`를 눌러야 바로 답장을 받습니다. 답장에 나온 숫자를 그대로 `ALLOWED_CHAT_IDS`에 넣으세요. 개인 채팅과 그룹방의 chat id는 다르며, 그룹방은 보통 `-100...` 같은 음수입니다.

브라우저로 확인할 수도 있습니다.

```text
https://api.telegram.org/bot<토큰>/getUpdates
```

응답의 `message.chat.id`가 방 번호입니다. `result: []`이면 Actions나 로컬 봇이 이미 메시지를 가져간 것이므로 새 메시지를 보낸 뒤 다시 확인하세요.

## Streamlit 장기 백테스트

로컬에서 대시보드를 실행하려면:

```powershell
streamlit run longterm_app.py
```

대시보드에서 티커, 시작일, ATR 승수, 무위험 이자율, 초기 자본을 바꾸며 40년형 장기 추세 전략을 확인할 수 있습니다.

## 테스트

인터넷 없이 계산 로직만 확인하려면:

```powershell
python test_smoke.py
```

## 주의

이 앱은 장기 추세 추종 전략의 연구/검증용 도구입니다. 실제 매수/매도 주문을 자동 실행하지 않으며, 투자 조언이나 수익 보장을 의미하지 않습니다. yfinance 데이터는 지연되거나 누락될 수 있으므로 실제 주문 전 증권사/거래소 시세를 반드시 확인하세요.
