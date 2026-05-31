from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from request_parser import TICKER_RE, parse_request
from stock_analyzer import AnalyzerError, analyze_ticker, format_telegram_report


TELEGRAM_LIMIT = 3900


def allowed_chat_ids() -> set[int]:
    raw = os.getenv("ALLOWED_CHAT_IDS", "").strip()
    if not raw:
        return set()
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def main() -> None:
    ensure_utf8_stdout()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN secret이 없습니다.")

    updates = telegram_call(token, "getUpdates", {"timeout": 0, "allowed_updates": json.dumps(["message"])})
    if not updates.get("ok"):
        raise SystemExit(f"Telegram getUpdates 실패: {updates}")

    items = updates.get("result", [])
    if not items:
        print("No Telegram updates.")
        return

    max_update_id = None
    for item in items:
        max_update_id = item.get("update_id", max_update_id)
        handle_update(token, item)

    if max_update_id is not None:
        telegram_call(token, "getUpdates", {"offset": int(max_update_id) + 1, "timeout": 0})
        print(f"Processed updates through {max_update_id}.")


def handle_update(token: str, update: dict[str, Any]) -> None:
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()
    if chat_id is None or not text:
        return

    command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
    if command in {"/chatid", "/id"}:
        send_message(token, chat_id, f"현재 chat id: {chat_id}")
        return

    allowed = allowed_chat_ids()
    if allowed and int(chat_id) not in allowed:
        send_message(
            token,
            chat_id,
            "이 봇을 사용할 권한이 없는 채팅입니다.\n"
            f"현재 chat id: {chat_id}\n"
            "이 번호를 ALLOWED_CHAT_IDS에 추가하세요.",
        )
        return

    if command in {"/start", "/help"}:
        send_message(token, chat_id, help_message())
        return

    ticker, period = parse_request(text)
    if not ticker:
        send_message(token, chat_id, "분석할 티커를 입력해주세요. 예: AAPL 또는 005930.KS")
        return
    if not TICKER_RE.match(ticker):
        send_message(token, chat_id, "티커 형식이 올바르지 않습니다. 예: AAPL, TSLA, 005930.KS")
        return

    send_message(token, chat_id, f"{ticker.upper()} 데이터를 GitHub Actions에서 분석 중입니다...")
    benchmark = os.getenv("BENCHMARK_TICKER", "SPY").strip() or "SPY"
    try:
        result = analyze_ticker(ticker, period, benchmark)
        send_long_message(token, chat_id, format_telegram_report(result))
    except AnalyzerError as exc:
        send_message(token, chat_id, str(exc))
    except Exception as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        send_message(token, chat_id, "분석 중 오류가 발생했습니다. 티커와 네트워크 상태를 확인해주세요.")


def help_message() -> str:
    return (
        "티커를 보내면 GitHub Actions에서 장기 투자용 EMA50/SMA200 추세 필터, "
        "ATR 추적 손절, 장기 백테스트, 켈리 비중을 분석해 답장합니다.\n\n"
        "예시:\n"
        "AAPL\n"
        "TSLA\n"
        "/analyze NVDA\n"
        "005930.KS\n"
        "035420.KQ\n"
        "/chatid\n\n"
        "주의: GitHub Actions 스케줄 방식은 보통 몇 분 단위로 응답합니다."
    )


def send_long_message(token: str, chat_id: int | str, text: str) -> None:
    chunks = [text[i : i + TELEGRAM_LIMIT] for i in range(0, len(text), TELEGRAM_LIMIT)]
    for chunk in chunks:
        send_message(token, chat_id, chunk)
        time.sleep(0.2)


def send_message(token: str, chat_id: int | str, text: str) -> None:
    telegram_call(token, "sendMessage", {"chat_id": chat_id, "text": text[:4096]})


def telegram_call(token: str, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{token}/{method}"
    encoded = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(url, data=encoded, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram API HTTP {exc.code}: {body}") from exc


def ensure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
