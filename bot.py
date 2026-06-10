from __future__ import annotations

import asyncio
import io
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from chart_generator import generate_backtest_chart
from request_parser import TICKER_RE, parse_request
from stock_analyzer import AnalyzerError, analyze_ticker, format_telegram_report


def allowed_chat_ids() -> set[int]:
    raw = os.getenv("ALLOWED_CHAT_IDS", "").strip()
    if not raw:
        return set()
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "티커를 보내면 장기 투자용 EMA50/SMA200 추세 필터, ATR 3.5배 추적 손절, "
        "40년형 백테스트, 장기 켈리 비중을 분석합니다.\n\n"
        "예시:\n"
        "AAPL\n"
        "TSLA\n"
        "005930.KS\n"
        "/analyze NVDA\n"
        "/chatid\n\n"
        "한국 종목은 yfinance 표기처럼 .KS 또는 .KQ를 붙여 보내세요."
    )


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await run_analysis(update, " ".join(context.args))


async def chatid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_chat is None:
        return
    await update.message.reply_text(f"현재 chat id: {update.effective_chat.id}")


async def analyze_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await run_analysis(update, update.message.text or "")


async def run_analysis(update: Update, raw_text: str) -> None:
    if update.message is None:
        return

    allowed = allowed_chat_ids()
    chat_id = update.effective_chat.id if update.effective_chat else None
    if allowed and chat_id not in allowed:
        await update.message.reply_text(
            "이 봇을 사용할 권한이 없는 채팅입니다.\n"
            f"현재 chat id: {chat_id}\n"
            "이 번호를 ALLOWED_CHAT_IDS에 추가하세요."
        )
        return

    ticker, period = parse_request(raw_text)
    if not ticker:
        await update.message.reply_text("분석할 티커를 입력해주세요. 예: AAPL 또는 005930.KS")
        return

    if not TICKER_RE.match(ticker):
        await update.message.reply_text("티커 형식이 올바르지 않습니다. 예: AAPL, TSLA, 005930.KS")
        return

    waiting_message = await update.message.reply_text(f"{ticker.upper()} 데이터를 가져와 분석 중입니다...")
    benchmark = os.getenv("BENCHMARK_TICKER", "SPY").strip() or "SPY"

    try:
        result = await asyncio.to_thread(analyze_ticker, ticker, period, benchmark)
        report = format_telegram_report(result)
        await waiting_message.edit_text(report[:3900])

        chart_bytes = await asyncio.to_thread(generate_backtest_chart, result)
        if chart_bytes:
            await update.message.reply_photo(
                photo=io.BytesIO(chart_bytes),
                caption=f"{result.ticker} 백테스트 차트",
            )
    except AnalyzerError as exc:
        await waiting_message.edit_text(str(exc))
    except Exception:
        logging.exception("analysis failed")
        await waiting_message.edit_text("분석 중 오류가 발생했습니다. 티커와 네트워크 상태를 확인해주세요.")


def build_application() -> Application:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN이 없습니다. .env 파일에 봇 토큰을 설정하세요.")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler(["chatid", "id"], chatid_command))
    app.add_handler(CommandHandler(["analyze", "a"], analyze_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text))
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = build_application()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
