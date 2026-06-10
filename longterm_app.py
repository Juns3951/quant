from __future__ import annotations

import pandas as pd
import streamlit as st

from stock_analyzer import (
    LONGTERM_START,
    analyze_price_frame,
    fetch_history,
    pct,
)


st.set_page_config(page_title="Long-Term Quant Backtester", layout="wide")

st.title("Long-Term Quant Backtester")

with st.sidebar:
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    start = st.text_input("Start date", value=LONGTERM_START)
    atr_multiplier = st.slider("ATR trailing stop multiplier", 3.0, 4.0, 3.5, 0.1)
    risk_free_rate = st.number_input("Risk-free rate", value=0.03, min_value=0.0, max_value=0.2, step=0.005)
    initial_capital = st.number_input("Initial capital", value=10_000_000, min_value=1_000, step=100_000)
    run = st.button("Run backtest", type="primary")


@st.cache_data(ttl=60 * 60)
def load_data(symbol: str, start_date: str) -> pd.DataFrame:
    return fetch_history(symbol, period="max", start=start_date)


if run or ticker:
    try:
        raw = load_data(ticker, start)
        result = analyze_price_frame(
            ticker=ticker,
            frame=raw,
            atr_multiplier=atr_multiplier,
            risk_free_rate=risk_free_rate,
            initial_capital=float(initial_capital),
        )
        df = result.frame
        if df is None:
            st.stop()

        st.subheader(f"{result.ticker} 장기 퀀트 분석")
        st.caption(f"{result.start_date} ~ {result.as_of} / {result.rows:,} trading days")

        cols = st.columns(7)
        cols[0].metric("Signal", result.action)
        cols[1].metric("Strategy CAGR", pct(result.cagr_strategy), pct(result.cagr_strategy - result.cagr_buy_hold))
        cols[2].metric("Strategy MDD", pct(result.mdd_strategy))
        cols[3].metric("Sharpe", f"{result.sharpe_ratio:.2f}" if not (result.sharpe_ratio != result.sharpe_ratio) else "N/A")
        cols[4].metric("Win Rate", pct(result.win_rate))
        cols[5].metric("# Trades", str(result.num_trades))
        cols[6].metric("Market exposure", pct(result.market_exposure))

        st.markdown("### Price, EMA50, SMA200, ATR trailing stop")
        price_chart = df[["Close", "EMA_50", "SMA_200", "ATR_Trailing_Stop"]].rename(
            columns={
                "Close": "Close",
                "EMA_50": "EMA50",
                "SMA_200": "SMA200",
                "ATR_Trailing_Stop": "ATR Stop",
            }
        )
        st.line_chart(price_chart)

        st.markdown("### Equity curve")
        equity_chart = df[["Strategy_Equity", "Buy_Hold_Equity"]].rename(
            columns={
                "Strategy_Equity": "EMA50/SMA200 + ATR Stop",
                "Buy_Hold_Equity": "Buy & Hold",
            }
        )
        st.line_chart(equity_chart)

        st.markdown("### Drawdown")
        dd_chart = df[["Strategy_Drawdown", "Buy_Hold_Drawdown"]].rename(
            columns={
                "Strategy_Drawdown": "Strategy DD",
                "Buy_Hold_Drawdown": "Buy & Hold DD",
            }
        )
        st.line_chart(dd_chart)

        st.markdown("### Current levels")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Current": result.current_price,
                        "EMA50": result.ema50,
                        "SMA200": result.sma200,
                        "ATR14": result.atr14,
                        "ATR Stop": result.trailing_stop,
                        "Invested": result.invested_now,
                    }
                ]
            ),
            use_container_width=True,
        )

        st.markdown("### Cross events")
        events = df.loc[df["Golden_Cross"] | df["Death_Cross"], ["Close", "EMA_50", "SMA_200", "Golden_Cross", "Death_Cross"]]
        st.dataframe(events.tail(30), use_container_width=True)

        st.markdown("### Trade log")
        if result.trades is not None and not result.trades.empty:
            display_trades = result.trades.copy()
            display_trades["Return %"] = (display_trades["Return"] * 100).round(2)
            display_trades = display_trades.drop(columns=["Return"])

            def _color_return(val: float) -> str:
                return "color: green" if val > 0 else "color: red"

            st.dataframe(
                display_trades.style.map(_color_return, subset=["Return %"]),
                use_container_width=True,
            )

            tc1, tc2, tc3, tc4 = st.columns(4)
            pf = result.profit_factor
            pf_str = "∞" if pf == float("inf") else f"{pf:.2f}" if pf == pf else "N/A"
            tc1.metric("Profit Factor", pf_str)
            tc2.metric("Avg hold", f"{result.avg_holding_days:.0f}일" if result.avg_holding_days == result.avg_holding_days else "N/A")
            tc3.metric("Best trade", pct(result.best_trade))
            tc4.metric("Worst trade", pct(result.worst_trade))
        else:
            st.info("해당 기간에 발생한 트레이드가 없습니다.")

        with st.expander("Formula guide"):
            st.markdown(
                r"""
                **CAGR**

                $$CAGR = \left(\text{Cumulative Return}\right)^{\frac{1}{Y}} - 1$$

                **MDD**

                $$MDD = \min\left(\frac{P_t}{\max(P_{1..t})} - 1\right)$$

                **Long-term Kelly allocation**

                $$f^* = \frac{\mu - r}{\sigma^2}$$
                """
            )

        st.info("이 대시보드는 연구/검증용입니다. 실제 투자 판단 전 데이터와 리스크를 별도로 확인하세요.")
    except Exception as exc:
        st.error(str(exc))
