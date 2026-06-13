"""
stock_analyzer.py — Long-term trend-following analysis engine.

Public API (backward-compatible):
    analyze_ticker()        — fetch + analyse by ticker symbol
    analyze_price_frame()   — analyse a pre-loaded OHLCV DataFrame
    LongTermResult          — result dataclass (all fields have defaults for new ones)
    format_telegram_report()
    fmt, pct, format_date, normalize_longterm_period
    AnalyzerError

New in this version:
    • Uses backtest_engine.Backtester (OOP state machine, ATR slippage, no lookahead)
    • Uses data_provider.fetch_price_data (SQLite incremental cache, Tiingo fallback)
    • Regime-conditioned entry score weights
    • bootstrap_ci, walk_forward_analysis re-exported from backtest_engine
    • Extended LongTermResult fields: calmar_ratio, exposure_adj_cagr,
      turnover, rolling_cagr_3y/5y, rolling_mdd_3y, sharpe_ci_lower, pf_ci_lower,
      commission_bps, slippage_beta, reentry_mode, adj_close_warnings
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_engine import (
    Backtester,
    BacktestConfig,
    GoldenCrossStrategy,
    TradeRecord,
    bootstrap_ci,
    compute_metrics,
    walk_forward_analysis,
    backtest_entry_score,
    TRADING_DAYS,
    _atr as _atr_fn,
    _rsi as _rsi_fn,
    _rolling_cagr,
    _rolling_mdd,
)
from data_provider import fetch_price_data, validate_adjusted_close

LONGTERM_START = "1986-01-01"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LongTermResult:
    # ── Core ──────────────────────────────────────────────────────────────
    ticker: str
    as_of: str
    start_date: str
    rows: int
    action: str
    regime: str
    confidence: str
    current_price: float
    ema50: float
    sma200: float
    atr14: float
    atr_multiplier: float
    trailing_stop: float
    initial_stop: float
    golden_cross_today: bool
    death_cross_today: bool
    invested_now: bool
    market_exposure: float
    cagr_strategy: float
    cagr_buy_hold: float
    mdd_strategy: float
    mdd_buy_hold: float
    cumulative_strategy: float
    cumulative_buy_hold: float
    annual_return: float
    annual_volatility: float
    risk_free_rate: float
    kelly_fraction: float
    suggested_fractional_kelly: float
    warnings: list[str]
    insights: list[str]
    # ── Optional / new ────────────────────────────────────────────────────
    frame: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    sharpe_ratio: float = float("nan")
    sortino_ratio: float = float("nan")
    win_rate: float = float("nan")
    profit_factor: float = float("nan")
    num_trades: int = 0
    avg_holding_days: float = float("nan")
    best_trade: float = float("nan")
    worst_trade: float = float("nan")
    entry_score: float = float("nan")
    entry_verdict: str = ""
    entry_factors: list[dict[str, Any]] | None = None
    rsi14: float = float("nan")
    # ── Extended metrics (v2) ─────────────────────────────────────────────
    calmar_ratio: float = float("nan")
    exposure_adj_cagr: float = float("nan")
    turnover: float = float("nan")
    rolling_cagr_3y: float = float("nan")
    rolling_cagr_5y: float = float("nan")
    rolling_mdd_3y: float = float("nan")
    sharpe_ci_lower: float = float("nan")
    pf_ci_lower: float = float("nan")
    commission_bps: float = 0.0
    slippage_beta: float = 0.0
    reentry_mode: str = "next_cross"
    adj_close_warnings: list[str] | None = None


class AnalyzerError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Data fetching (delegates to data_provider)
# ---------------------------------------------------------------------------

def fetch_history(
    ticker: str,
    period: str = "max",
    start: str = LONGTERM_START,
) -> pd.DataFrame:
    """
    Fetch adjusted OHLCV.  Uses SQLite incremental cache; falls back to
    Tiingo if TIINGO_API_KEY env var is set.
    """
    try:
        return fetch_price_data(ticker=ticker, period=period, start=start)
    except RuntimeError as exc:
        raise AnalyzerError(str(exc)) from exc
    except ImportError as exc:
        raise AnalyzerError(
            "yfinance가 설치되어 있지 않습니다. `pip install -r requirements.txt`를 먼저 실행하세요."
        ) from exc


# ---------------------------------------------------------------------------
# Main analysis entry points
# ---------------------------------------------------------------------------

def analyze_ticker(
    ticker: str,
    period: str = "max",
    benchmark_ticker: str | None = None,
    commission_bps: float = 5.0,
    slippage_beta: float = 0.1,
    reentry_mode: str = "next_cross",
) -> LongTermResult:
    del benchmark_ticker
    ticker = ticker.strip().upper()
    effective_period = normalize_longterm_period(period)
    frame = fetch_history(ticker, period=effective_period)
    return analyze_price_frame(
        ticker=ticker,
        frame=frame,
        commission_bps=commission_bps,
        slippage_beta=slippage_beta,
        reentry_mode=reentry_mode,
    )


def analyze_price_frame(
    ticker: str,
    frame: pd.DataFrame,
    atr_multiplier: float = 3.5,
    risk_free_rate: float = 0.03,
    initial_capital: float = 10_000_000.0,
    commission_bps: float = 5.0,
    slippage_beta: float = 0.1,
    reentry_mode: str = "next_cross",
    run_bootstrap: bool = False,   # slow — off by default
    run_wfa: bool = False,         # very slow — off by default
) -> LongTermResult:
    """
    Core analysis pipeline:
    1. Clean & validate raw OHLCV.
    2. Run event-driven backtester (OOP, ATR slippage, 1-day execution lag).
    3. Compute performance metrics.
    4. Compute entry score with regime-conditioned weights.
    5. Optionally run bootstrap CI and walk-forward analysis.
    """
    # --- 1. Validate & clean ---
    cleaned = clean_price_frame(frame)
    adj_warnings = validate_adjusted_close(cleaned)

    cfg = BacktestConfig(
        atr_multiplier=atr_multiplier,
        commission_bps=commission_bps,
        slippage_beta=slippage_beta,
        initial_capital=initial_capital,
        reentry_mode=reentry_mode,
    )
    bt = Backtester(cfg)

    # --- 2. Prepare indicators ---
    prepared = bt.prepare(cleaned)
    df = prepared.dropna(subset=["EMA_50", "SMA_200", "ATR_14"])

    if len(df) < 220:
        raise AnalyzerError(
            "장기 분석에 필요한 데이터가 부족합니다. 최소 220거래일 이상이 필요합니다."
        )

    # --- 3. Run backtester ---
    df, trade_records = bt.run(df)

    # --- 4. Performance metrics ---
    m = compute_metrics(df, trade_records, risk_free_rate=risk_free_rate)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    action = decide_longterm_action(latest, prev)
    regime_label = classify_longterm_regime(latest)
    insights, warnings_ = explain_longterm(latest, prev, m, atr_multiplier)
    if adj_warnings:
        warnings_.extend(adj_warnings)

    # --- 5. Entry assessment (regime-conditioned) ---
    market_regime = _detect_market_regime(df)
    entry = entry_assessment(latest, m, market_regime=market_regime)

    # --- 6. Convert TradeRecord list → DataFrame ---
    if trade_records:
        trades_df = pd.DataFrame([{
            "Entry Date": t.entry_date,
            "Entry Price": t.entry_close,   # display raw close for readability
            "Exit Date": t.exit_date,
            "Exit Price": t.exit_close,
            "Return": t.net_return,         # net (cost-adjusted)
            "Gross Return": t.gross_return,
            "Holding Days": t.holding_days,
            "Exit Reason": t.exit_reason,
        } for t in trade_records])
    else:
        trades_df = pd.DataFrame(columns=[
            "Entry Date", "Entry Price", "Exit Date", "Exit Price",
            "Return", "Gross Return", "Holding Days", "Exit Reason",
        ])

    # --- 7. Bootstrap CI (optional) ---
    boot: dict[str, float] = {"sharpe_ci_lower": float("nan"), "pf_ci_lower": float("nan")}
    if run_bootstrap:
        boot = bootstrap_ci(trade_records)

    # --- 8. Rolling metrics ---
    eq = df["Strategy_Equity"]
    rc3y = float(_rolling_cagr(eq, 3.0).iloc[-1]) if len(df) >= 3 * TRADING_DAYS else float("nan")
    rc5y = float(_rolling_cagr(eq, 5.0).iloc[-1]) if len(df) >= 5 * TRADING_DAYS else float("nan")
    rm3y = float(_rolling_mdd(df["Strategy_Drawdown"], 3.0).iloc[-1]) if len(df) >= 3 * TRADING_DAYS else float("nan")

    return LongTermResult(
        ticker=ticker.strip().upper(),
        as_of=format_date(df.index[-1]),
        start_date=format_date(df.index[0]),
        rows=len(df),
        action=action,
        regime=regime_label,
        confidence=confidence_label(latest, m),
        current_price=float(latest["Close"]),
        ema50=float(latest["EMA_50"]),
        sma200=float(latest["SMA_200"]),
        atr14=float(latest["ATR_14"]),
        atr_multiplier=float(atr_multiplier),
        trailing_stop=float(latest["ATR_Trailing_Stop"]),
        initial_stop=float(latest["Close"] - atr_multiplier * latest["ATR_14"]),
        golden_cross_today=bool(latest["Golden_Cross"]),
        death_cross_today=bool(latest["Death_Cross"]),
        invested_now=bool(latest["Invested"]),
        market_exposure=float(m["market_exposure"]),
        cagr_strategy=float(m["cagr_strategy"]),
        cagr_buy_hold=float(m["cagr_buy_hold"]),
        mdd_strategy=float(m["mdd_strategy"]),
        mdd_buy_hold=float(m["mdd_buy_hold"]),
        cumulative_strategy=float(m["cumulative_strategy"]),
        cumulative_buy_hold=float(m["cumulative_buy_hold"]),
        annual_return=float(m["annual_return"]),
        annual_volatility=float(m["annual_volatility"]),
        risk_free_rate=float(risk_free_rate),
        kelly_fraction=float(m["kelly_fraction"]),
        suggested_fractional_kelly=float(m["suggested_fractional_kelly"]),
        warnings=warnings_,
        insights=insights,
        frame=df,
        trades=trades_df,
        sharpe_ratio=float(m["sharpe_ratio"]),
        sortino_ratio=float(m["sortino_ratio"]),
        win_rate=float(m["win_rate"]),
        profit_factor=float(m["profit_factor"]),
        num_trades=int(m["num_trades"]),
        avg_holding_days=float(m["avg_holding_days"]),
        best_trade=float(m["best_trade"]),
        worst_trade=float(m["worst_trade"]),
        entry_score=float(entry["score"]),
        entry_verdict=entry["verdict"],
        entry_factors=entry["factors"],
        rsi14=float(entry["rsi14"]),
        calmar_ratio=float(m["calmar_ratio"]),
        exposure_adj_cagr=float(m["exposure_adj_cagr"]),
        turnover=float(m["turnover"]),
        rolling_cagr_3y=rc3y,
        rolling_cagr_5y=rc5y,
        rolling_mdd_3y=rm3y,
        sharpe_ci_lower=float(boot["sharpe_ci_lower"]),
        pf_ci_lower=float(boot["pf_ci_lower"]),
        commission_bps=commission_bps,
        slippage_beta=slippage_beta,
        reentry_mode=reentry_mode,
        adj_close_warnings=adj_warnings if adj_warnings else None,
    )


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise AnalyzerError("가격 데이터가 비어 있습니다.")

    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename_map = {str(col).lower(): col for col in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in rename_map]
    if missing:
        raise AnalyzerError(f"필수 가격 컬럼이 없습니다: {', '.join(missing)}")

    df = df[[rename_map[name] for name in required]].copy()
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.apply(pd.to_numeric, errors="coerce")
    df["Volume"] = df["Volume"].fillna(0.0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Legacy vectorized backtester (kept for external callers / unit tests)
# ---------------------------------------------------------------------------

def calculate_longterm_backtest(
    df: pd.DataFrame,
    atr_multiplier: float = 3.5,
    initial_capital: float = 10_000_000.0,
) -> pd.DataFrame:
    """
    Vectorized backtester (legacy).  Kept for backward compatibility.
    analyze_price_frame() now uses the event-driven Backtester instead.

    NOTE: Invested = Invested_Raw.shift(1) — signal fires at EOD day T,
    position held from day T+1 onward (same-day close execution; strictly
    the event backtester uses next-day close which is more conservative).
    """
    out = df.copy()
    close = out["Close"]
    out["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    out["SMA_200"] = close.rolling(200).mean()
    out["ATR_14"] = _atr_fn(out["High"], out["Low"], close)
    out["RSI_14"] = _rsi_fn(close)

    out["Bull_Regime"] = out["EMA_50"] > out["SMA_200"]
    prev_bull = out["Bull_Regime"].shift(1, fill_value=False).astype(bool)
    out["Golden_Cross"] = out["Bull_Regime"] & ~prev_bull
    out["Death_Cross"] = ~out["Bull_Regime"] & prev_bull

    regime_group = out["Bull_Regime"].ne(out["Bull_Regime"].shift()).cumsum()
    regime_high = close.where(out["Bull_Regime"]).groupby(regime_group).cummax()
    out["ATR_Trailing_Stop"] = regime_high - (atr_multiplier * out["ATR_14"])

    stop_ok = (close >= out["ATR_Trailing_Stop"]) | out["ATR_Trailing_Stop"].isna()
    out["Invested_Raw"] = out["Bull_Regime"] & stop_ok
    out["Invested"] = out["Invested_Raw"].shift(1).fillna(False).astype(bool)

    out["Asset_Return"] = close.pct_change().fillna(0.0)
    out["Strategy_Return"] = out["Asset_Return"] * out["Invested"].astype(float)
    out["Buy_Hold_Equity"] = initial_capital * (1.0 + out["Asset_Return"]).cumprod()
    out["Strategy_Equity"] = initial_capital * (1.0 + out["Strategy_Return"]).cumprod()
    out["Strategy_Drawdown"] = out["Strategy_Equity"] / out["Strategy_Equity"].cummax() - 1.0
    out["Buy_Hold_Drawdown"] = out["Buy_Hold_Equity"] / out["Buy_Hold_Equity"].cummax() - 1.0
    return out


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract trade log from vectorized Invested column (legacy helper).
    The event backtester generates trades directly; use result.trades instead.
    """
    invested = df["Invested"].astype(bool)
    prev_invested = invested.shift(1, fill_value=False)

    entry_mask = invested & ~prev_invested
    exit_mask = ~invested & prev_invested

    entry_indices = df.index[entry_mask].tolist()
    exit_indices = df.index[exit_mask].tolist()

    open_at_end = bool(invested.iloc[-1])
    if open_at_end:
        exit_indices.append(df.index[-1])

    records = []
    for i, (entry_idx, exit_idx) in enumerate(zip(entry_indices, exit_indices)):
        entry_price = float(df.loc[entry_idx, "Close"])
        is_end_of_data = open_at_end and i == len(exit_indices) - 1

        if is_end_of_data:
            last_invested_idx = exit_idx
            exit_reason = "End of Data"
        else:
            last_invested_pos = df.index.get_loc(exit_idx) - 1
            last_invested_idx = df.index[last_invested_pos]
            signal_row = df.loc[last_invested_idx]
            atr_stop = signal_row["ATR_Trailing_Stop"]
            if signal_row["Death_Cross"]:
                exit_reason = "Death Cross"
            elif pd.notna(atr_stop) and signal_row["Close"] < atr_stop:
                exit_reason = "ATR Stop"
            else:
                exit_reason = "Death Cross"

        exit_price = float(df.loc[last_invested_idx, "Close"])
        holding_days = (last_invested_idx - entry_idx).days
        trade_return = exit_price / entry_price - 1.0

        records.append({
            "Entry Date": entry_idx.date(),
            "Entry Price": round(entry_price, 4),
            "Exit Date": last_invested_idx.date(),
            "Exit Price": round(exit_price, 4),
            "Return": round(trade_return, 6),
            "Holding Days": holding_days,
            "Exit Reason": exit_reason,
        })

    return pd.DataFrame(records)


def trade_metrics(
    trades_df: pd.DataFrame,
    df: pd.DataFrame,
    risk_free_rate: float = 0.03,
) -> dict[str, float]:
    """Legacy trade metrics helper (kept for backward compat)."""
    nan = float("nan")
    n = len(trades_df)
    if n == 0:
        return {k: nan for k in (
            "num_trades", "win_rate", "profit_factor", "avg_holding_days",
            "best_trade", "worst_trade", "sharpe_ratio", "sortino_ratio",
        )} | {"num_trades": 0}

    returns = trades_df["Return"]
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())

    win_rate = len(wins) / n
    if gross_loss == 0:
        profit_factor = float("inf")
    elif gross_profit == 0:
        profit_factor = 0.0
    else:
        profit_factor = gross_profit / abs(gross_loss)

    strategy_ret = df["Strategy_Return"]
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = strategy_ret - daily_rf
    excess_mean = float(excess.mean())
    excess_std = float(excess.std(ddof=1))

    if excess_std > 0:
        sharpe_ratio = excess_mean / excess_std * sqrt(TRADING_DAYS)
    else:
        sharpe_ratio = float("inf") if excess_mean > 0 else nan

    downside = excess[excess < 0]
    if len(downside) > 0:
        ds_std = float(sqrt((downside ** 2).mean()))
        sortino_ratio = excess_mean / ds_std * sqrt(TRADING_DAYS) if ds_std > 0 else float("inf")
    else:
        sortino_ratio = float("inf")

    return {
        "num_trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_holding_days": float(trades_df["Holding Days"].mean()),
        "best_trade": float(returns.max()),
        "worst_trade": float(returns.min()),
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
    }


def performance_metrics(df: pd.DataFrame, risk_free_rate: float = 0.03) -> dict[str, float]:
    """Legacy performance metrics (kept for backward compat)."""
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)
    cumulative_strategy = df["Strategy_Equity"].iloc[-1] / df["Strategy_Equity"].iloc[0]
    cumulative_buy_hold = df["Buy_Hold_Equity"].iloc[-1] / df["Buy_Hold_Equity"].iloc[0]
    cagr_strategy = cumulative_strategy ** (1.0 / years) - 1.0
    cagr_buy_hold = cumulative_buy_hold ** (1.0 / years) - 1.0

    annual_return = df["Asset_Return"].mean() * TRADING_DAYS
    annual_volatility = df["Asset_Return"].std(ddof=0) * sqrt(TRADING_DAYS)
    kelly_fraction = 0.0
    if annual_volatility > 0:
        kelly_fraction = (annual_return - risk_free_rate) / (annual_volatility ** 2)

    return {
        "years": years,
        "cagr_strategy": cagr_strategy,
        "cagr_buy_hold": cagr_buy_hold,
        "mdd_strategy": df["Strategy_Drawdown"].min(),
        "mdd_buy_hold": df["Buy_Hold_Drawdown"].min(),
        "cumulative_strategy": cumulative_strategy,
        "cumulative_buy_hold": cumulative_buy_hold,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "kelly_fraction": kelly_fraction,
        "suggested_fractional_kelly": max(0.0, min(kelly_fraction * 0.5, 1.0)),
        "market_exposure": df["Invested"].mean(),
    }


# ---------------------------------------------------------------------------
# Market regime detection
# ---------------------------------------------------------------------------

def _detect_market_regime(df: pd.DataFrame, lookback: int = 50) -> str:
    """
    Classify recent market into 'bull', 'chop', or 'bear'.

    bull : Bull_Regime is True AND recent price range > 8 %
    chop : Bull_Regime is True BUT recent price range ≤ 8 % (low directional movement)
    bear : Bull_Regime is False
    """
    latest = df.iloc[-1]
    if not bool(latest.get("Bull_Regime", False)):
        return "bear"

    window = df["Close"].tail(lookback)
    if len(window) >= 10:
        hi, lo = float(window.max()), float(window.min())
        rng = (hi - lo) / lo if lo > 0 else 0.0
        if rng < 0.08:
            return "chop"
    return "bull"


# Factor weights per regime
_REGIME_WEIGHTS: dict[str, dict[str, int]] = {
    "bull": {"추세 방향": 30, "모멘텀(RSI)": 20, "이격도": 20, "손절 여유": 15, "변동성": 15},
    "chop": {"추세 방향": 20, "모멘텀(RSI)": 10, "이격도": 25, "손절 여유": 30, "변동성": 15},
    "bear": {"추세 방향": 40, "모멘텀(RSI)": 15, "이격도": 15, "손절 여유": 20, "변동성": 10},
}


# ---------------------------------------------------------------------------
# Entry assessment (regime-conditioned weights)
# ---------------------------------------------------------------------------

def entry_assessment(
    latest: pd.Series,
    metrics: dict[str, float],
    market_regime: str = "bull",
) -> dict[str, Any]:
    """0–100 entry suitability score with regime-conditioned factor weights."""
    close = float(latest["Close"])
    ema50 = float(latest["EMA_50"])
    sma200 = float(latest["SMA_200"])
    atr14 = float(latest["ATR_14"])
    rsi14 = float(latest.get("RSI_14", 50.0))
    stop = float(latest["ATR_Trailing_Stop"]) if pd.notna(latest.get("ATR_Trailing_Stop")) else None
    bull = bool(latest.get("Bull_Regime", False))

    weights = _REGIME_WEIGHTS.get(market_regime, _REGIME_WEIGHTS["bull"])
    factors: list[dict[str, Any]] = []

    # 1) Trend direction
    trend_gap = ema50 / sma200 - 1.0 if sma200 else 0.0
    if bull:
        trend_score = min(100.0, 50.0 + trend_gap * 1000.0)
    else:
        trend_score = max(0.0, 40.0 + trend_gap * 800.0)
    trend_score = max(0.0, min(100.0, trend_score))
    factors.append({
        "name": "추세 방향",
        "score": round(trend_score),
        "weight": weights["추세 방향"],
        "detail": (
            f"EMA50가 SMA200 {'위' if trend_gap >= 0 else '아래'} "
            f"{abs(trend_gap)*100:.1f}% ({'강세' if bull else '약세'})"
        ),
    })

    # 2) Momentum (RSI)
    if rsi14 >= 70:
        rsi_score = max(20.0, 100.0 - (rsi14 - 70.0) * 4.0)
        rsi_msg = "과열(되돌림 위험)"
    elif rsi14 >= 55:
        rsi_score = 90.0
        rsi_msg = "건강한 상승 모멘텀"
    elif rsi14 >= 45:
        rsi_score = 100.0
        rsi_msg = "안정 구간(눌림목)"
    elif rsi14 >= 30:
        rsi_score = 75.0
        rsi_msg = "약한 모멘텀"
    else:
        rsi_score = 40.0
        rsi_msg = "침체(하락 지속 위험)"
    factors.append({
        "name": "모멘텀(RSI)",
        "score": round(rsi_score),
        "weight": weights["모멘텀(RSI)"],
        "detail": f"RSI {rsi14:.0f} · {rsi_msg}",
    })

    # 3) EMA50 distance
    ema_gap = close / ema50 - 1.0 if ema50 else 0.0
    if -0.02 <= ema_gap <= 0.03:
        dist_score = 100.0
        dist_msg = "EMA50 부근(이상적 진입대)"
    elif 0.03 < ema_gap <= 0.08:
        dist_score = 70.0
        dist_msg = "다소 위로 확장"
    elif ema_gap > 0.08:
        dist_score = max(35.0, 70.0 - (ema_gap - 0.08) * 300.0)
        dist_msg = "과도하게 확장(추격 위험)"
    else:
        dist_score = 55.0 if bull else 35.0
        dist_msg = "EMA50 아래(눌림 또는 약세)"
    factors.append({
        "name": "이격도",
        "score": round(dist_score),
        "weight": weights["이격도"],
        "detail": (
            f"현재가가 EMA50 {'위' if ema_gap >= 0 else '아래'} "
            f"{abs(ema_gap)*100:.1f}% · {dist_msg}"
        ),
    })

    # 4) Stop-loss headroom (손절 여유)
    if stop is not None and close > 0:
        downside = max(0.0, close / stop - 1.0)
        if close < stop:
            risk_score = 10.0
            risk_msg = "손절선 이탈(진입 부적합)"
        elif downside <= 0.08:
            risk_score = 100.0
            risk_msg = f"손절선까지 -{downside*100:.1f}% (리스크 작음)"
        elif downside <= 0.18:
            risk_score = 70.0
            risk_msg = f"손절선까지 -{downside*100:.1f}%"
        else:
            risk_score = 45.0
            risk_msg = f"손절선까지 -{downside*100:.1f}% (손절폭 큼)"
    else:
        risk_score = 50.0
        risk_msg = "손절선 미형성"
    factors.append({
        "name": "손절 여유",
        "score": round(risk_score),
        "weight": weights["손절 여유"],
        "detail": risk_msg,
    })

    # 5) Volatility
    vol_ratio = atr14 / close if close else 0.0
    if vol_ratio <= 0.02:
        vol_score = 100.0
    elif vol_ratio <= 0.04:
        vol_score = 70.0
    else:
        vol_score = max(30.0, 70.0 - (vol_ratio - 0.04) * 1000.0)
    factors.append({
        "name": "변동성",
        "score": round(vol_score),
        "weight": weights["변동성"],
        "detail": f"일일 변동성(ATR/가격) {vol_ratio*100:.1f}%",
    })

    total_weight = sum(f["weight"] for f in factors)
    score = sum(f["score"] * f["weight"] for f in factors) / total_weight

    if not bull:
        score = min(score, 45.0)
    if stop is not None and close < stop:
        score = min(score, 30.0)

    score = round(max(0.0, min(100.0, score)), 1)

    if score >= 75:
        verdict = "적극 진입"
    elif score >= 60:
        verdict = "진입 고려"
    elif score >= 40:
        verdict = "관망"
    else:
        verdict = "진입 회피"

    return {"score": score, "verdict": verdict, "factors": factors, "rsi14": rsi14}


# ---------------------------------------------------------------------------
# Signal / regime helpers
# ---------------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return _atr_fn(high, low, close, period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return _rsi_fn(close, period)


def decide_longterm_action(latest: pd.Series, prev: pd.Series) -> str:
    if latest["Death_Cross"]:
        return "전량 매도/현금화: EMA50이 SMA200 아래로 데드크로스"
    if latest["Golden_Cross"]:
        return "장기 매수 전환: EMA50이 SMA200 위로 골든크로스"
    if not latest["Bull_Regime"]:
        return "신규 매수 보류: 장기 약세장 필터 작동"
    if latest["Close"] < latest["ATR_Trailing_Stop"]:
        return "비중 축소/청산: ATR 장기 추적 손절선 이탈"
    if prev["Close"] < prev["ATR_Trailing_Stop"] and latest["Close"] >= latest["ATR_Trailing_Stop"]:
        return "재진입 후보: 장기 상승장 안에서 ATR 손절선 회복"
    return "장기 보유 유지: 상승장 필터와 ATR 방어선 유지"


def classify_longterm_regime(latest: pd.Series) -> str:
    if latest["Bull_Regime"] and latest["Close"] >= latest["ATR_Trailing_Stop"]:
        return "장기 상승장/보유 국면"
    if latest["Bull_Regime"]:
        return "장기 상승장 안의 변동성 경고 국면"
    return "장기 약세장/현금화 국면"


def confidence_label(latest: pd.Series, metrics: dict[str, float]) -> str:
    distance = latest["EMA_50"] / latest["SMA_200"] - 1.0
    if latest["Bull_Regime"] and distance > 0.05 and metrics.get("market_exposure", 0) > 0.35:
        return "높음"
    if abs(distance) <= 0.03:
        return "보통"
    return "낮음" if not latest["Bull_Regime"] else "보통"


def explain_longterm(
    latest: pd.Series,
    prev: pd.Series,
    metrics: dict[str, float],
    atr_multiplier: float,
) -> tuple[list[str], list[str]]:
    insights: list[str] = []
    warnings: list[str] = [
        "본 결과는 장기 추세 추종 백테스트 기반 참고용 분석이며 투자 조언이나 수익 보장을 의미하지 않습니다.",
        "yfinance 데이터는 지연/누락될 수 있으므로 실제 주문 전 증권사/거래소 시세를 확인하세요.",
    ]

    if latest["EMA_50"] > latest["SMA_200"]:
        insights.append("EMA50이 SMA200 위에 있어 장기 강세 필터가 켜져 있습니다.")
    else:
        warnings.append("EMA50이 SMA200 아래에 있어 장기 약세장 방어 모드입니다.")

    if latest["Golden_Cross"]:
        insights.append("오늘 기준 장기 골든크로스가 발생했습니다.")
    if latest["Death_Cross"]:
        warnings.append("오늘 기준 장기 데드크로스가 발생했습니다.")

    stop_val = latest.get("ATR_Trailing_Stop")
    if pd.notna(stop_val):
        if latest["Close"] >= float(stop_val):
            insights.append(f"가격이 ATR {atr_multiplier:.1f}배 추적 손절선 위에 있습니다.")
        else:
            warnings.append(f"가격이 ATR {atr_multiplier:.1f}배 추적 손절선을 이탈했습니다.")

    cagr_strat = metrics.get("cagr_strategy", 0.0)
    cagr_bh = metrics.get("cagr_buy_hold", 0.0)
    if cagr_strat > cagr_bh:
        insights.append("해당 기간에는 장기 필터 전략 CAGR이 단순 보유보다 높았습니다.")
    else:
        warnings.append("해당 기간에는 단순 보유 CAGR이 장기 필터 전략보다 높았습니다.")

    mdd_strat = metrics.get("mdd_strategy", 0.0)
    mdd_bh = metrics.get("mdd_buy_hold", 0.0)
    if mdd_strat > mdd_bh:
        insights.append("장기 필터 전략의 최대 낙폭이 단순 보유보다 작았습니다.")
    else:
        warnings.append("장기 필터 전략의 최대 낙폭 축소 효과가 제한적이었습니다.")

    kelly = metrics.get("kelly_fraction", 0.0)
    if kelly <= 0:
        warnings.append("장기 켈리 비중이 0 이하로 계산되어 기대수익 대비 변동성이 불리합니다.")
    elif kelly > 1:
        warnings.append("장기 켈리 비중이 100%를 초과하므로 실전에서는 분수 켈리로 제한하는 편이 안전합니다.")

    return insights[:5], warnings


# ---------------------------------------------------------------------------
# Telegram report formatter
# ---------------------------------------------------------------------------

def format_telegram_report(result: LongTermResult) -> str:
    insights = "\n".join(f"- {item}" for item in result.insights) or "- 뚜렷한 장기 강세 근거가 제한적입니다."
    warnings_text = "\n".join(f"- {item}" for item in result.warnings)
    factors = result.entry_factors or []
    factor_lines = "\n".join(f"- {f['name']}: {f['score']}점 · {f['detail']}" for f in factors)

    cost_note = ""
    if result.commission_bps > 0 or result.slippage_beta > 0:
        cost_note = f" (수수료 {result.commission_bps:.0f}bps + ATR슬리피지 β={result.slippage_beta})"

    return (
        f"{result.ticker} 진입 분석 ({result.as_of})\n\n"
        f"■ 진입 판정: {result.entry_verdict}\n"
        f"■ 진입 적합도: {result.entry_score:.0f} / 100점\n\n"
        "지표별 점수\n"
        f"{factor_lines}\n\n"
        f"전략 신호: {result.action}\n"
        f"신뢰도: {result.confidence}\n"
        f"국면: {result.regime}\n"
        f"분석 기간: {result.start_date} ~ {result.as_of} / {result.rows:,}거래일\n\n"
        "장기 추세 필터\n"
        f"- 현재가: {fmt(result.current_price)}\n"
        f"- EMA50: {fmt(result.ema50)}\n"
        f"- SMA200: {fmt(result.sma200)}\n"
        f"- 상태: {'보유 가능' if result.invested_now else '현금/관망'}\n\n"
        "ATR 장기 추적 손절\n"
        f"- ATR14: {fmt(result.atr14)}\n"
        f"- 승수: {result.atr_multiplier:.1f}x\n"
        f"- 초기 손절 참고선: {fmt(result.initial_stop)}\n"
        f"- 추적 손절선: {fmt(result.trailing_stop)}\n\n"
        "40년형 백테스트 요약\n"
        f"- 전략 CAGR: {pct(result.cagr_strategy)} / 단순보유 CAGR: {pct(result.cagr_buy_hold)}\n"
        f"- 전략 MDD: {pct(result.mdd_strategy)} / 단순보유 MDD: {pct(result.mdd_buy_hold)}\n"
        f"- 전략 누적배수: {result.cumulative_strategy:.2f}x / 단순보유: {result.cumulative_buy_hold:.2f}x\n"
        f"- Calmar: {_fmt_ratio(result.calmar_ratio)} / 노출조정 CAGR: {pct(result.exposure_adj_cagr)}\n"
        f"- 시장 노출 비율: {pct(result.market_exposure)}\n\n"
        "장기 켈리 자산 배분\n"
        f"- 연환산 기대수익률 μ: {pct(result.annual_return)}\n"
        f"- 연환산 변동성 σ: {pct(result.annual_volatility)}\n"
        f"- 무위험 이자율 r: {pct(result.risk_free_rate)}\n"
        f"- 켈리 비중 f*: {pct(result.kelly_fraction)}\n"
        f"- 실전 참고 분수 켈리(0~100% 제한): {pct(result.suggested_fractional_kelly)}\n\n"
        "트레이드별 성과\n"
        f"- 총 거래 수: {result.num_trades}회 | 승률: {pct(result.win_rate)}{cost_note}\n"
        f"- Sharpe: {_fmt_ratio(result.sharpe_ratio)} | Sortino: {_fmt_ratio(result.sortino_ratio)}\n"
        f"- Profit Factor: {_fmt_ratio(result.profit_factor)}\n"
        f"- Turnover: {_fmt_ratio(result.turnover)}회/년\n"
        f"- 평균 보유: {_fmt_days(result.avg_holding_days)} | "
        f"최고: {pct(result.best_trade)} / 최저: {pct(result.worst_trade)}\n\n"
        "핵심 해석\n"
        f"{insights}\n\n"
        "주의\n"
        f"{warnings_text}"
    )


# ---------------------------------------------------------------------------
# Utility formatters
# ---------------------------------------------------------------------------

def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(number) or np.isinf(number):
        return "N/A"
    if abs(number) >= 100:
        return f"{number:,.2f}"
    if abs(number) >= 1:
        return f"{number:,.2f}"
    return f"{number:,.4f}"


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(number) or np.isinf(number):
        return "N/A"
    return f"{number * 100:,.2f}%"


def format_date(value: Any) -> str:
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def normalize_longterm_period(period: str) -> str:
    normalized = (period or "max").lower()
    if normalized in {"5y", "10y", "max"}:
        return normalized
    return "max"


def _fmt_ratio(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(number):
        return "N/A"
    if np.isinf(number):
        return "∞" if number > 0 else "-∞"
    return f"{number:.2f}"


def _fmt_days(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(number):
        return "N/A"
    return f"{number:.0f}일"
