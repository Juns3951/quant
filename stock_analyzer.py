from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LONGTERM_START = "1986-01-01"
TRADING_DAYS = 252


@dataclass(frozen=True)
class LongTermResult:
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


class AnalyzerError(RuntimeError):
    pass


def fetch_history(ticker: str, period: str = "max", start: str = LONGTERM_START) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise AnalyzerError(
            "yfinance가 설치되어 있지 않습니다. `pip install -r requirements.txt`를 먼저 실행하세요."
        ) from exc

    cache_dir = Path(__file__).resolve().parent / ".yfinance-cache"
    cache_dir.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    if period == "max":
        data = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
    else:
        data = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)

    if data.empty:
        raise AnalyzerError(f"{ticker} 데이터를 가져오지 못했습니다. 티커 표기를 확인하세요.")
    return data


def analyze_ticker(
    ticker: str,
    period: str = "max",
    benchmark_ticker: str | None = None,
) -> LongTermResult:
    del benchmark_ticker
    ticker = ticker.strip().upper()
    effective_period = normalize_longterm_period(period)
    frame = fetch_history(ticker, period=effective_period)
    return analyze_price_frame(ticker=ticker, frame=frame)


def analyze_price_frame(
    ticker: str,
    frame: pd.DataFrame,
    atr_multiplier: float = 3.5,
    risk_free_rate: float = 0.03,
    initial_capital: float = 10_000_000.0,
) -> LongTermResult:
    df = calculate_longterm_backtest(
        clean_price_frame(frame),
        atr_multiplier=atr_multiplier,
        initial_capital=initial_capital,
    )
    df = df.dropna(subset=["Close", "EMA_50", "SMA_200", "ATR_14"])
    if len(df) < 220:
        raise AnalyzerError("장기 분석에 필요한 데이터가 부족합니다. 최소 220거래일 이상이 필요합니다.")

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    metrics = performance_metrics(df, risk_free_rate=risk_free_rate)
    action = decide_longterm_action(latest, prev)
    regime = classify_longterm_regime(latest)
    insights, warnings = explain_longterm(latest, prev, metrics, atr_multiplier)

    trades_df = extract_trades(df)
    t_metrics = trade_metrics(trades_df, df, risk_free_rate=risk_free_rate)

    return LongTermResult(
        ticker=ticker.strip().upper(),
        as_of=format_date(df.index[-1]),
        start_date=format_date(df.index[0]),
        rows=len(df),
        action=action,
        regime=regime,
        confidence=confidence_label(latest, metrics),
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
        market_exposure=float(metrics["market_exposure"]),
        cagr_strategy=float(metrics["cagr_strategy"]),
        cagr_buy_hold=float(metrics["cagr_buy_hold"]),
        mdd_strategy=float(metrics["mdd_strategy"]),
        mdd_buy_hold=float(metrics["mdd_buy_hold"]),
        cumulative_strategy=float(metrics["cumulative_strategy"]),
        cumulative_buy_hold=float(metrics["cumulative_buy_hold"]),
        annual_return=float(metrics["annual_return"]),
        annual_volatility=float(metrics["annual_volatility"]),
        risk_free_rate=float(risk_free_rate),
        kelly_fraction=float(metrics["kelly_fraction"]),
        suggested_fractional_kelly=float(metrics["suggested_fractional_kelly"]),
        warnings=warnings,
        insights=insights,
        frame=df,
        trades=trades_df,
        sharpe_ratio=float(t_metrics["sharpe_ratio"]),
        sortino_ratio=float(t_metrics["sortino_ratio"]),
        win_rate=float(t_metrics["win_rate"]),
        profit_factor=float(t_metrics["profit_factor"]),
        num_trades=int(t_metrics["num_trades"]),
        avg_holding_days=float(t_metrics["avg_holding_days"]),
        best_trade=float(t_metrics["best_trade"]),
        worst_trade=float(t_metrics["worst_trade"]),
    )


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


def calculate_longterm_backtest(
    df: pd.DataFrame,
    atr_multiplier: float = 3.5,
    initial_capital: float = 10_000_000.0,
) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    out["SMA_200"] = close.rolling(200).mean()
    out["ATR_14"] = atr(out["High"], out["Low"], close)

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
    nan = float("nan")
    n = len(trades_df)
    if n == 0:
        return {
            "num_trades": 0,
            "win_rate": nan,
            "profit_factor": nan,
            "avg_holding_days": nan,
            "best_trade": nan,
            "worst_trade": nan,
            "sharpe_ratio": nan,
            "sortino_ratio": nan,
        }

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
        downside_std = float(sqrt((downside**2).mean()))
        sortino_ratio = excess_mean / downside_std * sqrt(TRADING_DAYS) if downside_std > 0 else float("inf")
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


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def performance_metrics(df: pd.DataFrame, risk_free_rate: float = 0.03) -> dict[str, float]:
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)
    cumulative_strategy = df["Strategy_Equity"].iloc[-1] / df["Strategy_Equity"].iloc[0]
    cumulative_buy_hold = df["Buy_Hold_Equity"].iloc[-1] / df["Buy_Hold_Equity"].iloc[0]
    cagr_strategy = cumulative_strategy ** (1.0 / years) - 1.0
    cagr_buy_hold = cumulative_buy_hold ** (1.0 / years) - 1.0

    annual_return = df["Asset_Return"].mean() * TRADING_DAYS
    annual_volatility = df["Asset_Return"].std(ddof=0) * sqrt(TRADING_DAYS)
    kelly_fraction = 0.0
    if annual_volatility > 0:
        kelly_fraction = (annual_return - risk_free_rate) / (annual_volatility**2)

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
    if latest["Bull_Regime"] and distance > 0.05 and metrics["market_exposure"] > 0.35:
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

    stop_gap = latest["Close"] / latest["ATR_Trailing_Stop"] - 1.0
    if latest["Close"] >= latest["ATR_Trailing_Stop"]:
        insights.append(f"가격이 ATR {atr_multiplier:.1f}배 추적 손절선 위에 있습니다.")
    else:
        warnings.append(f"가격이 ATR {atr_multiplier:.1f}배 추적 손절선을 이탈했습니다.")

    if metrics["cagr_strategy"] > metrics["cagr_buy_hold"]:
        insights.append("해당 기간에는 장기 필터 전략 CAGR이 단순 보유보다 높았습니다.")
    else:
        warnings.append("해당 기간에는 단순 보유 CAGR이 장기 필터 전략보다 높았습니다.")

    if metrics["mdd_strategy"] > metrics["mdd_buy_hold"]:
        insights.append("장기 필터 전략의 최대 낙폭이 단순 보유보다 작았습니다.")
    else:
        warnings.append("장기 필터 전략의 최대 낙폭 축소 효과가 제한적이었습니다.")

    if abs(stop_gap) < 0.05 and latest["Bull_Regime"]:
        warnings.append("현재가가 추적 손절선과 5% 이내로 가까워 방어선 테스트 구간입니다.")

    if metrics["kelly_fraction"] <= 0:
        warnings.append("장기 켈리 비중이 0 이하로 계산되어 기대수익 대비 변동성이 불리합니다.")
    elif metrics["kelly_fraction"] > 1:
        warnings.append("장기 켈리 비중이 100%를 초과하므로 실전에서는 분수 켈리로 제한하는 편이 안전합니다.")

    return insights[:5], warnings


def format_telegram_report(result: LongTermResult) -> str:
    insights = "\n".join(f"- {item}" for item in result.insights) or "- 뚜렷한 장기 강세 근거가 제한적입니다."
    warnings = "\n".join(f"- {item}" for item in result.warnings)

    return (
        f"{result.ticker} 장기 퀀트 분석 ({result.as_of})\n\n"
        f"판정: {result.action}\n"
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
        f"- 전략 누적배수: {result.cumulative_strategy:.2f}x / 단순보유 누적배수: {result.cumulative_buy_hold:.2f}x\n"
        f"- 시장 노출 비율: {pct(result.market_exposure)}\n\n"
        "장기 켈리 자산 배분\n"
        f"- 연환산 기대수익률 μ: {pct(result.annual_return)}\n"
        f"- 연환산 변동성 σ: {pct(result.annual_volatility)}\n"
        f"- 무위험 이자율 r: {pct(result.risk_free_rate)}\n"
        f"- 켈리 비중 f*: {pct(result.kelly_fraction)}\n"
        f"- 실전 참고 분수 켈리(0~100% 제한): {pct(result.suggested_fractional_kelly)}\n\n"
        "트레이드별 성과\n"
        f"- 총 거래 수: {result.num_trades}회 | 승률: {pct(result.win_rate)}\n"
        f"- Sharpe: {_fmt_ratio(result.sharpe_ratio)} | Sortino: {_fmt_ratio(result.sortino_ratio)}\n"
        f"- Profit Factor: {_fmt_ratio(result.profit_factor)}\n"
        f"- 평균 보유: {_fmt_days(result.avg_holding_days)} | "
        f"최고: {pct(result.best_trade)} / 최저: {pct(result.worst_trade)}\n\n"
        "핵심 해석\n"
        f"{insights}\n\n"
        "주의\n"
        f"{warnings}"
    )


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
