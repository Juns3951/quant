from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilterResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class AnalysisResult:
    ticker: str
    as_of: str
    action: str
    confidence: str
    regime: str
    score: int
    latest: dict[str, float]
    filters: list[FilterResult]
    risk_levels: dict[str, float]
    bullish_points: list[str]
    bearish_points: list[str]
    warnings: list[str]
    kill_switch: list[str]
    false_breakout_flags: list[str]
    rows: int


class AnalyzerError(RuntimeError):
    pass


def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise AnalyzerError(
            "yfinance가 설치되어 있지 않습니다. `pip install -r requirements.txt`를 먼저 실행하세요."
        ) from exc

    data = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if data.empty:
        raise AnalyzerError(f"{ticker} 데이터를 가져오지 못했습니다. 티커 표기를 확인하세요.")
    return data


def analyze_ticker(
    ticker: str,
    period: str = "1y",
    benchmark_ticker: str | None = "SPY",
) -> AnalysisResult:
    ticker = ticker.strip().upper()
    price_frame = fetch_history(ticker, period=period)

    benchmark_frame = None
    if benchmark_ticker:
        try:
            benchmark_frame = fetch_history(benchmark_ticker, period=max_period(period, "1y"))
        except Exception:
            benchmark_frame = None

    vix_frame = None
    try:
        vix_frame = fetch_history("^VIX", period="3mo")
    except Exception:
        vix_frame = None

    return analyze_price_frame(
        ticker=ticker,
        frame=price_frame,
        benchmark_frame=benchmark_frame,
        vix_frame=vix_frame,
    )


def analyze_price_frame(
    ticker: str,
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None = None,
    vix_frame: pd.DataFrame | None = None,
) -> AnalysisResult:
    df = calculate_indicators(clean_price_frame(frame))
    df = df.dropna(subset=["Close", "EMA_50", "RSI", "MACD", "ATR_14", "ADX_14"])
    if len(df) < 60:
        raise AnalyzerError("분석에 필요한 일봉 데이터가 부족합니다. 최소 60개 이상의 캔들이 필요합니다.")

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    regime = classify_regime(latest)
    filters = build_filter_chain(latest, prev)
    false_breakout_flags = detect_false_breakout(df)
    kill_switch = detect_kill_switch(df, benchmark_frame, vix_frame)
    bullish_points, bearish_points = explain_signals(latest, prev, regime, false_breakout_flags)
    score = composite_score(latest, prev, filters, false_breakout_flags, kill_switch)
    risk_levels = build_risk_levels(df)
    action = decide_action(score, filters, latest, kill_switch, false_breakout_flags, risk_levels)
    confidence = confidence_label(score, filters, kill_switch, false_breakout_flags)

    warnings = [
        "본 결과는 기술적 지표 기반 참고용 분석이며 투자 조언이나 수익 보장을 의미하지 않습니다.",
        "yfinance 데이터는 지연/누락될 수 있으므로 실제 주문 전 증권사 시세로 재확인하세요.",
    ]
    if latest["Volume_Ratio"] < 0.7:
        warnings.append("최근 거래량이 20일 평균보다 낮아 신호 신뢰도가 떨어질 수 있습니다.")
    if latest["ATR_Ratio"] > 1.5:
        warnings.append("ATR이 평소보다 급등해 손절 폭과 포지션 크기를 보수적으로 잡아야 합니다.")

    return AnalysisResult(
        ticker=ticker,
        as_of=format_date(df.index[-1]),
        action=action,
        confidence=confidence,
        regime=regime,
        score=score,
        latest=latest_to_dict(latest),
        filters=filters,
        risk_levels=risk_levels,
        bullish_points=bullish_points,
        bearish_points=bearish_points,
        warnings=warnings,
        kill_switch=kill_switch,
        false_breakout_flags=false_breakout_flags,
        rows=len(df),
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


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    out["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    out["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA_200"] = close.ewm(span=200, adjust=False).mean()
    out["RSI"] = rsi(close)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(close)
    out["ATR_14"] = atr(high, low, close)
    out["ADX_14"] = adx(high, low, close)

    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    out["BB_Middle"] = bb_middle
    out["BB_Upper"] = bb_middle + (2.0 * bb_std)
    out["BB_Lower"] = bb_middle - (2.0 * bb_std)
    out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / bb_middle.replace(0.0, np.nan)

    typical_price = (high + low + close) / 3.0
    rolling_volume = volume.rolling(20).sum()
    out["VWAP_20"] = (typical_price * volume).rolling(20).sum() / rolling_volume.replace(0.0, np.nan)
    out["Volume_MA20"] = volume.rolling(20).mean()
    out["Volume_Ratio"] = volume / out["Volume_MA20"].replace(0.0, np.nan)
    out["ATR_MA20"] = out["ATR_14"].rolling(20).mean()
    out["ATR_Ratio"] = out["ATR_14"] / out["ATR_MA20"].replace(0.0, np.nan)
    out["Realized_Vol_5D"] = close.pct_change().rolling(5).std(ddof=0)

    rsi_min = out["RSI"].rolling(14).min()
    rsi_max = out["RSI"].rolling(14).max()
    out["Stoch_RSI_K"] = 100.0 * (out["RSI"] - rsi_min) / (rsi_max - rsi_min).replace(0.0, np.nan)
    out["Stoch_RSI_D"] = out["Stoch_RSI_K"].rolling(3).mean()
    return out


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    value = 100.0 - (100.0 / (1.0 + rs))
    value = value.where(avg_loss != 0.0, 100.0)
    value = value.where(avg_gain != 0.0, 0.0)
    return value


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    line = ema_12 - ema_26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    atr_value = atr(high, low, close, period)
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr_value.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr_value.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def classify_regime(latest: pd.Series) -> str:
    if latest["ATR_Ratio"] >= 1.5 or latest["BB_Width"] >= 0.18:
        return "변동성 급증 국면"
    if latest["ADX_14"] >= 20:
        if latest["Close"] > latest["EMA_50"] and latest["EMA_10"] > latest["EMA_50"]:
            return "강한 상승 추세 국면"
        if latest["Close"] < latest["EMA_50"] and latest["EMA_10"] < latest["EMA_50"]:
            return "하락 추세 국면"
        return "추세 전환 감시 국면"
    return "박스권/평균회귀 국면"


def build_filter_chain(latest: pd.Series, prev: pd.Series) -> list[FilterResult]:
    trend_pass = bool(latest["Close"] > latest["EMA_50"] and latest["ADX_14"] > 20)
    rsi_rebound = bool(prev["RSI"] <= 30 < latest["RSI"])
    rsi_constructive = bool(30 < latest["RSI"] < 70)
    macd_bull = bool(latest["MACD"] > latest["MACD_Signal"])
    momentum_pass = bool(macd_bull and (rsi_rebound or rsi_constructive))
    volume_pass = bool(latest["Volume_Ratio"] > 1.5 and latest["Close"] > latest["VWAP_20"])

    return [
        FilterResult(
            "1단계 추세",
            trend_pass,
            f"종가 {fmt(latest['Close'])} / EMA50 {fmt(latest['EMA_50'])} / ADX {latest['ADX_14']:.1f}",
        ),
        FilterResult(
            "2단계 모멘텀",
            momentum_pass,
            f"RSI {latest['RSI']:.1f} / MACD {fmt(latest['MACD'])} vs Signal {fmt(latest['MACD_Signal'])}",
        ),
        FilterResult(
            "3단계 거래량",
            volume_pass,
            f"거래량 {latest['Volume_Ratio']:.2f}배 / VWAP20 {fmt(latest['VWAP_20'])}",
        ),
    ]


def detect_false_breakout(df: pd.DataFrame) -> list[str]:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prior = df.iloc[:-1]
    if len(prior) < 60:
        return []

    resistance = prior["High"].tail(60).max()
    flags: list[str] = []
    is_breakout = latest["Close"] > resistance
    if not is_breakout:
        return flags

    prev_resistance = df.iloc[:-2]["High"].tail(60).max()
    if not (prev["Close"] > prev_resistance and latest["Close"] > resistance):
        flags.append("저항선 상단에서 2거래일 연속 종가 안착이 아직 확인되지 않았습니다.")

    recent_price_high = prior["Close"].tail(20).max()
    recent_rsi_high = prior["RSI"].tail(20).max()
    if latest["Close"] > recent_price_high and latest["RSI"] < recent_rsi_high:
        flags.append("가격은 신고점을 만들었지만 RSI 고점은 낮아져 하락 다이버전스 위험이 있습니다.")

    if latest["ATR_Ratio"] < 1.5:
        flags.append("돌파 대비 ATR 확장이 부족해 변동성 동력이 약합니다.")

    if abs(latest["Close"] - latest["VWAP_20"]) > 1.2 * latest["ATR_14"]:
        flags.append("VWAP20 대비 이격이 1.2 ATR을 초과해 고무줄 되돌림 위험이 큽니다.")

    return flags


def detect_kill_switch(
    df: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None,
    vix_frame: pd.DataFrame | None,
) -> list[str]:
    triggers: list[str] = []
    latest = df.iloc[-1]
    if latest["Realized_Vol_5D"] > 0.07:
        triggers.append("종목 5일 실현 변동성이 7%를 초과했습니다.")

    if benchmark_frame is not None and not benchmark_frame.empty:
        try:
            benchmark = calculate_indicators(clean_price_frame(benchmark_frame)).dropna(subset=["EMA_200"])
            if len(benchmark) > 0:
                b = benchmark.iloc[-1]
                if b["Close"] < b["EMA_200"]:
                    triggers.append("벤치마크가 200일 EMA 아래에 있어 시장 방어 모드가 필요합니다.")
        except Exception:
            pass

    if vix_frame is not None and not vix_frame.empty:
        try:
            vix = clean_price_frame(vix_frame)
            vix_close = float(vix["Close"].iloc[-1])
            if vix_close > 40:
                triggers.append(f"VIX가 {vix_close:.1f}로 40을 초과했습니다.")
        except Exception:
            pass

    return triggers


def explain_signals(
    latest: pd.Series,
    prev: pd.Series,
    regime: str,
    false_breakout_flags: list[str],
) -> tuple[list[str], list[str]]:
    bullish: list[str] = []
    bearish: list[str] = []

    if latest["EMA_10"] > latest["EMA_50"]:
        bullish.append("EMA10이 EMA50 위에 있어 중기 추세가 우호적입니다.")
    else:
        bearish.append("EMA10이 EMA50 아래에 있어 추세 확인이 약합니다.")

    if latest["MACD"] > latest["MACD_Signal"]:
        bullish.append("MACD가 시그널선 위에 있어 모멘텀이 개선 중입니다.")
    else:
        bearish.append("MACD가 시그널선 아래에 있어 상승 모멘텀이 부족합니다.")

    if 35 <= latest["RSI"] <= 65:
        bullish.append("RSI가 과열/과매도 극단을 벗어난 안정 구간입니다.")
    elif latest["RSI"] > 70:
        bearish.append("RSI가 과매수권이라 신규 추격 매수는 보수적으로 봐야 합니다.")
    elif latest["RSI"] < 30:
        bearish.append("RSI가 과매도권이지만 아직 탈출 확인이 필요합니다.")

    if latest["Volume_Ratio"] > 1.5:
        bullish.append("거래량이 20일 평균의 1.5배를 넘어 수급 확인이 강합니다.")
    else:
        bearish.append("거래량 필터가 아직 강한 수급 유입을 확인하지 못했습니다.")

    if latest["Close"] > latest["VWAP_20"]:
        bullish.append("종가가 VWAP20 위에 있어 평균 매입 단가 대비 우위가 있습니다.")
    else:
        bearish.append("종가가 VWAP20 아래라 매수세 장악력이 약합니다.")

    if "변동성 급증" in regime:
        bearish.append("변동성 급증 국면이라 포지션 크기와 손절 폭을 줄여야 합니다.")
    if false_breakout_flags:
        bearish.append("돌파 신호에 가짜 돌파 의심 조건이 붙었습니다.")

    return bullish[:4], bearish[:5]


def composite_score(
    latest: pd.Series,
    prev: pd.Series,
    filters: list[FilterResult],
    false_breakout_flags: list[str],
    kill_switch: list[str],
) -> int:
    score = 0.0

    if latest["Close"] > latest["EMA_50"]:
        score += 12
    if latest["EMA_10"] > latest["EMA_50"]:
        score += 13
    if latest["ADX_14"] > 20:
        score += 10
    if latest["MACD"] > latest["MACD_Signal"]:
        score += 15
    if prev["RSI"] <= 30 < latest["RSI"]:
        score += 12
    elif 35 <= latest["RSI"] <= 65:
        score += 10
    elif latest["RSI"] > 75:
        score -= 8
    if latest["Close"] > latest["VWAP_20"]:
        score += 10
    if latest["Volume_Ratio"] > 1.5:
        score += 13
    elif latest["Volume_Ratio"] > 1.0:
        score += 6
    if latest["Close"] > latest["BB_Middle"]:
        score += 5
    if latest["ATR_Ratio"] > 1.5:
        score -= 10

    score += 5 * sum(1 for item in filters if item.passed)
    score -= 8 * len(false_breakout_flags)
    score -= 20 * len(kill_switch)
    return int(max(0, min(100, round(score))))


def build_risk_levels(df: pd.DataFrame) -> dict[str, float]:
    latest = df.iloc[-1]
    recent_high = float(df["High"].tail(20).max())
    recent_low = float(df["Low"].tail(20).min())
    close = float(latest["Close"])
    atr_value = float(latest["ATR_14"])
    initial_stop = close - (2.0 * atr_value)
    trailing_stop = max(initial_stop, recent_high - (2.0 * atr_value))
    return {
        "close": close,
        "initial_stop": initial_stop,
        "trailing_stop": trailing_stop,
        "risk_per_share": close - initial_stop,
        "recent_support": recent_low,
        "recent_resistance": recent_high,
        "bollinger_upper": float(latest["BB_Upper"]),
        "bollinger_lower": float(latest["BB_Lower"]),
    }


def decide_action(
    score: int,
    filters: list[FilterResult],
    latest: pd.Series,
    kill_switch: list[str],
    false_breakout_flags: list[str],
    risk_levels: dict[str, float],
) -> str:
    if kill_switch:
        return "위험 회피: 신규 매수 금지 및 보유 포지션 축소/청산 검토"

    all_filters_pass = all(item.passed for item in filters)
    sell_pressure = (
        latest["Close"] < risk_levels["trailing_stop"]
        or (latest["Close"] < latest["EMA_50"] and latest["MACD"] < latest["MACD_Signal"])
        or (latest["RSI"] > 75 and latest["Stoch_RSI_K"] < latest["Stoch_RSI_D"])
    )
    if sell_pressure:
        return "매도/비중 축소 우선"
    if all_filters_pass and score >= 70 and not false_breakout_flags:
        return "매수 후보: 3단계 필터 통과"
    if score >= 60 and not false_breakout_flags:
        return "관심 후보: 분할 진입만 검토"
    if score >= 45:
        return "관망/보유: 추가 확인 필요"
    return "신규 진입 보류"


def confidence_label(
    score: int,
    filters: list[FilterResult],
    kill_switch: list[str],
    false_breakout_flags: list[str],
) -> str:
    if kill_switch:
        return "낮음"
    passed = sum(1 for item in filters if item.passed)
    if score >= 70 and passed == 3 and not false_breakout_flags:
        return "높음"
    if score >= 50 and passed >= 2:
        return "보통"
    return "낮음"


def latest_to_dict(latest: pd.Series) -> dict[str, float]:
    fields = [
        "Close",
        "EMA_10",
        "EMA_20",
        "EMA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "ADX_14",
        "ATR_14",
        "ATR_Ratio",
        "VWAP_20",
        "Volume_Ratio",
        "BB_Upper",
        "BB_Lower",
    ]
    return {field: float(latest[field]) for field in fields}


def format_telegram_report(result: AnalysisResult) -> str:
    filter_lines = "\n".join(
        f"- {item.name}: {'PASS' if item.passed else 'FAIL'} | {item.detail}" for item in result.filters
    )
    bullish = "\n".join(f"- {item}" for item in result.bullish_points) or "- 뚜렷한 강세 근거가 부족합니다."
    bearish = "\n".join(f"- {item}" for item in result.bearish_points) or "- 뚜렷한 약세 근거가 제한적입니다."
    kill = "\n".join(f"- {item}" for item in result.kill_switch) or "- 감지된 킬스위치 조건 없음"
    breakout = "\n".join(f"- {item}" for item in result.false_breakout_flags) or "- 가짜 돌파 위험 신호 없음"
    warnings = "\n".join(f"- {item}" for item in result.warnings)

    latest = result.latest
    risk = result.risk_levels
    return (
        f"{result.ticker} 기술적 분석 ({result.as_of})\n\n"
        f"판정: {result.action}\n"
        f"신뢰도: {result.confidence} / 복합점수: {result.score}/100\n"
        f"시장 국면: {result.regime}\n"
        f"현재가: {fmt(latest['Close'])}\n\n"
        "3단계 필터\n"
        f"{filter_lines}\n\n"
        "핵심 강세 근거\n"
        f"{bullish}\n\n"
        "주의/약세 근거\n"
        f"{bearish}\n\n"
        "리스크 가격대\n"
        f"- 초기 손절 기준: {fmt(risk['initial_stop'])} (약 2 ATR)\n"
        f"- 추적 손절 기준: {fmt(risk['trailing_stop'])}\n"
        f"- 최근 지지/저항: {fmt(risk['recent_support'])} / {fmt(risk['recent_resistance'])}\n"
        f"- 볼린저 하단/상단: {fmt(risk['bollinger_lower'])} / {fmt(risk['bollinger_upper'])}\n\n"
        "가짜 돌파 점검\n"
        f"{breakout}\n\n"
        "킬스위치 점검\n"
        f"{kill}\n\n"
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


def format_date(value: Any) -> str:
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def max_period(period: str, minimum: str) -> str:
    order = {
        "1mo": 1,
        "3mo": 3,
        "6mo": 6,
        "1y": 12,
        "2y": 24,
        "5y": 60,
        "10y": 120,
        "max": 9999,
    }
    return period if order.get(period, 12) >= order.get(minimum, 12) else minimum
