"""
backtest_engine.py — OOP backtesting framework.

Architecture
────────────
  BacktestConfig   — strategy parameters
  GoldenCrossStrategy — EMA50/SMA200 + ATR trailing stop state machine
  Backtester       — drives the strategy, handles execution, builds equity curve
  BacktestMetrics  — computes all performance statistics from a completed run

Design principles
─────────────────
* Indicators (EMA, SMA, ATR, RSI) are computed in `Backtester.prepare()` and
  stored on the DataFrame.  The strategy's `next()` only reads pre-computed
  arrays — no forward-looking computation.
* Execution model: signal fires at EOD on day T → order executes at close of
  day T+1 ("next_close").  This provides a 1-day execution lag.
* ATR-based dynamic slippage:
    buy:  P_exec = P_t * (1 + α_fee + β * ATR14 / P_t)
    sell: P_exec = P_t * (1 - α_fee - β * ATR14 / P_t)
  where α_fee = commission_bps/10000, β = slippage_beta.
* Dividends/splits: handled implicitly by using adjusted close prices
  (yfinance auto_adjust=True).  The price series is already a total-return
  series, so no explicit reinvestment loop is required.
* Lookahead bias: all signal checks read index i; orders execute at index i+1.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    atr_multiplier: float = 3.5
    commission_bps: float = 5.0       # fixed cost per side (bps)
    slippage_beta: float = 0.1        # ATR-impact coefficient β
    initial_capital: float = 10_000_000.0
    reentry_mode: str = "next_cross"  # "next_cross" | "regime_active"


# ---------------------------------------------------------------------------
# Abstract strategy interface
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """
    Base class for all strategies.  Subclasses implement `init()` to cache
    indicator arrays and `next(i)` to emit buy/sell signals.
    """

    def __init__(self) -> None:
        self._data: pd.DataFrame | None = None
        self._pending: str | None = None          # "BUY" | "SELL" | None
        self._exit_reason: str = ""

    def _attach(self, data: pd.DataFrame) -> None:
        self._data = data
        self.init()

    @abstractmethod
    def init(self) -> None:
        """Cache indicator columns as numpy arrays for fast access."""

    @abstractmethod
    def next(self, i: int) -> None:
        """
        Called once per bar (index i).  Should call self._buy() or self._sell()
        to queue an order that executes on bar i+1.
        """

    def _buy(self) -> None:
        if self._pending is None:
            self._pending = "BUY"

    def _sell(self, reason: str = "Signal") -> None:
        if self._pending != "SELL":
            self._pending = "SELL"
            self._exit_reason = reason


# ---------------------------------------------------------------------------
# Concrete strategy: EMA50/SMA200 Golden Cross + ATR Trailing Stop
# ---------------------------------------------------------------------------

class GoldenCrossStrategy(Strategy):
    """
    Entry: EMA50 crosses above SMA200 (golden cross) on bar i → buy at bar i+1.
    Exit 1: EMA50 crosses below SMA200 (death cross) on bar i → sell at bar i+1.
    Exit 2: Close[i] < trailing_stop[i] → sell at bar i+1.
    Trailing stop: max(Close since entry) − ATR14 × multiplier.
    Reentry after ATR stop-out depends on reentry_mode.
    """

    def __init__(self, config: BacktestConfig) -> None:
        super().__init__()
        self.config = config
        # These are set by init()
        self._bull: np.ndarray = np.array([])
        self._close: np.ndarray = np.array([])
        self._atr: np.ndarray = np.array([])
        self._highest_close: float = 0.0
        self._was_stopped: bool = False
        self._in_position: bool = False

    def init(self) -> None:
        df = self._data
        assert df is not None
        self._close = df["Close"].to_numpy(dtype=float)
        self._atr = df["ATR_14"].to_numpy(dtype=float)
        self._bull = df["Bull_Regime"].to_numpy(dtype=bool)

    def reset(self) -> None:
        """Reset mutable state for repeated runs."""
        self._highest_close = 0.0
        self._was_stopped = False
        self._in_position = False
        self._pending = None
        self._exit_reason = ""

    def on_entry(self, close: float) -> None:
        self._in_position = True
        self._highest_close = close

    def on_exit(self) -> None:
        self._was_stopped = (self._exit_reason == "ATR Stop")
        self._in_position = False
        self._highest_close = 0.0

    def update_trailing(self, close: float) -> None:
        if self._in_position:
            self._highest_close = max(self._highest_close, close)

    def trailing_stop(self, i: int) -> float:
        atr_i = self._atr[i]
        if np.isnan(atr_i):
            return float("nan")
        return self._highest_close - self.config.atr_multiplier * atr_i

    def next(self, i: int) -> None:
        if i < 1:
            return
        close_i = self._close[i]
        bull_i = self._bull[i]
        bull_prev = self._bull[i - 1]

        if self._in_position:
            stop = self.trailing_stop(i)
            if not np.isnan(stop) and close_i < stop:
                self._sell("ATR Stop")
            elif not bull_i and bull_prev:  # death cross
                self._sell("Death Cross")
        else:
            if bull_i and not bull_prev:  # golden cross
                self._buy()
            elif (
                self.config.reentry_mode == "regime_active"
                and self._was_stopped
                and bull_i
            ):
                self._buy()
                self._was_stopped = False


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    entry_date: Any
    entry_price: float       # net (after buy cost)
    entry_close: float       # raw close on entry day
    exit_date: Any
    exit_price: float        # net (after sell cost)
    exit_close: float        # raw close on exit day
    gross_return: float      # exit_close / entry_close - 1
    net_return: float        # exit_price / entry_price - 1 (cost-adjusted)
    holding_days: int
    exit_reason: str


class Backtester:
    """
    Drives a Strategy over a prepared DataFrame, handling order execution,
    ATR-based dynamic slippage, and equity curve construction.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------------
    # Execution price helpers
    # ------------------------------------------------------------------

    def _exec_buy(self, close: float, atr: float) -> float:
        """P_exec_buy = P_t * (1 + α_fee + β * ATR14/P_t)"""
        if close <= 0:
            return close
        alpha = self.config.commission_bps / 10_000
        beta = self.config.slippage_beta
        slippage = beta * (atr / close) if close > 0 and not np.isnan(atr) else 0.0
        return close * (1 + alpha + slippage)

    def _exec_sell(self, close: float, atr: float) -> float:
        """P_exec_sell = P_t * (1 - α_fee - β * ATR14/P_t)"""
        if close <= 0:
            return close
        alpha = self.config.commission_bps / 10_000
        beta = self.config.slippage_beta
        slippage = beta * (atr / close) if close > 0 and not np.isnan(atr) else 0.0
        return close * (1 - alpha - slippage)

    # ------------------------------------------------------------------
    # Prepare indicators
    # ------------------------------------------------------------------

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators needed by the strategy and return an
        annotated copy.  Called once before run().
        """
        out = df.copy()
        close = out["Close"]

        out["EMA_50"] = close.ewm(span=50, adjust=False).mean()
        out["SMA_200"] = close.rolling(200).mean()
        out["ATR_14"] = _atr(out["High"], out["Low"], close)
        out["RSI_14"] = _rsi(close)

        out["Bull_Regime"] = out["EMA_50"] > out["SMA_200"]
        prev_bull = out["Bull_Regime"].shift(1, fill_value=False).astype(bool)
        out["Golden_Cross"] = out["Bull_Regime"] & ~prev_bull
        out["Death_Cross"] = ~out["Bull_Regime"] & prev_bull

        # Vectorized ATR trailing stop (for display/chart purposes only)
        regime_group = out["Bull_Regime"].ne(out["Bull_Regime"].shift()).cumsum()
        regime_high = close.where(out["Bull_Regime"]).groupby(regime_group).cummax()
        out["ATR_Trailing_Stop"] = regime_high - self.config.atr_multiplier * out["ATR_14"]

        out["Asset_Return"] = close.pct_change().fillna(0.0)
        out["Buy_Hold_Equity"] = self.config.initial_capital * (1 + out["Asset_Return"]).cumprod()
        out["Buy_Hold_Drawdown"] = out["Buy_Hold_Equity"] / out["Buy_Hold_Equity"].cummax() - 1.0

        return out

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self, df: pd.DataFrame, strategy: Strategy | None = None
    ) -> tuple[pd.DataFrame, list[TradeRecord]]:
        """
        Run the strategy over *df* (must be pre-processed by prepare()).
        Returns (annotated_df, trade_list).

        Execution model
        ───────────────
        * Signal fires at EOD day i (based on close[i]).
        * Order executes at close of day i+1 (1-day lag).
        * ATR slippage applied at execution day.
        """
        if strategy is None:
            strategy = GoldenCrossStrategy(self.config)
        strategy._attach(df)
        if isinstance(strategy, GoldenCrossStrategy):
            strategy.reset()

        n = len(df)
        closes = df["Close"].to_numpy(dtype=float)
        atrs = df["ATR_14"].to_numpy(dtype=float)
        index = df.index

        cfg = self.config
        ev_equity = np.full(n, cfg.initial_capital, dtype=float)
        ev_invested = np.zeros(n, dtype=bool)

        state: str = "OUT"   # "OUT" | "IN"
        cash: float = cfg.initial_capital
        shares: float = 0.0
        entry_i: int = -1
        entry_net: float = 0.0
        entry_close_price: float = 0.0
        trades: list[TradeRecord] = []

        for i in range(n):
            c = closes[i]
            atr_i = atrs[i] if not np.isnan(atrs[i]) else 0.0

            # --- Execute pending order from yesterday ---
            if strategy._pending == "BUY" and state == "OUT":
                net_buy = self._exec_buy(c, atr_i)
                shares = cash / net_buy
                cash = 0.0
                state = "IN"
                entry_i = i
                entry_net = net_buy
                entry_close_price = c
                strategy._pending = None
                if isinstance(strategy, GoldenCrossStrategy):
                    strategy.on_entry(c)

            elif strategy._pending == "SELL" and state == "IN":
                net_sell = self._exec_sell(c, atr_i)
                gross_ret = c / entry_close_price - 1.0
                net_ret = net_sell / entry_net - 1.0
                cash = shares * net_sell
                h_days = (index[i] - index[entry_i]).days if hasattr(index[i], "__sub__") else 0
                trades.append(TradeRecord(
                    entry_date=index[entry_i].date() if hasattr(index[entry_i], "date") else index[entry_i],
                    entry_price=round(entry_net, 4),
                    entry_close=round(entry_close_price, 4),
                    exit_date=index[i].date() if hasattr(index[i], "date") else index[i],
                    exit_price=round(net_sell, 4),
                    exit_close=round(c, 4),
                    gross_return=round(gross_ret, 6),
                    net_return=round(net_ret, 6),
                    holding_days=h_days,
                    exit_reason=strategy._exit_reason,
                ))
                shares = 0.0
                state = "OUT"
                if isinstance(strategy, GoldenCrossStrategy):
                    strategy.on_exit()
                strategy._pending = None

            # --- Mark-to-market ---
            if state == "IN":
                ev_equity[i] = shares * c
                ev_invested[i] = True
                if isinstance(strategy, GoldenCrossStrategy):
                    strategy.update_trailing(c)
            else:
                ev_equity[i] = cash

            # --- Generate signal for next bar ---
            if i < n - 1:
                strategy.next(i)

        # --- Close open position at end of data ---
        if state == "IN" and entry_i >= 0:
            last_i = n - 1
            c = closes[last_i]
            atr_i = atrs[last_i] if not np.isnan(atrs[last_i]) else 0.0
            net_sell = self._exec_sell(c, atr_i)
            gross_ret = c / entry_close_price - 1.0
            net_ret = net_sell / entry_net - 1.0
            cash = shares * net_sell
            h_days = (index[last_i] - index[entry_i]).days if hasattr(index[last_i], "__sub__") else 0
            trades.append(TradeRecord(
                entry_date=index[entry_i].date() if hasattr(index[entry_i], "date") else index[entry_i],
                entry_price=round(entry_net, 4),
                entry_close=round(entry_close_price, 4),
                exit_date=index[last_i].date() if hasattr(index[last_i], "date") else index[last_i],
                exit_price=round(net_sell, 4),
                exit_close=round(c, 4),
                gross_return=round(gross_ret, 6),
                net_return=round(net_ret, 6),
                holding_days=h_days,
                exit_reason="End of Data",
            ))
            ev_equity[last_i] = cash

        # --- Annotate df ---
        out = df.copy()
        out["Event_Equity"] = ev_equity
        out["Event_Invested"] = ev_invested
        out["Event_Drawdown"] = pd.Series(ev_equity, index=df.index) / pd.Series(ev_equity, index=df.index).cummax() - 1.0
        # Alias to standard names for chart_generator compatibility
        out["Strategy_Equity"] = out["Event_Equity"]
        out["Strategy_Drawdown"] = out["Event_Drawdown"]
        out["Invested"] = out["Event_Invested"]
        # Strategy_Return derived from equity changes (correctly handles cost at boundaries)
        eq_series = pd.Series(ev_equity, index=df.index)
        out["Strategy_Return"] = eq_series.pct_change().fillna(0.0)

        return out, trades


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    df: pd.DataFrame,
    trades: list[TradeRecord],
    risk_free_rate: float = 0.03,
) -> dict[str, float]:
    """
    Compute comprehensive backtest performance metrics from an annotated df
    (must contain Strategy_Equity, Strategy_Return, Strategy_Drawdown,
    Buy_Hold_Equity, Buy_Hold_Drawdown, Invested).
    """
    m: dict[str, float] = {}

    eq = df["Strategy_Equity"]
    bh = df["Buy_Hold_Equity"]
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)

    cum_strat = float(eq.iloc[-1] / eq.iloc[0])
    cum_bh = float(bh.iloc[-1] / bh.iloc[0])

    m["years"] = years
    m["cagr_strategy"] = cum_strat ** (1.0 / years) - 1.0
    m["cagr_buy_hold"] = cum_bh ** (1.0 / years) - 1.0
    m["cumulative_strategy"] = cum_strat
    m["cumulative_buy_hold"] = cum_bh
    m["mdd_strategy"] = float(df["Strategy_Drawdown"].min())
    m["mdd_buy_hold"] = float(df["Buy_Hold_Drawdown"].min())
    m["market_exposure"] = float(df["Invested"].mean())

    # --- Risk-adjusted returns ---
    strat_ret = df["Strategy_Return"]
    asset_ret = df["Asset_Return"]
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = strat_ret - daily_rf
    excess_mean = float(excess.mean())
    excess_std = float(excess.std(ddof=1))

    m["annual_return"] = float(asset_ret.mean()) * TRADING_DAYS
    m["annual_volatility"] = float(asset_ret.std(ddof=0)) * sqrt(TRADING_DAYS)

    if excess_std > 0:
        m["sharpe_ratio"] = excess_mean / excess_std * sqrt(TRADING_DAYS)
    else:
        m["sharpe_ratio"] = float("inf") if excess_mean > 0 else float("nan")

    downside = excess[excess < 0]
    if len(downside) > 0:
        ds_std = float(sqrt((downside ** 2).mean()))
        m["sortino_ratio"] = excess_mean / ds_std * sqrt(TRADING_DAYS) if ds_std > 0 else float("inf")
    else:
        m["sortino_ratio"] = float("inf")

    # Calmar ratio
    if m["mdd_strategy"] < 0:
        m["calmar_ratio"] = m["cagr_strategy"] / abs(m["mdd_strategy"])
    else:
        m["calmar_ratio"] = float("inf")

    # Exposure-adjusted CAGR: what CAGR would be if always invested
    exp = m["market_exposure"]
    if 0 < exp < 1:
        # (1 + cagr)^(1/exposure) - 1
        m["exposure_adj_cagr"] = (1 + m["cagr_strategy"]) ** (1.0 / exp) - 1.0
    else:
        m["exposure_adj_cagr"] = m["cagr_strategy"]

    # Kelly
    vol2 = m["annual_volatility"] ** 2
    if vol2 > 0:
        kelly = (m["annual_return"] - risk_free_rate) / vol2
    else:
        kelly = 0.0
    m["kelly_fraction"] = kelly
    m["suggested_fractional_kelly"] = max(0.0, min(kelly * 0.5, 1.0))

    # --- Trade-level metrics ---
    n = len(trades)
    m["num_trades"] = float(n)
    if n == 0:
        for k in ("win_rate", "profit_factor", "avg_holding_days", "best_trade",
                  "worst_trade", "turnover"):
            m[k] = float("nan")
    else:
        net_rets = np.array([t.net_return for t in trades])
        wins = net_rets[net_rets > 0]
        losses = net_rets[net_rets <= 0]
        gross_profit = float(wins.sum())
        gross_loss = float(losses.sum())

        m["win_rate"] = len(wins) / n
        if gross_loss == 0:
            m["profit_factor"] = float("inf")
        elif gross_profit == 0:
            m["profit_factor"] = 0.0
        else:
            m["profit_factor"] = gross_profit / abs(gross_loss)

        m["avg_holding_days"] = float(np.mean([t.holding_days for t in trades]))
        m["best_trade"] = float(net_rets.max())
        m["worst_trade"] = float(net_rets.min())
        # Turnover: trades per year (round trips)
        m["turnover"] = n / years

    # --- Rolling metrics (last valid value) ---
    m["rolling_cagr_3y"] = float(_rolling_cagr(eq, window_years=3).iloc[-1])
    m["rolling_cagr_5y"] = float(_rolling_cagr(eq, window_years=5).iloc[-1])
    m["rolling_mdd_3y"] = float(_rolling_mdd(df["Strategy_Drawdown"], window_years=3).iloc[-1])

    return m


def bootstrap_ci(
    trades: list[TradeRecord],
    n_boot: int = 2000,
    ci_pct: float = 97.5,
) -> dict[str, float]:
    """
    BCa-style bootstrap confidence intervals (lower bound at (100-ci_pct)th pct).
    Uses trade-level net returns, not daily returns.
    """
    result: dict[str, float] = {
        "sharpe_ci_lower": float("nan"),
        "pf_ci_lower": float("nan"),
    }
    if len(trades) < 10:
        return result

    returns = np.array([t.net_return for t in trades])
    n = len(returns)
    rng = np.random.default_rng(42)
    boot_sharpes: list[float] = []
    boot_pfs: list[float] = []

    for _ in range(n_boot):
        sample = rng.choice(returns, size=n, replace=True)
        std = sample.std()
        if std > 0:
            boot_sharpes.append(float(sample.mean() / std * sqrt(TRADING_DAYS)))
        wins = sample[sample > 0].sum()
        losses = abs(sample[sample <= 0].sum())
        if losses > 0:
            boot_pfs.append(float(wins / losses))

    lower_pct = 100.0 - ci_pct
    if boot_sharpes:
        result["sharpe_ci_lower"] = float(np.percentile(boot_sharpes, lower_pct))
    if boot_pfs:
        result["pf_ci_lower"] = float(np.percentile(boot_pfs, lower_pct))
    return result


def walk_forward_analysis(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    atr_candidates: list[float] | None = None,
    config_base: BacktestConfig | None = None,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward analysis.

    For each fold:
      1. Select best ATR multiplier on the in-sample window (maximize CAGR).
      2. Evaluate on the out-of-sample window.

    Returns a DataFrame with one row per fold.
    """
    if atr_candidates is None:
        atr_candidates = [3.0, 3.5, 4.0]
    if config_base is None:
        config_base = BacktestConfig()

    n = len(df)
    records: list[dict] = []

    for fold in range(n_splits):
        total = int(n * (fold + 1) / n_splits)
        train_end = int(total * train_ratio)

        train_df = df.iloc[:train_end].copy()
        oos_df = df.iloc[train_end:total].copy()

        if len(train_df) < 250 or len(oos_df) < 50:
            continue

        # Find best ATR on in-sample
        best_atr = atr_candidates[0]
        best_cagr = float("-inf")
        for atr_m in atr_candidates:
            try:
                cfg = BacktestConfig(
                    atr_multiplier=atr_m,
                    commission_bps=config_base.commission_bps,
                    slippage_beta=config_base.slippage_beta,
                    initial_capital=config_base.initial_capital,
                )
                bt = Backtester(cfg)
                prepared = bt.prepare(train_df)
                prepared = prepared.dropna(subset=["EMA_50", "SMA_200", "ATR_14"])
                bt_df, bt_trades = bt.run(prepared)
                m = compute_metrics(bt_df, bt_trades)
                if m["cagr_strategy"] > best_cagr:
                    best_cagr = m["cagr_strategy"]
                    best_atr = atr_m
            except Exception:
                pass

        # Evaluate on OOS
        try:
            cfg_oos = BacktestConfig(
                atr_multiplier=best_atr,
                commission_bps=config_base.commission_bps,
                slippage_beta=config_base.slippage_beta,
                initial_capital=config_base.initial_capital,
            )
            bt_oos = Backtester(cfg_oos)
            prepared_oos = bt_oos.prepare(oos_df)
            prepared_oos = prepared_oos.dropna(subset=["EMA_50", "SMA_200", "ATR_14"])
            bt_oos_df, oos_trades = bt_oos.run(prepared_oos)
            oos_m = compute_metrics(bt_oos_df, oos_trades)
            records.append({
                "fold": fold + 1,
                "is_end": str(train_df.index[-1].date()) if hasattr(train_df.index[-1], "date") else str(train_df.index[-1]),
                "oos_start": str(oos_df.index[0].date()) if hasattr(oos_df.index[0], "date") else str(oos_df.index[0]),
                "oos_end": str(oos_df.index[-1].date()) if hasattr(oos_df.index[-1], "date") else str(oos_df.index[-1]),
                "best_atr": best_atr,
                "is_cagr": round(best_cagr, 4),
                "oos_cagr": round(oos_m["cagr_strategy"], 4),
                "oos_mdd": round(oos_m["mdd_strategy"], 4),
                "oos_sharpe": round(oos_m.get("sharpe_ratio", float("nan")), 3),
                "oos_trades": int(oos_m["num_trades"]),
            })
        except Exception:
            pass

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry score backtesting
# ---------------------------------------------------------------------------

def backtest_entry_score(
    df: pd.DataFrame,
    entry_score_fn: Any,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute historical entry scores and bucket forward returns.

    Parameters
    ──────────
    df              : prepared DataFrame (must have standard indicator columns)
    entry_score_fn  : callable(row, metrics_dict) -> float (entry_assessment wrapper)
    horizons        : list of calendar days to measure forward return

    Returns a DataFrame with rows = score buckets, columns = horizon stats.
    """
    if horizons is None:
        horizons = [21, 63, 126, 252]  # ~1M, 3M, 6M, 12M

    BUCKETS = [(0, 40, "0–40"), (40, 60, "40–60"), (60, 75, "60–75"), (75, 101, "75–100")]
    n = len(df)
    closes = df["Close"].to_numpy(dtype=float)
    index = df.index

    scores: list[tuple[int, float]] = []
    for i in range(200, n):
        try:
            row = df.iloc[i]
            # Quick score proxy using Bull_Regime and RSI only (avoids full metrics recompute)
            score = _quick_score(row)
            scores.append((i, score))
        except Exception:
            pass

    records: list[dict] = []
    for lo, hi, label in BUCKETS:
        bucket_rows = [(i, s) for (i, s) in scores if lo <= s < hi]
        for h in horizons:
            fwd_returns: list[float] = []
            for (i, _s) in bucket_rows:
                j = i + h
                if j < n:
                    fwd_returns.append(closes[j] / closes[i] - 1.0)
            if not fwd_returns:
                records.append({"bucket": label, "horizon_days": h,
                                 "sample_count": 0, "avg_fwd_return": float("nan"),
                                 "median_fwd_return": float("nan"), "win_rate": float("nan")})
            else:
                arr = np.array(fwd_returns)
                records.append({
                    "bucket": label,
                    "horizon_days": h,
                    "sample_count": len(arr),
                    "avg_fwd_return": float(arr.mean()),
                    "median_fwd_return": float(np.median(arr)),
                    "win_rate": float((arr > 0).mean()),
                })

    return pd.DataFrame(records)


def _quick_score(row: pd.Series) -> float:
    """Simplified entry score based on available indicator columns."""
    bull = bool(row.get("Bull_Regime", False))
    rsi = float(row.get("RSI_14", 50.0))
    close = float(row.get("Close", 1.0))
    ema50 = float(row.get("EMA_50", close))
    sma200 = float(row.get("SMA_200", close))

    trend_gap = ema50 / sma200 - 1.0 if sma200 else 0.0
    trend_score = min(100.0, 50.0 + trend_gap * 1000.0) if bull else max(0.0, 40.0 + trend_gap * 800.0)

    if rsi >= 70:
        rsi_score = max(20.0, 100.0 - (rsi - 70.0) * 4.0)
    elif rsi >= 55:
        rsi_score = 90.0
    elif rsi >= 45:
        rsi_score = 100.0
    elif rsi >= 30:
        rsi_score = 75.0
    else:
        rsi_score = 40.0

    score = 0.5 * trend_score + 0.5 * rsi_score
    if not bull:
        score = min(score, 45.0)
    return round(max(0.0, min(100.0, score)), 1)


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------

def _rolling_cagr(equity: pd.Series, window_years: float = 3.0) -> pd.Series:
    window = int(window_years * TRADING_DAYS)
    ratio = equity / equity.shift(window)
    return (ratio ** (1.0 / window_years) - 1.0).fillna(float("nan"))


def _rolling_mdd(drawdown: pd.Series, window_years: float = 3.0) -> pd.Series:
    window = int(window_years * TRADING_DAYS)
    return drawdown.rolling(window).min().fillna(float("nan"))


# ---------------------------------------------------------------------------
# Indicator helpers (mirrored from stock_analyzer for independence)
# ---------------------------------------------------------------------------

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)
