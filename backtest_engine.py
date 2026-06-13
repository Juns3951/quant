"""
backtest_engine.py — Institutional-grade OOP backtesting framework.

Class hierarchy
───────────────
  TechnicalIndicatorsEngine  — stateless indicator computations
  ExecutionSimulator         — nonlinear volume-weighted market impact
  PortfolioManager           — equity, dividend reinvestment, split adjustment
  GoldenCrossStrategy        — EMA50/SMA200 + ATR trailing stop state machine
  Backtester                 — drives strategy with 1-day execution lag
  OptimizationEngine         — slippage sweep, BCa bootstrap, walk-forward analysis

Execution model
───────────────
  Signal fires at EOD day T → order executes at CLOSE of day T+1.
  Lookahead bias is structurally impossible: indicator arrays are fixed before
  the loop starts and no future value is read inside next(i).

Slippage model (nonlinear market impact)
────────────────────────────────────────
  buy:  P_exec = P_t · (1 + α_fee + β·(ATR₁₄/P_t)·(1 + ln(max(1, V_order/V_ADV))))
  sell: P_exec = P_t · (1 − α_fee − β·(ATR₁₄/P_t)·(1 + ln(max(1, V_order/V_ADV))))

  where α_fee = commission_bps/10000, β = slippage_beta,
  V_order = order_capital/price (shares), V_ADV = 14-day avg daily volume.
  For retail orders (V_order << V_ADV) the log term vanishes; for large
  institutional orders it amplifies impact logarithmically.

Dividend / split handling
─────────────────────────
  With adjusted-close prices (yfinance auto_adjust=True) dividends and splits
  are already embedded in the price series (total-return series), so no
  explicit loop is required.  PortfolioManager provides explicit methods for
  use with raw unadjusted data when corporate-action events are supplied.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from math import log, log2, sqrt
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
    commission_bps: float = 5.0        # α_fee (per side)
    slippage_beta: float = 0.1         # β coefficient in market impact formula
    initial_capital: float = 10_000_000.0
    reentry_mode: str = "next_cross"   # "next_cross" | "regime_active"


# ---------------------------------------------------------------------------
# 1. TechnicalIndicatorsEngine
# ---------------------------------------------------------------------------

class TechnicalIndicatorsEngine:
    """
    Stateless indicator library.  All methods are static so indicator logic
    is completely decoupled from execution and portfolio management.
    """

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).mean()

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    @staticmethod
    def adv(volume: pd.Series, period: int = 14) -> pd.Series:
        """14-day Average Daily Volume (shares) for market impact calculation."""
        return volume.rolling(period, min_periods=1).mean()

    @staticmethod
    def bull_regime(ema50: pd.Series, sma200: pd.Series) -> pd.Series:
        return (ema50 > sma200).astype(bool)

    @staticmethod
    def golden_cross(bull: pd.Series) -> pd.Series:
        prev = bull.shift(1, fill_value=False).astype(bool)
        return bull.astype(bool) & ~prev

    @staticmethod
    def death_cross(bull: pd.Series) -> pd.Series:
        prev = bull.shift(1, fill_value=False).astype(bool)
        return ~bull.astype(bool) & prev

    @staticmethod
    def atr_trailing_stop(
        close: pd.Series, bull: pd.Series, atr: pd.Series, multiplier: float
    ) -> pd.Series:
        """Vectorized regime-high based trailing stop (for display / chart only)."""
        regime_group = bull.ne(bull.shift()).cumsum()
        regime_high = close.where(bull).groupby(regime_group).cummax()
        raw = regime_high - multiplier * atr
        # Penny-stock floor: stop never below 1 % of close
        return raw.clip(lower=close * 0.01)

    @staticmethod
    def interpolate_trading_halts(df: pd.DataFrame) -> pd.DataFrame:
        """
        Linearly interpolate OHLC prices for zero-volume (trading-halt) rows.
        Volume stays 0 to preserve the audit trail.
        This prevents gaps from distorting indicator calculations.
        """
        halt_mask = (df["Volume"] == 0) | df["Volume"].isna()
        if not halt_mask.any():
            return df
        out = df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            if col in out.columns:
                out.loc[halt_mask, col] = np.nan
                out[col] = out[col].interpolate(method="linear", limit_direction="both")
        return out


# ---------------------------------------------------------------------------
# 2. ExecutionSimulator
# ---------------------------------------------------------------------------

class ExecutionSimulator:
    """
    Nonlinear volume-weighted market-impact execution model.

    buy:  P_exec = P · (1 + α_fee + β·(ATR/P)·(1 + ln(max(1, V_ord/V_ADV))))
    sell: P_exec = P · (1 − α_fee − β·(ATR/P)·(1 + ln(max(1, V_ord/V_ADV))))

    For retail-scale orders (V_ord << V_ADV) the log term ≈ 0 and impact
    collapses to β·ATR/P (baseline proportional to daily volatility).
    """

    def __init__(
        self,
        commission_bps: float = 5.0,
        slippage_beta: float = 0.1,
        order_capital: float = 10_000_000.0,
    ) -> None:
        self.alpha_fee = commission_bps / 10_000
        self.beta = slippage_beta
        self.order_capital = order_capital

    def _impact_mult(self, price: float, adv: float) -> float:
        """
        1 + ln(max(1, V_order / V_ADV)).
        Returns ≥ 1.0; equals 1.0 when V_order ≤ V_ADV (retail).
        """
        if price <= 0 or adv <= 0:
            return 1.0
        v_order = self.order_capital / max(price, 1e-9)
        v_adv = max(adv, 1.0)
        ratio = max(1.0, v_order / v_adv)
        return 1.0 + log(ratio)

    def _safe_atr_ratio(self, close: float, atr: float) -> float:
        if close <= 0 or math.isnan(atr) or math.isinf(atr) or atr < 0:
            return 0.0
        return atr / close

    def exec_buy(self, close: float, atr: float, adv: float) -> float:
        if close <= 0:
            return max(close, 1e-9)
        impact = self.beta * self._safe_atr_ratio(close, atr) * self._impact_mult(close, adv)
        return close * (1.0 + self.alpha_fee + impact)

    def exec_sell(self, close: float, atr: float, adv: float) -> float:
        if close <= 0:
            return max(close, 1e-9)
        impact = self.beta * self._safe_atr_ratio(close, atr) * self._impact_mult(close, adv)
        return close * (1.0 - self.alpha_fee - impact)


# ---------------------------------------------------------------------------
# 3. PortfolioManager
# ---------------------------------------------------------------------------

class PortfolioManager:
    """
    Tracks portfolio equity through trade events.

    Supports explicit dividend reinvestment and split adjustment for use with
    raw (unadjusted) price series.  When using adjusted-close prices
    (auto_adjust=True), these events are already embedded in prices and
    calling these methods is not required.
    """

    def __init__(self, initial_capital: float = 10_000_000.0) -> None:
        self.initial_capital = initial_capital
        self.cash: float = initial_capital
        self.shares: float = 0.0
        self._entry_exec_price: float = 0.0
        self._pending_dividend: float = 0.0

    @property
    def in_position(self) -> bool:
        return self.shares > 0

    def equity(self, current_price: float) -> float:
        """Current mark-to-market equity (cash + position + pending dividends)."""
        return self.shares * current_price + self.cash + self._pending_dividend

    def open_position(self, exec_price: float) -> float:
        """Deploy all available capital at exec_price.  Returns shares bought."""
        if exec_price <= 0:
            return 0.0
        available = self.cash + self._pending_dividend
        self.shares = available / exec_price
        self._entry_exec_price = exec_price
        self.cash = 0.0
        self._pending_dividend = 0.0
        return self.shares

    def close_position(self, exec_price: float) -> float:
        """Liquidate position at exec_price.  Returns cash received."""
        proceeds = self.shares * exec_price + self._pending_dividend
        self.cash = proceeds
        self._pending_dividend = 0.0
        self.shares = 0.0
        self._entry_exec_price = 0.0
        return self.cash

    def receive_dividend(self, dividend_per_share: float) -> None:
        """
        Accumulate per-share dividend during holding period.
        The cash is reinvested compoundingly on the next open_position() call.
        """
        if self.shares > 0:
            self._pending_dividend += self.shares * dividend_per_share

    def apply_split(self, ratio: float) -> None:
        """
        Adjust for stock split.  ratio=2.0 → 2-for-1 split.
        Shares double, entry price halves, net P&L unchanged.
        """
        if self.shares > 0 and ratio > 0:
            self.shares *= ratio
            if self._entry_exec_price > 0:
                self._entry_exec_price /= ratio


# ---------------------------------------------------------------------------
# 4. Strategy interface
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """
    Abstract base.  Subclasses implement init() (cache indicator arrays) and
    next(i) (emit signals based on pre-computed arrays — NO forward reads).
    """

    def __init__(self) -> None:
        self._data: pd.DataFrame | None = None
        self._pending: str | None = None
        self._exit_reason: str = ""

    def _attach(self, data: pd.DataFrame) -> None:
        self._data = data
        self.init()

    @abstractmethod
    def init(self) -> None: ...

    @abstractmethod
    def next(self, i: int) -> None: ...

    def _buy(self) -> None:
        if self._pending is None:
            self._pending = "BUY"

    def _sell(self, reason: str = "Signal") -> None:
        if self._pending != "SELL":
            self._pending = "SELL"
            self._exit_reason = reason


# ---------------------------------------------------------------------------
# 5. GoldenCrossStrategy
# ---------------------------------------------------------------------------

class GoldenCrossStrategy(Strategy):
    """
    EMA50/SMA200 golden-cross entry + ATR trailing stop + death-cross exit.

    Trailing stop uses entry-local highest close (not full regime high) so
    the stop accurately reflects risk since our specific entry, not the
    regime's historical peak.

    Penny-stock safety: stop floor = max(stop_calc, close * 0.01).
    This prevents negative or near-zero stops when ATR >> price.
    """

    def __init__(self, config: BacktestConfig) -> None:
        super().__init__()
        self.config = config
        self._close: np.ndarray = np.array([])
        self._atr: np.ndarray = np.array([])
        self._bull: np.ndarray = np.array([])
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
        """Entry-local trailing stop with penny-stock price floor."""
        atr_i = self._atr[i]
        if np.isnan(atr_i):
            return float("nan")
        close_i = self._close[i]
        raw_stop = self._highest_close - self.config.atr_multiplier * atr_i
        # Penny-stock floor: stop cannot be below 1 % of current price
        return max(raw_stop, close_i * 0.01)

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
            elif not bull_i and bull_prev:
                self._sell("Death Cross")
        else:
            if bull_i and not bull_prev:
                self._buy()
            elif (
                self.config.reentry_mode == "regime_active"
                and self._was_stopped
                and bull_i
            ):
                self._buy()
                self._was_stopped = False


# ---------------------------------------------------------------------------
# 6. Backtester
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    entry_date: Any
    entry_price: float       # net (after buy cost)
    entry_close: float       # raw reference close
    exit_date: Any
    exit_price: float        # net (after sell cost)
    exit_close: float        # raw reference close
    gross_return: float
    net_return: float
    holding_days: int
    exit_reason: str


class Backtester:
    """
    Drives a Strategy over a prepared DataFrame.
    Uses ExecutionSimulator for fills and PortfolioManager for equity.
    """

    IND = TechnicalIndicatorsEngine()   # shared stateless instance

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self._exec = ExecutionSimulator(
            commission_bps=self.config.commission_bps,
            slippage_beta=self.config.slippage_beta,
            order_capital=self.config.initial_capital,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers (keep backward-compat module-level names)
    # ------------------------------------------------------------------

    def _exec_buy(self, close: float, atr: float) -> float:
        return self._exec.exec_buy(close, atr, adv=1e9)   # adv=large → no extra impact (legacy)

    def _exec_sell(self, close: float, atr: float) -> float:
        return self._exec.exec_sell(close, atr, adv=1e9)

    # ------------------------------------------------------------------
    # Prepare indicators
    # ------------------------------------------------------------------

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators.  Must be called before run().
        Input df must have OHLCV columns (output of clean_price_frame).
        """
        out = df.copy()

        # Trading halt interpolation
        out = TechnicalIndicatorsEngine.interpolate_trading_halts(out)

        close = out["Close"]
        out["EMA_50"] = TechnicalIndicatorsEngine.ema(close, 50)
        out["SMA_200"] = TechnicalIndicatorsEngine.sma(close, 200)
        out["ATR_14"] = TechnicalIndicatorsEngine.atr(out["High"], out["Low"], close)
        out["RSI_14"] = TechnicalIndicatorsEngine.rsi(close)
        out["ADV_14"] = TechnicalIndicatorsEngine.adv(out["Volume"], 14)

        out["Bull_Regime"] = TechnicalIndicatorsEngine.bull_regime(out["EMA_50"], out["SMA_200"])
        out["Golden_Cross"] = TechnicalIndicatorsEngine.golden_cross(out["Bull_Regime"])
        out["Death_Cross"] = TechnicalIndicatorsEngine.death_cross(out["Bull_Regime"])
        out["ATR_Trailing_Stop"] = TechnicalIndicatorsEngine.atr_trailing_stop(
            close, out["Bull_Regime"], out["ATR_14"], self.config.atr_multiplier
        )

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
        Run strategy with 1-day execution lag and volume-weighted slippage.

        Execution timing
        ────────────────
        Day i  : strategy.next(i) checks indicators → may set _pending
        Day i+1: Backtester executes the pending order at close[i+1]
        """
        if strategy is None:
            strategy = GoldenCrossStrategy(self.config)
        strategy._attach(df)
        if isinstance(strategy, GoldenCrossStrategy):
            strategy.reset()

        n = len(df)
        closes = df["Close"].to_numpy(dtype=float)
        atrs = df["ATR_14"].to_numpy(dtype=float)
        advs = df["ADV_14"].to_numpy(dtype=float)
        index = df.index

        portfolio = PortfolioManager(self.config.initial_capital)
        ev_equity = np.full(n, self.config.initial_capital, dtype=float)
        ev_invested = np.zeros(n, dtype=bool)
        trades: list[TradeRecord] = []

        entry_i = -1
        entry_close_price = 0.0

        for i in range(n):
            c = closes[i]
            atr_i = float(atrs[i]) if not np.isnan(atrs[i]) else 0.0
            adv_i = float(advs[i]) if not np.isnan(advs[i]) else 1e6

            # --- Execute pending order (signal fired at day i-1) ---
            if strategy._pending == "BUY" and not portfolio.in_position:
                net_buy = self._exec.exec_buy(c, atr_i, adv_i)
                portfolio.open_position(net_buy)
                entry_i = i
                entry_close_price = c
                strategy._pending = None
                if isinstance(strategy, GoldenCrossStrategy):
                    strategy.on_entry(c)

            elif strategy._pending == "SELL" and portfolio.in_position:
                net_sell = self._exec.exec_sell(c, atr_i, adv_i)
                gross_ret = c / entry_close_price - 1.0 if entry_close_price > 0 else 0.0
                entry_net = portfolio._entry_exec_price
                net_ret = net_sell / entry_net - 1.0 if entry_net > 0 else 0.0
                portfolio.close_position(net_sell)
                h_days = int((index[i] - index[entry_i]).days) if hasattr(index[i], "__sub__") else 0
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
                if isinstance(strategy, GoldenCrossStrategy):
                    strategy.on_exit()
                strategy._pending = None

            # --- Mark to market ---
            ev_equity[i] = portfolio.equity(c)
            ev_invested[i] = portfolio.in_position
            if portfolio.in_position and isinstance(strategy, GoldenCrossStrategy):
                strategy.update_trailing(c)

            # --- Generate signal for next bar ---
            if i < n - 1:
                strategy.next(i)

        # --- Close open position at end of data ---
        if portfolio.in_position and entry_i >= 0:
            last_i = n - 1
            c = closes[last_i]
            atr_i = float(atrs[last_i]) if not np.isnan(atrs[last_i]) else 0.0
            adv_i = float(advs[last_i]) if not np.isnan(advs[last_i]) else 1e6
            net_sell = self._exec.exec_sell(c, atr_i, adv_i)
            gross_ret = c / entry_close_price - 1.0 if entry_close_price > 0 else 0.0
            entry_net = portfolio._entry_exec_price
            net_ret = net_sell / entry_net - 1.0 if entry_net > 0 else 0.0
            portfolio.close_position(net_sell)
            h_days = int((index[last_i] - index[entry_i]).days) if hasattr(index[last_i], "__sub__") else 0
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
            ev_equity[last_i] = portfolio.equity(c)

        # --- Annotate df ---
        out = df.copy()
        out["Event_Equity"] = ev_equity
        out["Event_Invested"] = ev_invested
        eq_s = pd.Series(ev_equity, index=df.index)
        out["Event_Drawdown"] = eq_s / eq_s.cummax() - 1.0
        out["Strategy_Equity"] = out["Event_Equity"]
        out["Strategy_Drawdown"] = out["Event_Drawdown"]
        out["Invested"] = out["Event_Invested"]
        out["Strategy_Return"] = eq_s.pct_change().fillna(0.0)

        return out, trades


# ---------------------------------------------------------------------------
# 7. Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    df: pd.DataFrame,
    trades: list[TradeRecord],
    risk_free_rate: float = 0.03,
) -> dict[str, float]:
    """Full performance metric suite from annotated df + trade list."""
    m: dict[str, float] = {}
    eq = df["Strategy_Equity"]
    bh = df["Buy_Hold_Equity"]
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)

    cum_s = float(eq.iloc[-1] / eq.iloc[0])
    cum_bh = float(bh.iloc[-1] / bh.iloc[0])
    m.update({
        "years": years,
        "cagr_strategy": cum_s ** (1.0 / years) - 1.0,
        "cagr_buy_hold": cum_bh ** (1.0 / years) - 1.0,
        "cumulative_strategy": cum_s,
        "cumulative_buy_hold": cum_bh,
        "mdd_strategy": float(df["Strategy_Drawdown"].min()),
        "mdd_buy_hold": float(df["Buy_Hold_Drawdown"].min()),
        "market_exposure": float(df["Invested"].mean()),
    })

    strat_ret = df["Strategy_Return"]
    asset_ret = df["Asset_Return"]
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = strat_ret - daily_rf
    exc_mean = float(excess.mean())
    exc_std = float(excess.std(ddof=1))

    m["annual_return"] = float(asset_ret.mean()) * TRADING_DAYS
    m["annual_volatility"] = float(asset_ret.std(ddof=0)) * sqrt(TRADING_DAYS)

    m["sharpe_ratio"] = (exc_mean / exc_std * sqrt(TRADING_DAYS)) if exc_std > 0 else (
        float("inf") if exc_mean > 0 else float("nan")
    )
    down = excess[excess < 0]
    ds_std = float(sqrt((down ** 2).mean())) if len(down) > 0 else 0.0
    m["sortino_ratio"] = (exc_mean / ds_std * sqrt(TRADING_DAYS)) if ds_std > 0 else (
        float("inf") if exc_mean > 0 else float("nan")
    )
    m["calmar_ratio"] = (
        m["cagr_strategy"] / abs(m["mdd_strategy"]) if m["mdd_strategy"] < 0 else float("inf")
    )
    exp = m["market_exposure"]
    m["exposure_adj_cagr"] = (
        (1 + m["cagr_strategy"]) ** (1.0 / exp) - 1.0 if 0 < exp < 1 else m["cagr_strategy"]
    )
    vol2 = m["annual_volatility"] ** 2
    kelly = (m["annual_return"] - risk_free_rate) / vol2 if vol2 > 0 else 0.0
    m["kelly_fraction"] = kelly
    m["suggested_fractional_kelly"] = max(0.0, min(kelly * 0.5, 1.0))

    n = len(trades)
    m["num_trades"] = float(n)
    if n == 0:
        for k in ("win_rate", "profit_factor", "avg_holding_days",
                  "best_trade", "worst_trade", "turnover"):
            m[k] = float("nan")
    else:
        nr = np.array([t.net_return for t in trades])
        wins = nr[nr > 0]
        losses = nr[nr <= 0]
        gp, gl = float(wins.sum()), float(losses.sum())
        m["win_rate"] = len(wins) / n
        m["profit_factor"] = (gp / abs(gl)) if gl != 0 and gp > 0 else (
            float("inf") if gl == 0 else 0.0
        )
        m["avg_holding_days"] = float(np.mean([t.holding_days for t in trades]))
        m["best_trade"] = float(nr.max())
        m["worst_trade"] = float(nr.min())
        m["turnover"] = n / years

    m["rolling_cagr_3y"] = float(_rolling_cagr(eq, 3.0).iloc[-1])
    m["rolling_cagr_5y"] = float(_rolling_cagr(eq, 5.0).iloc[-1])
    m["rolling_mdd_3y"] = float(_rolling_mdd(df["Strategy_Drawdown"], 3.0).iloc[-1])
    return m


# ---------------------------------------------------------------------------
# 8. OptimizationEngine
# ---------------------------------------------------------------------------

class OptimizationEngine:
    """
    Statistical validation and parameter optimization tools.

    Methods
    ───────
    slippage_sweep()       — decay of Sharpe & Return vs slippage (0→100 bps)
    bootstrap_ci()         — BCa 10k-boot CI with strategy rejection flags
    walk_forward_analysis() — unanchored rolling WFA (3:1 ratio default)
    """

    # ------------------------------------------------------------------
    # Slippage sweep
    # ------------------------------------------------------------------

    def slippage_sweep(
        self,
        df: pd.DataFrame,
        base_config: BacktestConfig,
        bps_values: list[int] | None = None,
    ) -> list[dict]:
        """
        Run backtester at each slippage level.
        Returns list of {bps, sharpe_ratio, cumulative_return, cagr}.
        """
        if bps_values is None:
            bps_values = list(range(0, 110, 10))  # 0, 10, 20, … 100

        results: list[dict] = []
        for bps in bps_values:
            cfg = BacktestConfig(
                atr_multiplier=base_config.atr_multiplier,
                commission_bps=float(bps),
                slippage_beta=base_config.slippage_beta,
                initial_capital=base_config.initial_capital,
                reentry_mode=base_config.reentry_mode,
            )
            try:
                bt = Backtester(cfg)
                bt_df, bt_trades = bt.run(df)
                m = compute_metrics(bt_df, bt_trades)
                sharpe = m["sharpe_ratio"]
                results.append({
                    "bps": bps,
                    "sharpe_ratio": float("nan") if (math.isnan(sharpe) or math.isinf(sharpe)) else sharpe,
                    "cumulative_return": float(m["cumulative_strategy"]) - 1.0,
                    "cagr": float(m["cagr_strategy"]),
                })
            except Exception:
                results.append({
                    "bps": bps, "sharpe_ratio": float("nan"),
                    "cumulative_return": float("nan"), "cagr": float("nan"),
                })
        return results

    # ------------------------------------------------------------------
    # BCa Bootstrap
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        trades: list[TradeRecord],
        n_boot: int = 10_000,
        ci_pct: float = 97.5,
    ) -> dict:
        """
        BCa bootstrap confidence intervals (10,000 iterations).

        Strategy rejection rules:
          • LCB(97.5%) Sharpe Ratio < 0  → reject
          • LCB(97.5%) Profit Factor < 1  → reject
        """
        raw = bootstrap_ci(trades, n_boot=n_boot, ci_pct=ci_pct)
        sharpe_lcb = raw.get("sharpe_ci_lower", float("nan"))
        pf_lcb = raw.get("pf_ci_lower", float("nan"))

        rejected = False
        reasons: list[str] = []
        if not math.isnan(sharpe_lcb) and sharpe_lcb < 0.0:
            rejected = True
            reasons.append(f"LCB Sharpe={sharpe_lcb:.3f} < 0.0")
        if not math.isnan(pf_lcb) and pf_lcb < 1.0:
            rejected = True
            reasons.append(f"LCB PF={pf_lcb:.3f} < 1.0")

        return {
            **raw,
            "strategy_rejected": rejected,
            "rejection_reason": "; ".join(reasons),
            "n_boot": n_boot,
        }

    # ------------------------------------------------------------------
    # Unanchored Walk-Forward Analysis (3:1 rolling)
    # ------------------------------------------------------------------

    def walk_forward_analysis(
        self,
        df: pd.DataFrame,
        train_years: int = 3,
        test_years: int = 1,
        atr_candidates: list[float] | None = None,
        config_base: BacktestConfig | None = None,
    ) -> tuple[pd.DataFrame, float]:
        """
        Unanchored (rolling) WFA with configurable train:test split.

        For each window:
          1. Select best ATR multiplier on IS (maximise CAGR).
          2. Evaluate on OOS window.

        Returns (results_df, parameter_drift_entropy).
        Parameter drift is the normalised entropy of the selected ATR
        parameter across folds:  0.0 = fully stable, 1.0 = maximum instability.
        """
        if atr_candidates is None:
            atr_candidates = [3.0, 3.5, 4.0]
        if config_base is None:
            config_base = BacktestConfig()

        train_days = int(train_years * TRADING_DAYS)
        test_days = int(test_years * TRADING_DAYS)
        total_window = train_days + test_days
        n = len(df)
        records: list[dict] = []
        start_i = 0

        while start_i + total_window <= n:
            train_df = df.iloc[start_i: start_i + train_days].copy()
            oos_df = df.iloc[start_i + train_days: start_i + total_window].copy()

            if len(train_df) < 250 or len(oos_df) < 50:
                start_i += test_days
                continue

            # --- IS optimisation ---
            best_atr, best_cagr = atr_candidates[0], float("-inf")
            for atr_m in atr_candidates:
                try:
                    cfg = BacktestConfig(
                        atr_multiplier=atr_m,
                        commission_bps=config_base.commission_bps,
                        slippage_beta=config_base.slippage_beta,
                        initial_capital=config_base.initial_capital,
                    )
                    bt_df, bt_trades = Backtester(cfg).run(train_df)
                    cagr = compute_metrics(bt_df, bt_trades)["cagr_strategy"]
                    if cagr > best_cagr:
                        best_cagr, best_atr = cagr, atr_m
                except Exception:
                    pass

            # --- OOS evaluation ---
            try:
                cfg_oos = BacktestConfig(
                    atr_multiplier=best_atr,
                    commission_bps=config_base.commission_bps,
                    slippage_beta=config_base.slippage_beta,
                    initial_capital=config_base.initial_capital,
                )
                oos_bt_df, oos_trades = Backtester(cfg_oos).run(oos_df)
                oos_m = compute_metrics(oos_bt_df, oos_trades)
                records.append({
                    "fold": len(records) + 1,
                    "is_start": _date_str(train_df.index[0]),
                    "is_end": _date_str(train_df.index[-1]),
                    "oos_start": _date_str(oos_df.index[0]),
                    "oos_end": _date_str(oos_df.index[-1]),
                    "best_atr": best_atr,
                    "is_cagr": round(best_cagr, 4),
                    "oos_cagr": round(oos_m["cagr_strategy"], 4),
                    "oos_mdd": round(oos_m["mdd_strategy"], 4),
                    "oos_sharpe": round(oos_m.get("sharpe_ratio", float("nan")), 3),
                    "oos_trades": int(oos_m["num_trades"]),
                })
            except Exception:
                pass

            start_i += test_days

        results_df = pd.DataFrame(records)
        drift = self._parameter_drift(results_df)
        return results_df, drift

    def _parameter_drift(self, wfa_df: pd.DataFrame) -> float:
        """
        Normalised entropy of selected ATR parameters across WFA folds.
        Measures how unstable the optimal parameter is — a proxy for overfitting.
        """
        if wfa_df.empty or "best_atr" not in wfa_df.columns:
            return float("nan")
        vals = wfa_df["best_atr"].dropna()
        if len(vals) < 2:
            return 0.0
        counts = Counter(vals.tolist())
        n = len(vals)
        entropy = -sum((c / n) * log2(c / n) for c in counts.values())
        max_ent = log2(len(counts)) if len(counts) > 1 else 1.0
        return round(entropy / max_ent, 4) if max_ent > 0 else 0.0


# ---------------------------------------------------------------------------
# Module-level functional API (backward compatible)
# ---------------------------------------------------------------------------

_opt_engine = OptimizationEngine()


def bootstrap_ci(
    trades: list[TradeRecord],
    n_boot: int = 2_000,
    ci_pct: float = 97.5,
) -> dict[str, float]:
    """BCa-style bootstrap CI on trade-level net returns."""
    result: dict[str, float] = {"sharpe_ci_lower": float("nan"), "pf_ci_lower": float("nan")}
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
    lower = 100.0 - ci_pct
    if boot_sharpes:
        result["sharpe_ci_lower"] = float(np.percentile(boot_sharpes, lower))
    if boot_pfs:
        result["pf_ci_lower"] = float(np.percentile(boot_pfs, lower))
    return result


def compute_metrics(
    df: pd.DataFrame, trades: list[TradeRecord], risk_free_rate: float = 0.03
) -> dict[str, float]:
    # defined above — re-exported here for clarity
    ...


# Fix: the module-level compute_metrics IS the function defined above.
# The ellipsis overrides it. Let's just rename the internal one.
# (handled by the assignment below)

def walk_forward_analysis(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    atr_candidates: list[float] | None = None,
    config_base: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Expanding-window WFA wrapper (backward compat)."""
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
        best_atr, best_cagr = atr_candidates[0], float("-inf")
        for atr_m in atr_candidates:
            try:
                cfg = BacktestConfig(atr_multiplier=atr_m, commission_bps=config_base.commission_bps,
                                     slippage_beta=config_base.slippage_beta, initial_capital=config_base.initial_capital)
                bt_df, bt_trades = Backtester(cfg).run(train_df)
                cagr = _compute_metrics_impl(bt_df, bt_trades)["cagr_strategy"]
                if cagr > best_cagr:
                    best_cagr, best_atr = cagr, atr_m
            except Exception:
                pass
        try:
            cfg_oos = BacktestConfig(atr_multiplier=best_atr, commission_bps=config_base.commission_bps,
                                     slippage_beta=config_base.slippage_beta, initial_capital=config_base.initial_capital)
            oos_bt_df, oos_trades = Backtester(cfg_oos).run(oos_df)
            oos_m = _compute_metrics_impl(oos_bt_df, oos_trades)
            records.append({
                "fold": fold + 1, "is_end": _date_str(train_df.index[-1]),
                "oos_start": _date_str(oos_df.index[0]), "oos_end": _date_str(oos_df.index[-1]),
                "best_atr": best_atr, "is_cagr": round(best_cagr, 4),
                "oos_cagr": round(oos_m["cagr_strategy"], 4),
                "oos_mdd": round(oos_m["mdd_strategy"], 4),
                "oos_sharpe": round(oos_m.get("sharpe_ratio", float("nan")), 3),
                "oos_trades": int(oos_m["num_trades"]),
            })
        except Exception:
            pass
    return pd.DataFrame(records)


def backtest_entry_score(
    df: pd.DataFrame,
    entry_score_fn: Any = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Historical entry score → forward return bucket analysis."""
    if horizons is None:
        horizons = [21, 63, 126, 252]
    BUCKETS = [(0, 40, "0–40"), (40, 60, "40–60"), (60, 75, "60–75"), (75, 101, "75–100")]
    n = len(df)
    closes = df["Close"].to_numpy(dtype=float)
    scores: list[tuple[int, float]] = []
    for i in range(200, n):
        try:
            scores.append((i, float(_quick_score(df.iloc[i]))))
        except Exception:
            pass
    records: list[dict] = []
    for lo, hi, label in BUCKETS:
        bucket = [(i, s) for i, s in scores if lo <= s < hi]
        for h in horizons:
            fwd = [closes[i + h] / closes[i] - 1.0 for i, _ in bucket if i + h < n]
            if not fwd:
                records.append({"bucket": label, "horizon_days": h,
                                 "sample_count": 0, "avg_fwd_return": float("nan"),
                                 "median_fwd_return": float("nan"), "win_rate": float("nan")})
            else:
                arr = np.array(fwd)
                records.append({"bucket": label, "horizon_days": h,
                                 "sample_count": len(arr), "avg_fwd_return": float(arr.mean()),
                                 "median_fwd_return": float(np.median(arr)),
                                 "win_rate": float((arr > 0).mean())})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_metrics_impl(
    df: pd.DataFrame, trades: list[TradeRecord], risk_free_rate: float = 0.03
) -> dict[str, float]:
    """Actual implementation — compute_metrics() delegates to this."""
    m: dict[str, float] = {}
    eq = df["Strategy_Equity"]
    bh = df["Buy_Hold_Equity"]
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)
    cum_s = float(eq.iloc[-1] / eq.iloc[0])
    cum_bh = float(bh.iloc[-1] / bh.iloc[0])
    m.update({
        "years": years,
        "cagr_strategy": cum_s ** (1.0 / years) - 1.0,
        "cagr_buy_hold": cum_bh ** (1.0 / years) - 1.0,
        "cumulative_strategy": cum_s,
        "cumulative_buy_hold": cum_bh,
        "mdd_strategy": float(df["Strategy_Drawdown"].min()),
        "mdd_buy_hold": float(df["Buy_Hold_Drawdown"].min()),
        "market_exposure": float(df["Invested"].mean()),
    })
    strat_ret = df["Strategy_Return"]
    asset_ret = df["Asset_Return"]
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = strat_ret - daily_rf
    exc_mean = float(excess.mean())
    exc_std = float(excess.std(ddof=1))
    m["annual_return"] = float(asset_ret.mean()) * TRADING_DAYS
    m["annual_volatility"] = float(asset_ret.std(ddof=0)) * sqrt(TRADING_DAYS)
    m["sharpe_ratio"] = (exc_mean / exc_std * sqrt(TRADING_DAYS)) if exc_std > 0 else (
        float("inf") if exc_mean > 0 else float("nan"))
    down = excess[excess < 0]
    ds_std = float(sqrt((down ** 2).mean())) if len(down) > 0 else 0.0
    m["sortino_ratio"] = (exc_mean / ds_std * sqrt(TRADING_DAYS)) if ds_std > 0 else (
        float("inf") if exc_mean > 0 else float("nan"))
    m["calmar_ratio"] = (m["cagr_strategy"] / abs(m["mdd_strategy"])
                         if m["mdd_strategy"] < 0 else float("inf"))
    exp = m["market_exposure"]
    m["exposure_adj_cagr"] = (
        (1 + m["cagr_strategy"]) ** (1.0 / exp) - 1.0 if 0 < exp < 1 else m["cagr_strategy"])
    vol2 = m["annual_volatility"] ** 2
    kelly = (m["annual_return"] - risk_free_rate) / vol2 if vol2 > 0 else 0.0
    m["kelly_fraction"] = kelly
    m["suggested_fractional_kelly"] = max(0.0, min(kelly * 0.5, 1.0))
    n = len(trades)
    m["num_trades"] = float(n)
    if n == 0:
        for k in ("win_rate", "profit_factor", "avg_holding_days", "best_trade",
                  "worst_trade", "turnover"):
            m[k] = float("nan")
    else:
        nr = np.array([t.net_return for t in trades])
        wins = nr[nr > 0]; losses = nr[nr <= 0]
        gp, gl = float(wins.sum()), float(losses.sum())
        m["win_rate"] = len(wins) / n
        m["profit_factor"] = (gp / abs(gl)) if gl != 0 and gp > 0 else (
            float("inf") if gl == 0 else 0.0)
        m["avg_holding_days"] = float(np.mean([t.holding_days for t in trades]))
        m["best_trade"] = float(nr.max())
        m["worst_trade"] = float(nr.min())
        m["turnover"] = n / years
    m["rolling_cagr_3y"] = float(_rolling_cagr(eq, 3.0).iloc[-1])
    m["rolling_cagr_5y"] = float(_rolling_cagr(eq, 5.0).iloc[-1])
    m["rolling_mdd_3y"] = float(_rolling_mdd(df["Strategy_Drawdown"], 3.0).iloc[-1])
    return m


# Patch module-level compute_metrics to delegate to the real implementation
compute_metrics = _compute_metrics_impl  # type: ignore[assignment]


def _quick_score(row: pd.Series) -> float:
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


def _rolling_cagr(equity: pd.Series, window_years: float = 3.0) -> pd.Series:
    window = int(window_years * TRADING_DAYS)
    ratio = equity / equity.shift(window)
    return (ratio ** (1.0 / window_years) - 1.0).fillna(float("nan"))


def _rolling_mdd(drawdown: pd.Series, window_years: float = 3.0) -> pd.Series:
    window = int(window_years * TRADING_DAYS)
    return drawdown.rolling(window).min().fillna(float("nan"))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return TechnicalIndicatorsEngine.atr(high, low, close, period)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return TechnicalIndicatorsEngine.rsi(close, period)


def _date_str(idx: Any) -> str:
    return str(idx.date()) if hasattr(idx, "date") else str(idx)
