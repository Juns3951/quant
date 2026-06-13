from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

if TYPE_CHECKING:
    from stock_analyzer import LongTermResult


def generate_backtest_chart(result: "LongTermResult") -> bytes | None:
    df = result.frame
    if df is None or df.empty:
        return None

    try:
        has_sweep = (
            hasattr(result, "slippage_sweep")
            and result.slippage_sweep is not None
            and len(result.slippage_sweep) > 0
        )

        plt.style.use("dark_background")
        if has_sweep:
            fig = plt.figure(figsize=(12, 16), dpi=150)
            fig.patch.set_facecolor("#0d1117")
            gs = fig.add_gridspec(4, 1, height_ratios=[4, 3, 2.5, 2.5], hspace=0.1)
            ax_sweep = fig.add_subplot(gs[3])
        else:
            fig = plt.figure(figsize=(12, 13), dpi=150)
            fig.patch.set_facecolor("#0d1117")
            gs = fig.add_gridspec(3, 1, height_ratios=[4, 3.5, 2.5], hspace=0.08)

        ax_price = fig.add_subplot(gs[0])
        ax_equity = fig.add_subplot(gs[1], sharex=ax_price)
        ax_dd = fig.add_subplot(gs[2], sharex=ax_price)

        _plot_price(ax_price, df, result)
        _plot_equity(ax_equity, df, result)
        _plot_drawdown(ax_dd, df)

        if has_sweep:
            _plot_slippage_sweep(ax_sweep, result.slippage_sweep)

        title = (
            f"{result.ticker}  |  {result.start_date} ~ {result.as_of}"
            f"  |  Strategy CAGR {result.cagr_strategy:.1%} vs B&H {result.cagr_buy_hold:.1%}"
            f"  |  MDD {result.mdd_strategy:.1%}"
        )
        fig.suptitle(title, fontsize=9, color="#c9d1d9", y=0.995)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf.read()
    except Exception:
        return None
    finally:
        plt.close("all")


def _plot_price(ax: plt.Axes, df: pd.DataFrame, result: "LongTermResult") -> None:
    ax.set_facecolor("#0d1117")
    ax.plot(df.index, df["Close"], color="#e6edf3", linewidth=0.8, label="Close")
    ax.plot(df.index, df["EMA_50"], color="#f0883e", linewidth=1.0, label="EMA50")
    ax.plot(df.index, df["SMA_200"], color="#f85149", linewidth=1.0, label="SMA200")
    ax.plot(df.index, df["ATR_Trailing_Stop"], color="#8b949e", linewidth=0.8,
            linestyle="--", label="ATR Stop", alpha=0.7)

    if result.trades is not None and not result.trades.empty:
        trades = result.trades.copy()
        entry_dates = pd.to_datetime(trades["Entry Date"])
        exit_dates = pd.to_datetime(trades["Exit Date"])

        entry_prices = df["Close"].reindex(entry_dates, method="nearest").values
        exit_prices = df["Close"].reindex(exit_dates, method="nearest").values

        ax.scatter(entry_dates, entry_prices, marker="^", color="#3fb950", s=40,
                   zorder=5, label="Buy", alpha=0.9)
        ax.scatter(exit_dates, exit_prices, marker="v", color="#f85149", s=40,
                   zorder=5, label="Sell", alpha=0.9)

    ax.set_ylabel("Price", fontsize=8, color="#8b949e")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax.tick_params(labelbottom=False, colors="#8b949e", labelsize=7)
    ax.grid(axis="y", color="#21262d", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")


def _plot_equity(ax: plt.Axes, df: pd.DataFrame, result: "LongTermResult") -> None:
    ax.set_facecolor("#0d1117")
    ax.plot(df.index, df["Strategy_Equity"], color="#58a6ff", linewidth=1.0,
            label=f"Strategy ({result.cagr_strategy:.1%} CAGR)")
    ax.plot(df.index, df["Buy_Hold_Equity"], color="#8b949e", linewidth=0.8,
            linestyle="--", label=f"Buy & Hold ({result.cagr_buy_hold:.1%} CAGR)", alpha=0.8)

    ax.set_ylabel("Equity", fontsize=8, color="#8b949e")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax.tick_params(labelbottom=False, colors="#8b949e", labelsize=7)
    ax.grid(axis="y", color="#21262d", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")


def _plot_drawdown(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_facecolor("#0d1117")
    ax.fill_between(df.index, df["Strategy_Drawdown"] * 100, 0,
                    color="#f85149", alpha=0.5, label="Strategy DD")
    ax.fill_between(df.index, df["Buy_Hold_Drawdown"] * 100, 0,
                    color="#8b949e", alpha=0.3, label="B&H DD")

    ax.set_ylabel("Drawdown %", fontsize=8, color="#8b949e")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.3)
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.grid(axis="y", color="#21262d", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")


def _plot_slippage_sweep(ax: plt.Axes, sweep: list[dict]) -> None:
    """Dual-axis chart: Sharpe (left) and CAGR % (right) vs slippage bps."""
    import math
    ax.set_facecolor("#0d1117")

    bps_all = [s["bps"] for s in sweep]
    sharpes = [s.get("sharpe_ratio", float("nan")) for s in sweep]
    cagrs = [s.get("cagr", float("nan")) for s in sweep]

    valid_s = [(b, v) for b, v in zip(bps_all, sharpes) if not math.isnan(v)]
    valid_c = [(b, v) for b, v in zip(bps_all, cagrs) if not math.isnan(v)]

    ax2 = ax.twinx()
    ax2.set_facecolor("#0d1117")

    if valid_s:
        bx, sy = zip(*valid_s)
        ax.plot(bx, sy, color="#58a6ff", linewidth=1.5, marker="o", markersize=3, label="Sharpe")

    if valid_c:
        bx2, cy = zip(*valid_c)
        ax2.plot(bx2, [v * 100 for v in cy], color="#3fb950", linewidth=1.5,
                 marker="s", markersize=3, linestyle="--", label="CAGR %")

    ax.axhline(0, color="#f85149", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Slippage (bps)", fontsize=8, color="#8b949e")
    ax.set_ylabel("Sharpe Ratio", fontsize=8, color="#58a6ff")
    ax2.set_ylabel("CAGR %", fontsize=8, color="#3fb950")
    ax.set_title("Slippage Sensitivity", fontsize=8, color="#8b949e", pad=3)

    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b,
              loc="upper right", fontsize=7, framealpha=0.3)

    ax.tick_params(colors="#8b949e", labelsize=7)
    ax2.tick_params(colors="#8b949e", labelsize=7)
    ax.grid(axis="y", color="#21262d", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#21262d")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    import pandas as _pd

    rng = np.random.default_rng(42)
    rows = 800
    dates = _pd.date_range("2022-01-01", periods=rows, freq="B")
    trend = np.linspace(100.0, 155.0, rows)
    close = trend + rng.normal(0.0, 1.8, rows).cumsum()
    high = close + rng.uniform(0.8, 3.0, rows)
    low = close - rng.uniform(0.8, 3.0, rows)
    frame = _pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(800_000, 1_400_000, rows)},
        index=dates,
    )
    from stock_analyzer import analyze_price_frame
    result = analyze_price_frame("SAMPLE", frame)
    chart_bytes = generate_backtest_chart(result)
    if chart_bytes:
        with open("/tmp/backtest_sample.png", "wb") as f:
            f.write(chart_bytes)
        print(f"Chart saved: {len(chart_bytes):,} bytes")
    else:
        print("Chart generation failed")
