from __future__ import annotations

import numpy as np
import pandas as pd
import sys

from stock_analyzer import analyze_price_frame, format_telegram_report

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def make_sample_frame(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    trend = np.linspace(100.0, 155.0, rows)
    noise = rng.normal(0.0, 1.8, rows).cumsum()
    close = trend + noise
    high = close + rng.uniform(0.8, 3.0, rows)
    low = close - rng.uniform(0.8, 3.0, rows)
    open_ = close + rng.normal(0.0, 1.0, rows)
    volume = rng.integers(800_000, 1_400_000, rows)
    volume[-1] = int(volume[-20:].mean() * 1.8)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


if __name__ == "__main__":
    frame = make_sample_frame()
    result = analyze_price_frame("SAMPLE", frame, benchmark_frame=frame)
    report = format_telegram_report(result)
    assert "SAMPLE 기술적 분석" in report
    assert result.rows > 60
    print(report)
