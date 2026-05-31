from __future__ import annotations

import os
import re


TICKER_RE = re.compile(r"^[A-Za-z0-9.^=_-]{1,24}(?:\.[A-Za-z]{1,4})?$")
PERIOD_RE = re.compile(r"^(1mo|3mo|6mo|1y|2y|5y|10y|max)$", re.IGNORECASE)


def default_period() -> str:
    raw = os.getenv("DEFAULT_PERIOD", "max").strip().lower()
    return raw if PERIOD_RE.match(raw) else "max"


def parse_request(text: str) -> tuple[str | None, str]:
    tokens = [token.strip() for token in text.replace(",", " ").split() if token.strip()]
    if not tokens:
        return None, default_period()

    first = strip_bot_mention(tokens[0])
    if first.lower() in {"/analyze", "/a"}:
        tokens = tokens[1:]
    elif first.lower() in {"/start", "/help"}:
        return None, default_period()

    if not tokens:
        return None, default_period()

    ticker = strip_bot_mention(tokens[0]).upper()
    period = default_period()
    for token in tokens[1:]:
        normalized = token.lower().replace("period=", "")
        if PERIOD_RE.match(normalized):
            period = normalized
    return ticker, period


def strip_bot_mention(value: str) -> str:
    return value.split("@", 1)[0]
