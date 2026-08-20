"""Shared business/economics helpers."""
from __future__ import annotations

from core.config import Settings


def format_money(value: float, currency: str = "INR") -> str:
    """Format large South-Asian-style money amounts: 840000 -> Rs 8.4L, 12500000 -> Rs 1.25Cr."""
    symbol = {"INR": "Rs ", "USD": "$", "EUR": "\u20ac"}.get(currency, "")
    if currency == "INR":
        if abs(value) >= 10_000_000:
            return f"{symbol}{value / 10_000_000:.2f}Cr"
        if abs(value) >= 100_000:
            return f"{symbol}{value / 100_000:.1f}L"
        return f"{symbol}{value:,.0f}"
    return f"{symbol}{value:,.0f}"


def replacement_cost(annual_salary: float, settings: Settings | None = None) -> float:
    s = settings or __import__("core.config", fromlist=["get_settings"]).get_settings()
    return annual_salary * s.replacement_cost_multiplier
