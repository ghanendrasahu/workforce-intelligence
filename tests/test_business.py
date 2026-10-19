"""Tests for shared business/economics helpers."""

from __future__ import annotations

from core.business import format_money, replacement_cost
from core.config import Settings


def test_format_money_indian_units():
    assert format_money(84_000) == "Rs 84,000"
    assert format_money(840_000) == "Rs 8.4L"
    assert format_money(12_500_000) == "Rs 1.25Cr"
    assert format_money(-840_000) == "Rs -8.4L"


def test_format_money_other_currencies():
    assert format_money(1_500, "USD") == "$1,500"
    assert format_money(1_500, "EUR") == "\u20ac1,500"
    assert format_money(1_500, "GBP") == "1,500"


def test_replacement_cost_uses_multiplier():
    settings = Settings(replacement_cost_multiplier=0.5)
    assert replacement_cost(100_000, settings) == 50_000
