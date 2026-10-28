"""Checks the tool against the hand-calculated benchmark and the plan's edge cases.

Hand values live in data/benchmark_hand.csv, filled in by the Validation Engineer from the
plain spreadsheet (no AI). Rows with a blank hand value are skipped, not failed.
"""
from dataclasses import replace

import pandas as pd
import pytest

from core.benchmark import HAND_PATH, TOL, benchmark_inputs, tool_values
from core.cashflows import buy_cashflows, compare, eac, lease_cashflows, npv, rent_cashflows
from core.costs import hourly_cost


def hand_rows():
    if not HAND_PATH.exists():
        return []
    df = pd.read_csv(HAND_PATH)
    return [(r.item, r.hand_value) for r in df.itertuples() if pd.notna(r.hand_value)]


@pytest.mark.parametrize("item,hand", hand_rows() or [pytest.param(None, None, marks=pytest.mark.skip("no hand values yet"))])
def test_matches_hand_sheet(item, hand):
    tool = tool_values(benchmark_inputs())[item]
    assert tool == pytest.approx(hand, rel=TOL)


# ---------- formula checks ----------

def test_hourly_ownership_formulas():
    inp = replace(benchmark_inputs(), residual=lambda a: 0.2)
    rows, aai = hourly_cost(inp.hourly_dict())
    P, S, n, H = 3_500_000, 700_000, 9, 1500
    assert rows["Depreciation"] == pytest.approx((P - S) / (n * H))
    assert aai == pytest.approx((P * (n + 1) + S * (n - 1)) / (2 * n))
    assert rows["Interest, insurance, tax"] == pytest.approx((0.11 + 0.01) * aai / H)
    assert rows["Repairs & maintenance"] == pytest.approx(1.5 * P / (n * H))


def test_buy_cash_purchase_no_tax():
    """With t = 0, r = 0 and no opex, buy NPV is simply price minus salvage."""
    inp = replace(benchmark_inputs(), tax_rate=0, discount_rate=0, fuel_lph=0, repair_factor=0,
                  tyre_cost=0, operator_wage_month=0, insurance_rate=0, residual=lambda a: 0.3)
    assert npv(buy_cashflows(inp), 0) == pytest.approx(3_500_000 * 0.7)


def test_loan_at_discount_rate_is_npv_neutral_before_tax():
    """Borrowing at the discount rate (same effective annual rate) should not change pre-tax NPV."""
    nominal = 12 * (1.11 ** (1 / 12) - 1)
    inp = replace(benchmark_inputs(), tax_rate=0, discount_rate=0.11, loan_rate=nominal)
    cash = npv(buy_cashflows(inp), 0.11)
    loan = npv(buy_cashflows(replace(inp, loan_fraction=0.8, loan_years=3)), 0.11)
    assert loan == pytest.approx(cash, rel=1e-9)


# ---------- edge cases from the plan ----------

def test_very_low_utilisation_rent_wins():
    res = compare(benchmark_inputs().with_hours(100))
    assert res["Rent"]["npv"] < res["Buy"]["npv"]


def test_zero_salvage_runs_and_costs_more():
    base = compare(benchmark_inputs())["Buy"]["npv"]
    zero = compare(replace(benchmark_inputs(), residual=lambda a: 0.0))["Buy"]["npv"]
    assert zero > base


def test_eac_zero_discount_rate():
    assert eac(1000.0, 0.0, 5) == pytest.approx(200.0)
    res = compare(replace(benchmark_inputs(), discount_rate=0.0))
    assert res["Buy"]["eac"] == pytest.approx(res["Buy"]["npv"] / 5)


def test_horizon_longer_than_life_replaces_machine():
    inp = replace(benchmark_inputs(), horizon_years=12, tax_rate=0)
    cf = buy_cashflows(inp)
    assert cf[9] > 3_500_000 * 0.5  # repurchase at year 9 outweighs the salvage received


def test_lease_term_shorter_than_horizon_renews():
    inp = replace(benchmark_inputs(), lease_term_years=2, horizon_years=5, rent_escalation=0.05, tax_rate=0)
    cf = lease_cashflows(inp)
    assert cf[3] > cf[2] and cf[2] == pytest.approx(cf[1])


def test_rent_monthly_package_overflow():
    inp = benchmark_inputs().with_hours(3000)  # 3000 > 12 x 240
    first_year = rent_cashflows(replace(inp, tax_rate=0, fuel_lph=0))[1]
    assert first_year == pytest.approx(12 * 90_000 + (3000 - 2880) * 450)
