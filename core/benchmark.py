"""JCB 3DX benchmark case from the project plan, shared by the Validation tab and pytest."""
from pathlib import Path

import pandas as pd

from core.cashflows import Inputs, break_even_hours, compare
from core.costs import hourly_cost
from core.residual import load_curve, retention

HAND_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark_hand.csv"
TOL = 0.005  # pass rule: within 0.5%


def benchmark_inputs():
    curve = load_curve()
    return Inputs(
        price=3_500_000, life_years=9, hours_per_year=1500, horizon_years=5,
        residual=lambda a: retention("BL", a, curve),
        discount_rate=0.11, loan_rate=0.11, tax_rate=0.2517,
        fuel_lph=6.0, diesel_price=97.83, lube_frac=0.25, repair_factor=1.5,
        tyre_cost=160_000, tyre_life_h=3000, operator_wage_month=25_000, operator_hours_month=200,
        rent_mode="monthly", rent_month=90_000, rent_month_hours=240, rent_extra_hour=450,
        rent_includes_operator=True, rent_includes_fuel=False,
        lease_rental_month=50_000, lease_term_years=5,
    )


def tool_values(inp):
    rows, _ = hourly_cost(inp.hourly_dict())
    out = {f"hourly: {k}": v for k, v in rows.items()}
    out["hourly: total"] = sum(rows.values())
    for name, r in compare(inp).items():
        out[f"{name} NPV"] = r["npv"]
        out[f"{name} EAC"] = r["eac"]
    out["Break-even H* (buy vs rent)"] = break_even_hours(inp)
    return out


def validation_table():
    """Hand values next to tool values with % difference and pass/fail (blank hand value = pending)."""
    hand = pd.read_csv(HAND_PATH)
    tool = tool_values(benchmark_inputs())
    hand["tool_value"] = hand["item"].map(tool)
    hand["diff_pct"] = (hand["tool_value"] - hand["hand_value"]) / hand["hand_value"] * 100
    hand["result"] = hand["diff_pct"].abs().le(TOL * 100).map({True: "pass", False: "FAIL"})
    hand.loc[hand["hand_value"].isna(), "result"] = "pending"
    return hand
