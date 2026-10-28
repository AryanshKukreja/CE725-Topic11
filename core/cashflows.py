"""After-tax buy / lease / rent cash flows, NPV, EAC and break-even utilisation.

Conventions (state these in the method note and the hand sheet):
- Cash flows are COSTS, so positive = money out. Lower NPV is better.
- Annual periods, payments at year end (arrears), except the purchase at year 0.
- Every operating cost and rental is tax-deductible at rate t; the buyer also gets a tax shield on
  tax depreciation (WDV, single asset; block pooling and the 180-day half-rate rule are ignored).
- On sale, tax is paid on (sale price - closing WDV); a negative gap is a tax saving.
- If the horizon is longer than the life, the machine is sold at end of life and replaced at the
  same price. A lease shorter than the horizon is renewed at the escalated rental.
"""
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from core.costs import hourly_cost, average_annual_investment


@dataclass
class Inputs:
    # Machine and ownership
    price: float
    life_years: int
    hours_per_year: float
    residual: Callable[[float], float] = field(default=lambda age: 0.10)  # resale / price at a given age
    salvage_scale: float = 1.0
    life_hours: float | None = None          # repair basis; defaults to life_years x hours_per_year
    insurance_rate: float = 0.01             # per year, on price (after-tax model) / on AAI (hourly model)
    prop_tax_rate: float = 0.0
    # Finance and tax
    discount_rate: float = 0.11
    tax_rate: float = 0.2517
    wdv_rate: float = 0.15
    loan_fraction: float = 0.0
    loan_rate: float = 0.11
    loan_years: int = 5
    horizon_years: int = 5
    # Operating
    fuel_lph: float = 6.0
    diesel_price: float = 97.83
    lube_frac: float = 0.25
    repair_factor: float = 1.5
    tyre_cost: float = 0.0
    tyre_life_h: float = 0.0
    operator_wage_month: float = 25000.0
    operator_hours_month: float = 200.0
    # Lease
    lease_rental_month: float = 0.0
    lease_term_years: int = 5
    lessor_covers_repairs: bool = False
    # Rent
    rent_mode: str = "monthly"               # "hourly" or "monthly"
    rent_hourly: float = 0.0
    rent_month: float = 0.0
    rent_month_hours: float = 240.0
    rent_extra_hour: float = 0.0
    rent_includes_operator: bool = True      # wet hire
    rent_includes_fuel: bool = False
    rent_escalation: float = 0.0             # per year, also applied to lease renewals

    def __post_init__(self):
        if self.life_hours is None:
            self.life_hours = self.life_years * self.hours_per_year

    def salvage_at(self, age):
        return self.price * self.residual(age) * self.salvage_scale

    def with_hours(self, H):
        """Same machine at a different utilisation; the repair basis (life hours) stays fixed."""
        return replace(self, hours_per_year=H)

    def hourly_dict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "residual"}
        d["salvage"] = self.salvage_at(self.life_years)
        return d


# ---------- helpers ----------

def discount_factors(r, T):
    return 1.0 / (1.0 + r) ** np.arange(T + 1)


def npv(cf, r):
    return float(np.sum(np.asarray(cf) * discount_factors(r, len(cf) - 1)))


def eac(npv_value, r, T):
    """Equivalent annual cost. r = 0 falls back to NPV / T (the annuity factor's limit)."""
    if abs(r) < 1e-12:
        return npv_value / T
    return npv_value * r / (1 - (1 + r) ** -T)


def loan_schedule_annual(principal, annual_rate, years, discount_rate=0.0):
    """Monthly EMI loan, aggregated by year. Returns (payments, interest, closing balance) per year.

    Each EMI is carried forward to year end at the discount rate, so lumping 12 payments into one
    annual cash flow does not understate their present value.
    """
    i, N = annual_rate / 12, years * 12
    emi = principal / N if i == 0 else principal * i / (1 - (1 + i) ** -N)
    bal, pay, intr, close = principal, [], [], []
    for _ in range(years):
        p_y = int_y = 0.0
        for m in range(1, 13):
            int_m = bal * i
            bal -= emi - int_m
            p_y += emi * (1 + discount_rate) ** ((12 - m) / 12)
            int_y += int_m
        pay.append(p_y)
        intr.append(int_y)
        close.append(max(bal, 0.0))
    return np.array(pay), np.array(intr), np.array(close)


def variable_cost_per_hour(inp, include_repairs=True, include_tyres=True,
                           include_fuel=True, include_operator=True):
    rows, _ = hourly_cost(inp.hourly_dict())
    v = 0.0
    if include_fuel:
        v += rows["Fuel"] + rows["Lubricants"]
    if include_repairs:
        v += rows["Repairs & maintenance"]
    if include_tyres:
        v += rows["Tyres"]
    if include_operator:
        v += rows["Operator"]
    return v


def rent_charge_per_year(inp, H):
    """Hire charge for H hours in a year, before escalation.

    Monthly mode: hire only as many months as needed (H / hours-per-month, continuous) up to 12;
    hours beyond 12 full months are billed at the extra-hour rate.
    """
    if inp.rent_mode == "hourly":
        return inp.rent_hourly * H
    full = 12 * inp.rent_month_hours
    if H <= full:
        return inp.rent_month * H / inp.rent_month_hours
    return 12 * inp.rent_month + (H - full) * inp.rent_extra_hour


# ---------- the three options ----------

def buy_cashflows(inp):
    T, t, H, P = inp.horizon_years, inp.tax_rate, inp.hours_per_year, inp.price
    cf = np.zeros(T + 1)

    opex = variable_cost_per_hour(inp) * H + (inp.insurance_rate + inp.prop_tax_rate) * P
    cf[1:] += opex * (1 - t)

    start = 0
    while start < T:
        end = min(start + inp.life_years, T)
        loan = P * inp.loan_fraction
        cf[start] += P - loan
        wdv = P
        for k in range(start + 1, end + 1):
            dep = wdv * inp.wdv_rate
            wdv -= dep
            cf[k] -= t * dep
        if loan > 0:
            pay, intr, close = loan_schedule_annual(loan, inp.loan_rate, inp.loan_years, inp.discount_rate)
            for j in range(min(inp.loan_years, end - start)):
                cf[start + 1 + j] += pay[j] - t * intr[j]
            held = end - start
            if held < inp.loan_years:
                cf[end] += close[held - 1]       # settle the outstanding loan on sale
        S = inp.salvage_at(end - start)
        cf[end] += -S + t * (S - wdv)
        start = end
    return cf


def lease_cashflows(inp):
    T, t, H = inp.horizon_years, inp.tax_rate, inp.hours_per_year
    cf = np.zeros(T + 1)
    v = variable_cost_per_hour(inp, include_repairs=not inp.lessor_covers_repairs)
    for k in range(1, T + 1):
        renewal = (k - 1) // inp.lease_term_years
        rental = inp.lease_rental_month * 12 * (1 + inp.rent_escalation) ** (renewal * inp.lease_term_years)
        cf[k] = (rental + v * H) * (1 - t)
    return cf


def rent_cashflows(inp):
    T, t, H = inp.horizon_years, inp.tax_rate, inp.hours_per_year
    cf = np.zeros(T + 1)
    v = variable_cost_per_hour(inp, include_repairs=False, include_tyres=False,
                               include_fuel=not inp.rent_includes_fuel,
                               include_operator=not inp.rent_includes_operator)
    base = rent_charge_per_year(inp, H)
    for k in range(1, T + 1):
        cf[k] = (base * (1 + inp.rent_escalation) ** (k - 1) + v * H) * (1 - t)
    return cf


OPTIONS = {"Buy": buy_cashflows, "Lease": lease_cashflows, "Rent": rent_cashflows}


def compare(inp):
    """NPV, EAC and EAC per hour for each option."""
    out = {}
    for name, fn in OPTIONS.items():
        cf = fn(inp)
        n = npv(cf, inp.discount_rate)
        e = eac(n, inp.discount_rate, inp.horizon_years)
        out[name] = {"cashflows": cf, "npv": n, "eac": e, "eac_per_hour": e / inp.hours_per_year}
    return out


def finance_lease_rental(price, residual_value, annual_rate, months):
    """Monthly rental of a finance lease that recovers price less the PV of the residual."""
    i = annual_rate / 12
    pv = price - residual_value / (1 + i) ** months
    return pv / months if i == 0 else pv * i / (1 - (1 + i) ** -months)


# ---------- break-even and sensitivity ----------

def break_even_hours(inp, a="Buy", b="Rent", lo=10.0, hi=6000.0):
    """Hours/year at which options a and b have the same after-tax NPV. None if no crossing."""
    f = lambda H: npv(OPTIONS[a](inp.with_hours(H)), inp.discount_rate) - npv(OPTIONS[b](inp.with_hours(H)), inp.discount_rate)
    f_lo, f_hi = f(lo), f(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return None
    return brentq(f, lo, hi, xtol=0.01)


def break_even_simple(inp):
    """Pre-tax textbook form H* = F_own / (R_rent - v_own). None if rent is never dearer per hour."""
    d = inp.hourly_dict()
    S, P, n = d["salvage"], inp.price, inp.life_years
    F_own = (P - S) / n + (inp.discount_rate + inp.insurance_rate + inp.prop_tax_rate) * average_annual_investment(P, S, n)
    v_own = variable_cost_per_hour(inp)
    rate = inp.rent_hourly if inp.rent_mode == "hourly" else inp.rent_month / inp.rent_month_hours
    R_rent = rate + variable_cost_per_hour(inp, include_repairs=False, include_tyres=False,
                                           include_fuel=not inp.rent_includes_fuel,
                                           include_operator=not inp.rent_includes_operator)
    if R_rent <= v_own:
        return None
    return F_own / (R_rent - v_own)


SENSITIVITY_PARAMS = {
    "Purchase price": lambda i, s: replace(i, price=i.price * s),
    "Utilisation (hours/yr)": lambda i, s: i.with_hours(i.hours_per_year * s),
    "Discount rate": lambda i, s: replace(i, discount_rate=i.discount_rate * s),
    "Rent rate": lambda i, s: replace(i, rent_hourly=i.rent_hourly * s, rent_month=i.rent_month * s,
                                      rent_extra_hour=i.rent_extra_hour * s),
    "Salvage value": lambda i, s: replace(i, salvage_scale=i.salvage_scale * s),
    "Diesel price": lambda i, s: replace(i, diesel_price=i.diesel_price * s),
}


def buy_minus_rent(inp):
    return npv(buy_cashflows(inp), inp.discount_rate) - npv(rent_cashflows(inp), inp.discount_rate)


def sensitivity(inp, delta=0.20):
    """Buy-minus-rent NPV gap at -delta and +delta on each input, sorted by swing (largest first)."""
    base = buy_minus_rent(inp)
    rows = []
    for name, apply in SENSITIVITY_PARAMS.items():
        low, high = buy_minus_rent(apply(inp, 1 - delta)), buy_minus_rent(apply(inp, 1 + delta))
        rows.append({"input": name, "low": low, "high": high, "swing": abs(high - low)})
    rows.sort(key=lambda r: r["swing"], reverse=True)
    return base, rows
