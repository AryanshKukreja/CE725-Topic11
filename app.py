"""Buy vs Lease vs Rent analyser - Streamlit UI only. All maths lives in core/."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.benchmark import validation_table
from core.cashflows import (Inputs, break_even_hours, break_even_simple, compare, discount_factors,
                            finance_lease_rental, sensitivity)
from core.costs import hourly_cost
from core.residual import GROUP_NAMES, POINTS_PATH, load_curve, retention

DATA = Path(__file__).parent / "data"
# Reference categorical palette, slots 1-8 in fixed order (first three validated all-pairs)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OPTION_COLOR = {"Buy": SERIES[0], "Lease": SERIES[1], "Rent": SERIES[2]}

st.set_page_config(page_title="Buy vs Lease vs Rent", layout="wide")


def inr(x):
    """Rupees in lakh for big numbers, plain for small ones."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"₹{x / 1e5:,.2f} lakh" if abs(x) >= 1e5 else f"₹{x:,.0f}"


def style(fig, **kw):
    layout = dict(margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
                  legend=dict(orientation="h", y=1.08, x=0))
    fig.update_layout(**{**layout, **kw})
    return fig


@st.cache_data
def load_machines():
    return pd.read_csv(DATA / "machines.csv")


@st.cache_data
def cached_curve():
    return load_curve()


machines, curve = load_machines(), cached_curve()

# ---------------- sidebar ----------------
sb = st.sidebar
sb.title("Inputs")
name = sb.selectbox("Machine", machines["machine"])
m = machines.set_index("machine").loc[name]
sb.caption(f"Data status: **{m['status']}**: {m['source']}")
k = lambda field: f"{name}:{field}"  # per-machine widget keys so presets reload on machine change

sb.subheader("Machine")
price = sb.number_input("Purchase price (₹)", value=float(m["price"]), step=50_000.0, key=k("price"))
life = sb.number_input("Life (years)", 1, 30, int(m["life_years"]), key=k("life"))
hours = sb.number_input("Utilisation (hours/year)", 50, 6000, int(m["hours_per_year"]), step=50, key=k("hours"))
group = m["bluebook_group"] if isinstance(m["bluebook_group"], str) else None
use_curve = sb.checkbox("Salvage from Blue Book curve", value=group is not None, disabled=group is None, key=k("curve"))
if use_curve:
    residual = lambda a: retention(group, a, curve)
    sb.caption(f"Retention at {life} yrs: {retention(group, life, curve):.1%} ({GROUP_NAMES[group]})")
else:
    salv_pct = sb.number_input("Salvage at end of life (% of price)", 0.0, 100.0,
                               float(m["salvage_pct_fallback"]) * 100, key=k("salv")) / 100
    # straight line from 100% at age 0 down to salv_pct at end of life, then flat
    residual = lambda a, n=life, s=salv_pct: max(s, 1 - (1 - s) * a / n)

sb.subheader("Finance and tax")
horizon = sb.number_input("Analysis horizon (years)", 1, 30, 5)
r = sb.number_input("Discount rate (%)", 0.0, 40.0, 11.0, 0.5) / 100
tax = sb.number_input("Tax rate (%)", 0.0, 50.0, 25.17, 0.01) / 100
wdv = sb.number_input("Tax depreciation, WDV (%)", 0.0, 100.0, 15.0) / 100
loan_frac = sb.slider("Loan share of price (%)", 0, 100, 0) / 100
loan_rate = sb.number_input("Loan rate (%)", 0.0, 40.0, 11.0, 0.25) / 100
loan_years = sb.number_input("Loan tenure (years)", 1, 15, 5)
ins = sb.number_input("Insurance (% of price per year)", 0.0, 10.0, 1.0, 0.1) / 100

sb.subheader("Operating")
fuel = sb.number_input("Fuel use (L/hr)", 0.0, 100.0, float(m["fuel_lph"]), 0.5, key=k("fuel"))
diesel = sb.number_input("Diesel price (₹/L)", 0.0, 300.0, 97.83, 0.5)
lube = sb.number_input("Lubricants (% of fuel cost)", 0.0, 100.0, 25.0) / 100
repair = sb.number_input("Repair factor (× price over life hours)", 0.0, 5.0, float(m["repair_factor"]), 0.1, key=k("rep"))
tyre_cost = sb.number_input("Tyre set cost (₹)", 0.0, value=float(m["tyre_cost"]), step=10_000.0, key=k("tyre"))
tyre_life = sb.number_input("Tyre life (hours)", 0.0, value=float(m["tyre_life_h"]), step=250.0, key=k("tyrel"))
wage = sb.number_input("Operator wage (₹/month)", 0.0, value=25_000.0, step=1_000.0)
op_hours = sb.number_input("Operator hours per month", 1.0, 744.0, 200.0, 10.0)

sb.subheader("Lease")
lease_term = sb.number_input("Lease term (years)", 1, 15, 5)
lessor_repairs = sb.checkbox("Lessor covers repairs", value=False,
                             help="Off by default: per our definition the lessee pays routine upkeep. "
                                  "Only tick if the quote is a maintained (operating) lease.")
fin_lease = sb.checkbox("Model rental as finance lease", value=True,
                        help="EMI on price less the PV of residual, at loan rate + spread. Untick to enter a quote.")
if fin_lease:
    spread = sb.number_input("Lessor spread over loan rate (%)", 0.0, 20.0, 2.0, 0.25) / 100
    resid_value = price * residual(lease_term)
    lease_rental = finance_lease_rental(price, resid_value, loan_rate + spread, lease_term * 12)
    sb.caption(f"Modelled rental: {inr(lease_rental)}/month")
else:
    lease_rental = sb.number_input("Lease rental (₹/month)", 0.0, value=50_000.0, step=1_000.0)

sb.subheader("Rent")
rent_mode = sb.radio("Rent priced", ["monthly", "hourly"], index=0 if m["rent_mode"] == "monthly" else 1,
                     horizontal=True, key=k("rmode"))
if rent_mode == "monthly":
    rent_month = sb.number_input("Rent (₹/month)", 0.0, value=float(m["rent_month"]), step=5_000.0, key=k("rm"))
    rent_month_hours = sb.number_input("Hours included per month", 1.0, 744.0, float(m["rent_month_hours"]), key=k("rmh"))
    rent_extra = sb.number_input("Extra hours (₹/hr)", 0.0, value=float(m["rent_extra_hour"]), key=k("rx"))
    rent_hourly = 0.0
else:
    rent_hourly = sb.number_input("Rent (₹/hr)", 0.0, value=float(m["rent_hourly"]), step=50.0, key=k("rh"))
    rent_month, rent_month_hours, rent_extra = 0.0, 240.0, 0.0
wet = sb.checkbox("Rent includes operator (wet hire)", value=bool(m["rent_includes_operator"]), key=k("wet"))
rent_fuel = sb.checkbox("Rent includes fuel", value=bool(m["rent_includes_fuel"]), key=k("rfuel"))
esc = sb.number_input("Rent and lease escalation (%/yr)", 0.0, 30.0, 0.0, 0.5) / 100

inp = Inputs(
    price=price, life_years=int(life), hours_per_year=float(hours), residual=residual,
    insurance_rate=ins, discount_rate=r, tax_rate=tax, wdv_rate=wdv, loan_fraction=loan_frac,
    loan_rate=loan_rate, loan_years=int(loan_years), horizon_years=int(horizon),
    fuel_lph=fuel, diesel_price=diesel, lube_frac=lube, repair_factor=repair, tyre_cost=tyre_cost,
    tyre_life_h=tyre_life, operator_wage_month=wage, operator_hours_month=op_hours,
    lease_rental_month=lease_rental, lease_term_years=int(lease_term), lessor_covers_repairs=lessor_repairs,
    rent_mode=rent_mode, rent_hourly=rent_hourly, rent_month=rent_month, rent_month_hours=rent_month_hours,
    rent_extra_hour=rent_extra, rent_includes_operator=wet, rent_includes_fuel=rent_fuel, rent_escalation=esc,
)

# ---------------- results ----------------
res = compare(inp)
best = min(res, key=lambda o: res[o]["npv"])
h_star = break_even_hours(inp)
h_star_lease = break_even_hours(inp, "Buy", "Lease")
base_gap, tornado = sensitivity(inp)

st.title("Buy vs Lease vs Rent")
st.caption(f"{name} · {hours:,} hours/year · {horizon}-year horizon · after-tax, costs as positive numbers")

tabs = st.tabs(["Hourly cost", "Buy vs Lease vs Rent", "Break-even", "Sensitivity",
                "Resale curve", "Validation", "Recommendation"])

with tabs[0]:
    rows, aai = hourly_cost(inp.hourly_dict())
    total = sum(rows.values())
    st.metric("Owning and operating cost (pre-tax)", f"₹{total:,.0f} per hour")
    fig = go.Figure()
    for i, (comp, val) in enumerate(rows.items()):
        fig.add_bar(name=comp, x=[val], y=["Buy"], orientation="h", marker_color=SERIES[i],
                    marker_line=dict(width=2, color="rgba(255,255,255,0.9)"),
                    hovertemplate=f"{comp}: ₹%{{x:,.0f}}/hr<extra></extra>")
    style(fig, barmode="stack", height=230, xaxis_title="₹ per hour", showlegend=True)
    st.plotly_chart(fig, width="stretch")
    tbl = pd.DataFrame({"Component": list(rows), "₹/hr": list(rows.values())})
    tbl["Share"] = tbl["₹/hr"] / total
    st.dataframe(tbl.style.format({"₹/hr": "{:,.1f}", "Share": "{:.1%}"}), hide_index=True, width="stretch")
    S = inp.salvage_at(life)
    st.caption(f"Salvage at {life} yrs: {inr(S)} · average annual investment {inr(aai)} · "
               f"repairs spread over {inp.life_hours:,.0f} life hours")

with tabs[1]:
    cols = st.columns(3)
    for col, (opt, v) in zip(cols, res.items()):
        label = f"{opt} (lowest cost)" if opt == best else opt
        delta = None if opt == best else f"+{inr(v['npv'] - res[best]['npv'])} vs {best}"
        col.metric(label, inr(v["npv"]), delta, delta_color="inverse")
        col.caption(f"EAC {inr(v['eac'])}/yr · ₹{v['eac_per_hour']:,.0f} per hour")
    df_ = discount_factors(r, horizon)
    fig = go.Figure()
    for opt, v in res.items():
        fig.add_scatter(x=list(range(horizon + 1)), y=np.cumsum(v["cashflows"] * df_), name=opt, mode="lines+markers",
                        line=dict(color=OPTION_COLOR[opt], width=2), marker=dict(size=8),
                        hovertemplate=f"{opt}, year %{{x}}: ₹%{{y:,.0f}}<extra></extra>")
    style(fig, height=380, title="Cumulative discounted after-tax cost", xaxis_title="Year",
          yaxis_title="₹", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    cf = pd.DataFrame({opt: v["cashflows"] for opt, v in res.items()}, index=pd.Index(range(horizon + 1), name="Year"))
    with st.expander("Year-by-year after-tax cash flows (₹)"):
        st.dataframe(cf.style.format("{:,.0f}"), width="stretch")

with tabs[2]:
    grid = np.linspace(100, max(4000, hours * 2), 60)
    fig = go.Figure()
    for opt in res:
        y = [compare(inp.with_hours(h))[opt]["eac_per_hour"] for h in grid]
        fig.add_scatter(x=grid, y=y, name=opt, mode="lines", line=dict(color=OPTION_COLOR[opt], width=2),
                        hovertemplate=f"{opt}: ₹%{{y:,.0f}}/hr at %{{x:,.0f}} h/yr<extra></extra>")
    fig.add_vline(x=hours, line=dict(dash="dot", width=1, color="gray"),
                  annotation_text="your utilisation", annotation_position="top left")
    if h_star:
        fig.add_vline(x=h_star, line=dict(width=1.5, color="gray"),
                      annotation_text=f"H* = {h_star:,.0f} h/yr", annotation_position="top right")
    style(fig, height=420, title="After-tax cost per productive hour (EAC ÷ hours)",
          xaxis_title="Hours per year", yaxis_title="₹ per hour", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Buy = Rent at (after-tax NPV)", f"{h_star:,.0f} h/yr" if h_star else "no crossing")
    c2.metric("Buy = Lease at", f"{h_star_lease:,.0f} h/yr" if h_star_lease else "no crossing")
    simple = break_even_simple(inp)
    c3.metric("Textbook H* = F/(R − v), pre-tax", f"{simple:,.0f} h/yr" if simple else "no crossing")
    if not h_star:
        st.info("Buying and renting never cost the same between 10 and 6,000 hours/year. This usually means "
                "the rent per hour is below the owner's variable cost per hour (fuel, repairs, tyres, "
                "operator), so renting wins at any utilisation. Check the rent quote and repair factor.")

with tabs[3]:
    fig = go.Figure()
    names = [t["input"] for t in tornado][::-1]
    fig.add_bar(y=names, x=[t["low"] - base_gap for t in tornado][::-1], base=base_gap, orientation="h",
                name="Input −20%", marker_color=SERIES[0],
                customdata=[t["low"] for t in tornado][::-1],
                hovertemplate="%{y} −20%: gap ₹%{customdata:,.0f}<extra></extra>")
    fig.add_bar(y=names, x=[t["high"] - base_gap for t in tornado][::-1], base=base_gap, orientation="h",
                name="Input +20%", marker_color=SERIES[1],
                customdata=[t["high"] for t in tornado][::-1],
                hovertemplate="%{y} +20%: gap ₹%{customdata:,.0f}<extra></extra>")
    fig.add_vline(x=base_gap, line=dict(width=1, color="gray"))
    if min(min(t["low"], t["high"]) for t in tornado) < 0 < max(max(t["low"], t["high"]) for t in tornado):
        fig.add_vline(x=0, line=dict(width=1, dash="dot", color="gray"), annotation_text="decision flips")
    style(fig, barmode="overlay", height=380, title="Buy minus Rent NPV (positive = renting is cheaper)",
          xaxis_title="₹")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Base gap {inr(base_gap)}. Diesel shows little swing because fuel is paid under every option "
               "unless the rent includes fuel.")
    st.dataframe(pd.DataFrame(tornado).style.format({"low": "{:,.0f}", "high": "{:,.0f}", "swing": "{:,.0f}"}),
                 hide_index=True, width="stretch")

with tabs[4]:
    if group is None:
        st.info(f"No Blue Book product group matches **{name}**, so salvage uses the fixed % in the sidebar.")
    else:
        pts = pd.read_csv(POINTS_PATH)
        pts = pts[pts["group"] == group]
        c = curve[curve["group"] == group]
        fig = go.Figure()
        s = pts[pts["kind"] == "sample"]
        fig.add_scatter(x=s["age"] + np.random.default_rng(0).uniform(-0.3, 0.3, len(s)), y=s["observed_retention"],
                        mode="markers", name="Individual sales (sample)", marker=dict(size=8, color=SERIES[0], opacity=0.25),
                        hovertemplate="age %{x:.0f}: %{y:.0%}<extra></extra>")
        med = pts[pts["kind"] == "median"]
        fig.add_scatter(x=med["age"], y=med["observed_retention"], mode="markers", name="Median by age",
                        marker=dict(size=9, color=SERIES[1], line=dict(width=2, color="white")),
                        customdata=med["count"], hovertemplate="age %{x}: median %{y:.0%} (n=%{customdata:,})<extra></extra>")
        fig.add_scatter(x=c["age"], y=c["retention"], mode="lines", name="Fitted curve",
                        line=dict(color=SERIES[2], width=2), hovertemplate="age %{x}: %{y:.1%}<extra></extra>")
        fig.add_vline(x=horizon, line=dict(dash="dot", width=1, color="gray"), annotation_text="horizon")
        style(fig, height=430, title=f"Resale ÷ price at age 1, inflation-adjusted: {GROUP_NAMES[group]}",
              xaxis_title="Age (years)", yaxis_title="Retention", yaxis_tickformat=".0%", yaxis_range=[0, 1.6])
        st.plotly_chart(fig, width="stretch")
        summ = pd.read_csv(DATA / "residual_fit_summary.csv").set_index("group").loc[group]
        st.caption(f"log(real price) = model effect + {summ.b_age:.4f}·age + {summ.b_age2:.5f}·age², "
                   f"within-R² {summ.r2_within:.2f}, n = {int(summ.n):,} sales of {int(summ.n_models)} models. "
                   "Only ratios are used; US dollar values never enter the tool.")

with tabs[5]:
    st.subheader("JCB 3DX benchmark: tool vs hand calculation")
    st.caption("Hand values come from data/benchmark_hand.csv (Validation Engineer's spreadsheet). "
               "Pass rule: within 0.5%. This uses fixed benchmark inputs, not the sidebar.")
    vt = validation_table()
    st.dataframe(vt.style.format({"hand_value": "{:,.2f}", "tool_value": "{:,.2f}", "diff_pct": "{:+.2f}%"}, na_rep="-"),
                 hide_index=True, width="stretch")
    done = vt[vt["result"] != "pending"]
    st.write(f"{(done['result'] == 'pass').sum()} of {len(done)} filled lines pass; {len(vt) - len(done)} pending.")

with tabs[6]:
    other = sorted((o for o in res if o != best), key=lambda o: res[o]["npv"])
    margin = res[other[0]]["npv"] - res[best]["npv"]
    top = tornado[0]
    flips = (top["low"] < 0 < top["high"]) or (top["high"] < 0 < top["low"])
    st.subheader(f"Recommendation: {best.lower()} the {name.split(' (')[0].lower()}")
    lines = [
        f"At **{hours:,} hours/year** over **{horizon} years**, **{best}** has the lowest after-tax cost: "
        f"{inr(res[best]['npv'])} in present value, or ₹{res[best]['eac_per_hour']:,.0f} per working hour.",
        f"The next best option is **{other[0]}**, which costs {inr(margin)} more "
        f"({margin / res[best]['npv']:.0%}).",
    ]
    if h_star:
        side = "above" if hours > h_star else "below"
        lines.append(f"Buying and renting cost the same at about **{h_star:,.0f} hours/year**. Your utilisation is {side} that.")
    else:
        lines.append("Buying never becomes cheaper than renting at any utilisation with these inputs.")
    lines.append(f"The input that moves the buy-vs-rent gap most is **{top['input'].lower()}** "
                 f"(±20% shifts it by {inr(top['swing'] / 2)} each way)"
                 + (", and that is enough to flip the decision." if flips else ", and the decision holds across that range."))
    for line in lines:
        st.markdown(f"- {line}")
    st.caption("Rule-based summary, no LLM. Excludes GST credit, fleet effects and productivity per m³.")
