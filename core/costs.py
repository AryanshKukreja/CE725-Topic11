"""Pre-tax hourly ownership and operating (O&O) cost, Peurifoy / USACE style.

Symbols follow the method note: P = delivered price, S = salvage, n = life (years),
H = hours/year. All money in rupees.
"""


def average_annual_investment(P, S, n):
    """AAI = [P(n+1) + S(n-1)] / 2n"""
    return (P * (n + 1) + S * (n - 1)) / (2 * n)


def ownership_cost_per_hour(P, S, n, H, interest, insurance, prop_tax):
    """Depreciation and interest-insurance-tax (IIT) per hour."""
    depreciation = (P - S) / (n * H)
    aai = average_annual_investment(P, S, n)
    iit = (interest + insurance + prop_tax) * aai / H
    return {"depreciation": depreciation, "interest_insurance_tax": iit, "aai": aai}


def operating_cost_per_hour(P, life_hours, fuel_lph, diesel_price, lube_frac, repair_factor,
                            tyre_cost, tyre_life_h, operator_wage_month, operator_hours_month):
    """Fuel, lubricants, repairs, tyres and operator per hour.

    Repairs use the MoRTH hire-charge form: repair_factor x P spread over life hours (n x H at
    the design utilisation), so the repair cost per hour stays fixed when utilisation changes.
    Operator: monthly wage / productive hours per month.
    """
    fuel = fuel_lph * diesel_price
    return {
        "fuel": fuel,
        "lubricants": lube_frac * fuel,
        "repairs": repair_factor * P / life_hours,
        "tyres": tyre_cost / tyre_life_h if tyre_life_h > 0 else 0.0,
        "operator": operator_wage_month / operator_hours_month,
    }


def hourly_cost(m):
    """Full hourly O&O breakdown for an inputs dict `m` (see core.cashflows.Inputs.hourly_dict)."""
    own = ownership_cost_per_hour(m["price"], m["salvage"], m["life_years"], m["hours_per_year"],
                                  m["discount_rate"], m["insurance_rate"], m["prop_tax_rate"])
    op = operating_cost_per_hour(m["price"], m["life_hours"], m["fuel_lph"],
                                 m["diesel_price"], m["lube_frac"], m["repair_factor"], m["tyre_cost"],
                                 m["tyre_life_h"], m["operator_wage_month"], m["operator_hours_month"])
    rows = {
        "Depreciation": own["depreciation"],
        "Interest, insurance, tax": own["interest_insurance_tax"],
        "Fuel": op["fuel"],
        "Lubricants": op["lubricants"],
        "Repairs & maintenance": op["repairs"],
        "Tyres": op["tyres"],
        "Operator": op["operator"],
    }
    return rows, own["aai"]
