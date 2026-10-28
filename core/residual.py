"""Resale-value retention curves fitted on the Kaggle Blue Book for Bulldozers data.

Only the *ratio* of resale price to near-new price is used, never US dollar values.
Model per product group, with machine-model fixed effects (within estimator):

    log(real price) = alpha_model + b1 * age + b2 * age^2 + e

    retention(a) = exp(f(a) - f(1)),  f(a) = b1*a + b2*a^2

so retention(1) = 1, i.e. S = P * retention(T) as in the project plan.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CURVE_PATH = DATA_DIR / "residual_curve.csv"
POINTS_PATH = DATA_DIR / "residual_points.csv"
# The raw Kaggle file is git-ignored; it may sit in data/ or in the repo root.
BLUEBOOK_CANDIDATES = (DATA_DIR / "TrainAndValid.csv", DATA_DIR.parent / "TrainAndValid.csv")

MAX_AGE = 25

# US CPI-U annual averages (BLS series CUUR0000SA0), used to put sale prices in 2012 dollars.
CPI_U = {
    1989: 124.0, 1990: 130.7, 1991: 136.2, 1992: 140.3, 1993: 144.5, 1994: 148.2,
    1995: 152.4, 1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6, 2000: 172.2,
    2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6,
    2007: 207.342, 2008: 215.303, 2009: 214.537, 2010: 218.056, 2011: 224.939, 2012: 229.594,
}

GROUP_NAMES = {
    "BL": "Backhoe Loaders",
    "TEX": "Track Excavators",
    "WL": "Wheel Loader",
    "MG": "Motor Graders",
    "TTT": "Track Type Tractors",
    "SSL": "Skid Steer Loaders",
}


def bluebook_path(csv_path=None):
    """Locate the raw Kaggle file: an explicit path, else data/, else the repo root."""
    if csv_path is not None:
        return csv_path
    for cand in BLUEBOOK_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "TrainAndValid.csv not found. Download it from the Kaggle Blue Book for Bulldozers "
        f"competition into {BLUEBOOK_CANDIDATES[0].parent} (see README)."
    )


def load_bluebook(csv_path=None):
    """Read the columns we need and build age, real log price and log hours."""
    csv_path = bluebook_path(csv_path)
    cols = ["SalePrice", "ModelID", "YearMade", "MachineHoursCurrentMeter", "saledate", "ProductGroup"]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    df["sale_year"] = pd.to_datetime(df["saledate"], format="%m/%d/%Y %H:%M").dt.year
    df["age"] = df["sale_year"] - df["YearMade"]
    # YearMade = 1000 is a placeholder for "unknown" in this dataset
    df = df[(df["YearMade"] > 1950) & df["age"].between(0, MAX_AGE) & (df["SalePrice"] > 0)].copy()
    df["real_price"] = df["SalePrice"] * CPI_U[2012] / df["sale_year"].map(CPI_U)
    df["log_p"] = np.log(df["real_price"])
    df["log_hours"] = np.log(df["MachineHoursCurrentMeter"].where(df["MachineHoursCurrentMeter"] > 0))
    return df


def _within_ols(g, x_cols):
    """OLS on model-demeaned data. Returns coefficients and their standard errors."""
    y = g["log_p"] - g.groupby("ModelID")["log_p"].transform("mean")
    X = np.column_stack([g[c] - g.groupby("ModelID")[c].transform("mean") for c in x_cols])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    resid = y.to_numpy() - X @ beta
    dof = len(g) - g["ModelID"].nunique() - len(x_cols)
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / (y.to_numpy() @ y.to_numpy())
    return beta, se, r2


def fit_group(g):
    """Fit one product group. Returns (curve rows, coefficient summary, model fixed effects)."""
    g = g.assign(age2=g["age"] ** 2)
    beta, se, r2 = _within_ols(g, ["age", "age2"])
    b1, b2 = beta

    ages = np.arange(0, MAX_AGE + 1)
    f = b1 * ages + b2 * ages**2
    retention = np.exp(f - (b1 + b2))
    # A quadratic can turn upward at old ages; resale value should never rise with age
    retention[1:] = np.minimum.accumulate(retention[1:])

    # Cross-check with hours, only on rows that have a positive meter reading
    gh = g.dropna(subset=["log_hours"])
    bh, seh, r2h = _within_ols(gh, ["age", "age2", "log_hours"]) if len(gh) > 1000 else ([np.nan] * 3, [np.nan] * 3, np.nan)

    summary = {
        "b_age": b1, "se_age": se[0], "b_age2": b2, "se_age2": se[1], "r2_within": r2, "n": len(g),
        "n_models": g["ModelID"].nunique(),
        "hours_b_age": bh[0], "hours_b_age2": bh[1], "hours_b_log_hours": bh[2], "hours_se_log_hours": seh[2],
        "hours_r2_within": r2h, "hours_n": len(gh),
    }
    alpha = (g["log_p"] - b1 * g["age"] - b2 * g["age2"]).groupby(g["ModelID"]).transform("mean")
    observed = np.exp(g["log_p"] - alpha - (b1 + b2))
    return retention, summary, observed


def fit_all(csv_path=None, seed=725):
    """Fit every product group. Returns (curve, summary, points) DataFrames."""
    df = load_bluebook(csv_path)
    curves, summaries, points = [], [], []
    rng = np.random.default_rng(seed)
    for grp, g in df.groupby("ProductGroup"):
        retention, summary, observed = fit_group(g)
        curves.append(pd.DataFrame({"group": grp, "age": np.arange(0, MAX_AGE + 1), "retention": retention}))
        summaries.append({"group": grp, "group_name": GROUP_NAMES.get(grp, grp), **summary})
        obs = pd.DataFrame({"group": grp, "age": g["age"].to_numpy(), "observed_retention": observed.to_numpy()})
        med = obs.groupby(["group", "age"])["observed_retention"].agg(["median", "count"]).reset_index()
        sample = obs.iloc[rng.choice(len(obs), size=min(400, len(obs)), replace=False)]
        points.append(med.assign(kind="median").rename(columns={"median": "observed_retention"}))
        points.append(sample.assign(kind="sample", count=1))
    return pd.concat(curves), pd.DataFrame(summaries), pd.concat(points, ignore_index=True)


def load_curve(path=CURVE_PATH):
    return pd.read_csv(path)


def retention(group, age, curve=None):
    """Retention ratio (resale / price at age 1) for a Blue Book group at a given age.

    Linear interpolation between whole years; ages beyond the fitted range use the last value.
    Returns None if the group is not in the curve (caller falls back to a fixed salvage %).
    """
    curve = load_curve() if curve is None else curve
    c = curve[curve["group"] == group]
    if c.empty:
        return None
    return float(np.interp(age, c["age"], c["retention"]))
