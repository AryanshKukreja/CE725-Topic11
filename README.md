# Buy vs Lease vs Rent Analyser (CE725, Topic 11)

A Streamlit dashboard that tells a contractor whether to buy, lease or rent one machine at the expected utilisation, compared on after-tax present value.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
pytest            # formula checks, edge cases, benchmark vs hand sheet
```

## Layout

| Path | What it does |
|---|---|
| `app.py` | Streamlit UI only (sidebar inputs, 7 tabs) |
| `core/costs.py` | Pre-tax hourly owning and operating (O&O) cost |
| `core/cashflows.py` | After-tax buy / lease / rent cash flows, NPV, EAC, break-even, sensitivity |
| `core/residual.py` | Blue Book retention-curve fit and lookup |
| `core/benchmark.py` | JCB 3DX benchmark inputs, tool vs hand comparison |
| `data/machines.csv` | Machine presets, each with a status and source |
| `data/sources.csv` | Every number with its source and access date |
| `data/residual_curve.csv` | Fitted retention by age per Blue Book group |
| `data/benchmark_hand.csv` | Hand-calculated benchmark values (Validation Engineer fills these) |
| `notebooks/01_bluebook_residual.ipynb` | The Blue Book fit (runs in Colab) |

## Blue Book data

`TrainAndValid.csv` is git-ignored because it is too large for GitHub. To refit the curves, download it from the [Kaggle competition](https://www.kaggle.com/c/bluebook-for-bulldozers/data) into `data/` (the repo root also works) and run the notebook. The app itself needs only the small fitted CSVs in `data/`.

## Model conventions

- Costs are positive and lower NPV is better. Periods are annual with payments at year end; the purchase falls at year 0. Monthly loan EMIs are carried forward to year end at the discount rate.
- Tax: all running costs and rentals are deductible. The buyer gets a 15% WDV depreciation shield and pays tax on (sale price − closing WDV). The asset is treated singly: no block pooling and no 180-day rule.
- Repairs: repair factor × price spread over life hours (MoRTH form), so repairs per hour stay fixed when utilisation changes.
- Horizon longer than life: the machine is sold and replaced at the same price. Lease shorter than horizon: renewed at the escalated rental.
- Monthly rent: you hire only the months you need (hours ÷ hours included per month, up to 12). Hours beyond 12 full months are billed at the extra-hour rate.
- Lease: by default modelled as a finance lease (capital recovery at loan rate + spread), with the lessee paying upkeep.
