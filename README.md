# NYISO Energy Demand Forecasting — Phase 1

Day-ahead forecasting of electricity demand for the **New York Independent
System Operator**, the balancing authority that keeps supply and demand matched
across New York State's grid.

A grid operator has to commit generation *before* it is needed. Commit too
little and you buy emergency power at punishing prices or shed load; commit too
much and you have paid to spin turbines nobody used. Peak demand sizes that
decision, so a day-ahead peak forecast is one of the load-bearing numbers in
grid operations.

**Phase 1 proves out the ML core** — data → features → model → experiment
tracking — end-to-end on a laptop, before any AWS or pipeline infrastructure is
wired up. Everything lives in one notebook on purpose.

---

## Results

Trained on 2019-01-15 → 2026-07-05, validated on the held-out
2026-07-06 → 2026-08-30 window (56 days).

| Metric | Persistence baseline | XGBoost | Improvement |
|---|---:|---:|---:|
| MAE  | 1,137.3 MW | **747.4 MW** | 34.3% |
| RMSE | 1,616.2 MW | **914.9 MW** | 43.4% |
| MAPE | 4.85% | **3.21%** | 33.9% |

The bar here is not a low MAE — it is **beating persistence** ("tomorrow looks
like today"), which is free to compute and hard to beat in a highly
autocorrelated series. `peak_lag_1` is also the model's single most important
feature (53% of gain), so without this comparison a strong-looking score could
just be the model echoing yesterday's value back.

Residual diagnostics: mean **+328.7 MW** (a mild systematic over-forecast),
std 861.5 MW, lag-1 autocorrelation +0.013 (no exploitable structure left).
Worst single day: 2026-07-16, 2,380 MW off.

---

## Layout

```
.
├── nyiso_phase1_eda_training.ipynb   # the entire project — 46 cells, §0–10
├── nyiso_phase1_eda_training.ipynb.bak
├── data/
│   ├── nyiso_daily_features.csv      # 2,785 rows × 19 cols, the modelling table
│   └── nyiso_weekly_demand.csv       # 401 weeks, second target (not yet modelled)
├── mlflow.db                         # MLflow tracking store (SQLite)
├── mlruns/                           # logged model artifacts
├── requirements.txt
└── .env                              # EIA_API_KEY, NOAA_TOKEN — never commit
```

## Setup

```bash
python3.11 -m venv env
./env/bin/pip install -r requirements.txt
```

Two free API credentials are needed. Both arrive instantly by email:

- **EIA API key** — https://www.eia.gov/opendata/register.php
- **NOAA NCEI CDO token** — https://www.ncdc.noaa.gov/cdo-web/token

Put them in a `.env` file next to the notebook (the notebook ships a minimal
loader, so `python-dotenv` is not required):

```
EIA_API_KEY=...
NOAA_TOKEN=...
```

Then:

```bash
./env/bin/jupyter lab nyiso_phase1_eda_training.ipynb
```

Run top to bottom. Sections 1 and 3 hit both APIs and take a few minutes;
everything after §9 reads from `data/` instead.

To browse the tracked runs:

```bash
./env/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Notebook map

| § | Contents |
|---|---|
| 0–1 | Setup, chart style, EIA hourly demand pull |
| 2–2b | Aggregation to targets, data-quality check, demand EDA |
| 3–4 | NOAA weather, merge, temperature response curve, feature engineering |
| 5 | Chronological train/validation split |
| 6–6b | XGBoost + MLflow, **persistence baseline comparison** |
| 7–8 | Predictions, residual diagnostics, error breakdown, feature importance |
| 9–10 | Persist processed data, limitations, next steps |

### Data sources

- [EIA API v2](https://www.eia.gov/opendata/) — hourly demand, NYISO
  (`respondent=NYIS`, `type=D`). 67,244 rows, 2019-01-01 → 2026-09-02, with 2
  missing hours and 1 duplicate timestamp across the whole series.
- [NOAA NCEI Climate Data Online](https://www.ncei.noaa.gov/cdo-web/webservices/v2)
  — daily TMAX/TMIN/PRCP for LaGuardia Airport (`GHCND:USW00014732`).

Both APIs paginate: EIA caps at 5,000 rows per call (looped over 200-day
windows), CDO at 1,000 records and a 1-year date range (looped year by year).

### Targets

| Target | Definition | Status |
|---|---|---|
| `daily_peak_mw` | Max hourly demand per day | **modelled** |
| `weekly_total_mwh` | Sum of hourly demand per week | computed and saved, **not yet modelled** |

### Features (17)

| Feature | Type | Definition | Why it's here |
|---|---|---|---|
| `temp_max_f`, `temp_min_f`, `temp_avg_f` | weather | Daily LaGuardia temps (°F) | Primary physical driver of load |
| `precip_in` | weather | Daily precipitation (in) | Weak, but cloud cover shifts lighting/AC load |
| `hdd`, `cdd` | weather | `max(0, 65−T)`, `max(0, T−65)` | Linearises each arm of the U-curve |
| `temp_lag_1` | weather | Yesterday's mean temp | Buildings have thermal mass; heat carries over |
| `day_of_week`, `is_weekend` | calendar | 0–6, and weekend flag | Commercial/industrial load collapses at weekends |
| `month`, `day_of_year` | calendar | Seasonal position | Daylight and non-temperature seasonality |
| `is_holiday` | calendar | US federal holiday flag | Holidays behave like weekends regardless of weekday |
| `peak_lag_1/7/14` | autoregressive | Peak 1/7/14 days ago | Load is persistent; lag-7 preserves day-of-week phase |
| `peak_roll_mean_7/14` | autoregressive | Trailing means, **shifted 1 day** | Recent level without leaking today's value |

Load against temperature is a **U** — both cold and heat create load, through
different appliances — so a single `temp_avg_f` coefficient would fit the bottom
of the curve and miss both arms. Splitting into HDD/CDD at a 65°F balance point
lets the model put a different slope on each side. Measured: hot days (>78°F)
peak **+50%** above mild days, cold days (<40°F) only **+21%** — NYISO is a
summer-peaking system.

**On leakage:** every autoregressive feature is `.shift(1)` before any rolling
window, so no feature contains information from the day being predicted. The
split is chronological for the same reason — shuffling would let the model see
next week while predicting last week, via `peak_lag_1` alone.

---

## Limitations

Stated plainly, because a forecast with unstated assumptions is a liability.

1. **This is a day-ahead model, not a general forecaster.** `peak_lag_1` assumes
   yesterday's actual peak is known. Longer horizons need that feature removed
   and the whole thing re-scored — expect materially worse numbers.
2. **Validation is a single contiguous window**, landing in cooling season, so
   3.21% MAPE describes summer performance, not the year. A rolling-origin
   backtest would give an honest annual figure.
3. **Timestamps are UTC.** The EIA pull uses `frequency="hourly"` (UTC) and never
   localises, so a "day" runs 19:00–19:00 local in winter. The evening peak sits
   near that boundary, so a minority of days book their peak against the wrong
   calendar date. `frequency="local-hourly"` is the fix and should land before
   any of these numbers are quoted as operational.
4. **One weather station stands in for a state-wide system.** NYISO spans all of
   New York; LaGuardia is one point in the warmest, densest corner. Upstate
   heating load is effectively unmodelled.
5. **No behind-the-meter solar term.** Distributed PV suppresses measured demand
   and has grown materially since 2019, putting a slow non-stationary trend in
   the target that none of these features can see.
6. **EIA demand figures are revised**, so the most recent rows are softer than
   the older ones.
7. **`weekly_total_mwh` is computed and saved but never modelled** — only half
   the stated target set is delivered here.

---

## Next steps (Phase 2+)

- [ ] Repeat §5–8 for the **weekly** target (`weekly_total_mwh`)
- [ ] Switch the EIA pull to `frequency="local-hourly"` and re-score (limitation 3)
- [ ] Extract §1–4 into a standalone `preprocess.py` and §5–6 into `train.py`
- [ ] Rolling-origin backtest across several folds for an annual error figure
- [ ] Point the MLflow tracking URI at a persistent server, not local `mlruns/`
- [ ] Register the first model in the MLflow / SageMaker Model Registry as the
      "champion" for future runs to beat

> **Note on the notebook's own "Next steps" cell:** it currently checks off
> `scripts/nyiso_preprocess.py`, `scripts/nyiso_train.py` and `make` targets
> that do not exist in this directory, and quotes MAE 731.9 MW / MAPE 3.11% /
> 38.7% baseline improvement, which disagrees with the notebook's own stored
> outputs (747.4 MW / 3.21% / 34.3%). The numbers in this README come from the
> executed cells and the MLflow store. That cell needs reconciling.
