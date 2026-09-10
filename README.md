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
├── nyiso_phase1_eda_training.ipynb   # the EDA record — 46 cells, §0–10
├── scripts/
│   ├── config.py                     # shared paths, FEATURE_COLS, balance point
│   ├── ingest.py                     # EIA + NOAA → data/raw/*.parquet (incremental)
│   ├── preprocess.py                 # raw → features + weekly totals
│   └── train.py                      # split, XGBoost, MLflow, champion gate
│   └── export_model.py               # champion run → artifacts/ for serving
├── serving/
│   ├── handler.py                    # Lambda inference (numpy only, no pandas)
│   └── build.sh                      # builds lambda.zip, pins xgboost to metadata
├── infra/                            # Terraform: S3 + Lambda + API Gateway + IAM
├── dags/nyiso_pipeline.py            # Airflow DAG wiring the three together
├── docker-compose.yml                # local Airflow (postgres + scheduler + API)
├── Dockerfile.airflow
├── data/
│   ├── raw/                          # parquet ingest cache (gitignored)
│   ├── nyiso_daily_features.csv      # the modelling table
│   └── nyiso_weekly_demand.csv       # second target (not yet modelled)
├── mlflow.db, mlruns/                # tracking store + artifacts (gitignored)
├── requirements.txt                  # full set, incl. notebook
├── requirements-pipeline.txt         # scripts only — no Jupyter/matplotlib
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

## Running the pipeline

The notebook is the exploratory record. The reproducible path is three scripts,
split at the raw/processed boundary so that iterating on features never re-pulls
seven years from two APIs.

```bash
python scripts/ingest.py       # EIA + NOAA → data/raw/*.parquet
python scripts/preprocess.py   # raw → data/nyiso_daily_features.csv
python scripts/train.py        # split, fit, log to MLflow
```

**Ingest is incremental.** It reads the high-water mark from the parquet and
requests only what is missing, *minus* a 7-day window, because EIA revises
recently published figures — a pure append would freeze the first value it ever
saw. Fresh rows win on overlap.

| | Time |
|---|---:|
| Cold pull, 2019 → today | 45s |
| Incremental re-run | 1.7s |

`--full` forces a complete refetch, `--source eia|noaa` runs one side.

**Preprocess is pure** — no network, no keys. `--check` compares its output
against the committed CSV instead of overwriting it, which is the regression
guard when you touch `add_features`:

```bash
$ python scripts/preprocess.py --check
--check: 2,785 overlapping rows, 19 columns
largest absolute difference: 0
MATCH
```

**Training is champion-gated.** Metrics and params are logged every run because
they cost nothing; the ~750 KB booster is written only when the run beats the
best `val_mae` so far. A nightly retrain that always logged would add ~268 MB a
year to say the same thing 365 times. `--as-of DATE` reproduces a past split;
`--always-log-model` bypasses the gate.

### Orchestration

`dags/nyiso_pipeline.py` runs the three daily at 11:00 UTC (07:00 EDT, after EIA
publishes), with both ingests in parallel and 2 retries per task — the APIs are
free and public and occasionally return 5xx.

```
ingest_eia  ──┐
              ├──> preprocess ──> train
ingest_noaa ──┘
```

```bash
cp .env.airflow.example .env.airflow    # add your API keys
docker compose --env-file .env.airflow up -d
open http://localhost:8080              # airflow / airflow
```

`catchup` is off deliberately: ingest is incremental against a high-water mark,
so backfilled runs would just refetch the same window.

---

## Serving the model (AWS, mocked locally)

The trained model is served behind an HTTP API. The AWS shape is real - S3,
Lambda, API Gateway, IAM - but it runs against
[LocalStack](https://localstack.cloud), so it costs nothing and needs no AWS
account. One variable switches the same Terraform to real AWS.

```
        +-------------+
POST -> | API Gateway |  public HTTPS
        +------+------+
               |
        +------v------+      +-----+
        |   Lambda    | ---> | S3  |  model.ubj + recent.json + metadata.json
        |  (python)   |      +-----+
        +-------------+
```

```bash
docker compose -f docker-compose.localstack.yml up -d   # free local AWS
python scripts/export_model.py                          # champion -> artifacts/
./serving/build.sh                                      # -> serving/lambda.zip
cd infra && terraform init && terraform apply -auto-approve
```

```bash
URL=$(terraform output -raw api_url)

curl -s "$URL/health"

curl -s -X POST "$URL/predict" -H 'Content-Type: application/json' \
  -d '{"date":"2026-09-06","temp_max_f":92,"temp_min_f":74,"precip_in":0.0}'
```
```json
{"date": "2026-09-06", "predicted_peak_mw": 24985.1,
 "baseline_persistence_mw": 22910.0, "delta_vs_baseline_mw": 2075.1,
 "model_run_id": "805fe2b3...", "model_val_mae": 747.4025}
```

The response carries the persistence baseline next to the prediction, so a
caller can see the model's contribution rather than taking the number on trust.
Sweeping temperature reproduces the U-curve end to end:

| Conditions | Predicted peak |
|---|---:|
| hot 92/74F | 24,985 MW |
| warm 78/62F | 20,370 MW |
| mild 68/55F | 19,908 MW |
| cold 34/22F | 21,351 MW |

### Going to real AWS

```bash
terraform apply -var use_localstack=false
```

Nothing else changes. Cost at portfolio traffic is a few cents a month: S3
storage, Lambda's free tier, API Gateway at $1/million requests. There is no
always-on compute - deliberately no SageMaker endpoint, which would be ~$50 a
month to answer the same question.

### Two things that bite here

**xgboost versions must match exactly between training and serving.** An older
xgboost reading a newer model file silently drops `base_score` and returns raw
tree sums - predictions near zero instead of ~20,000 MW, with no error raised.
This happened during development. The defences: `export_model.py` records
`xgboost_version` in `metadata.json`, `build.sh` pins the zip to it, and the
handler refuses to load on a mismatch.

**The Lambda zip runs close to the size limit.** xgboost pulls in scipy and the
package lands at ~194 MB unzipped against a 250 MB ceiling. pandas would not
fit, which is why the handler builds its feature vector with plain numpy. Note
also that `serving/build.sh` targets `manylinux_2_28`, not `manylinux2014` -
the older tag silently resolves xgboost down to 3.0.5.

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
- [x] Extract §1–4 into `scripts/ingest.py` + `scripts/preprocess.py`, §5–6b into `scripts/train.py`
- [x] Orchestrate the three as an Airflow DAG (`dags/nyiso_pipeline.py`)
- [ ] Rolling-origin backtest across several folds for an annual error figure
- [x] Serve the model over HTTP (S3 + Lambda + API Gateway, LocalStack-backed)
- [x] Serve the model over HTTP (S3 + Lambda + API Gateway, LocalStack-backed)
- [ ] Point the MLflow tracking URI at a persistent server, not local `mlruns/`
- [~] Champion gating — `train.py` compares against the best `val_mae` in the
      experiment and only then writes an artifact. Still not a real Model
      Registry entry with stages/aliases.

> **Note on the notebook's own "Next steps" cell:** it currently checks off
> `scripts/nyiso_preprocess.py`, `scripts/nyiso_train.py` and `make` targets
> that do not exist in this directory, and quotes MAE 731.9 MW / MAPE 3.11% /
> 38.7% baseline improvement, which disagrees with the notebook's own stored
> outputs (747.4 MW / 3.21% / 34.3%). The numbers in this README come from the
> executed cells and the MLflow store. That cell needs reconciling.
