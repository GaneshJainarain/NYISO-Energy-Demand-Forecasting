"""Lambda inference handler for the NYISO daily-peak forecaster.

Deliberately dependency-light: xgboost + numpy only. pandas (46 MB) and scipy
(144 MB) are what push a Lambda zip past the 250 MB limit, and building a
17-element feature vector needs neither.

Routes:
    GET  /health   model metadata and the history window it holds
    POST /predict  {"date","temp_max_f","temp_min_f","precip_in"} -> peak MW

Artifacts are pulled from S3 once per cold start and cached in module globals,
so warm invocations do no I/O.
"""

import datetime as dt
import json
import os

import numpy as np

BUCKET = os.environ.get("ARTIFACT_BUCKET", "")
PREFIX = os.environ.get("ARTIFACT_PREFIX", "artifacts")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL") or None

_MODEL = None
_RECENT = None
_META = None

# Training builds is_holiday with the `holidays` package, so serving uses the
# same library rather than a hand-rolled rule set. A hand-rolled version was
# tried and got 16 days wrong across 2019-2027: it missed observed-day shifting
# (July 4 on a Saturday is observed Friday the 3rd) and marked Juneteenth before
# it became federal in 2021. Those are exactly the holiday-adjacent days where
# load behaves unusually, so the 11 MB is worth it.
_HOLIDAYS = None


def _is_holiday(d):
    global _HOLIDAYS
    if _HOLIDAYS is None:
        import holidays as holidays_pkg
        _HOLIDAYS = holidays_pkg.US(years=range(2019, d.year + 5))
    return 1 if d in _HOLIDAYS else 0


def _s3():
    import boto3
    return boto3.client("s3", endpoint_url=S3_ENDPOINT)


def _load():
    """Fetch artifacts from S3 on cold start; no-op afterwards."""
    global _MODEL, _RECENT, _META
    if _MODEL is not None:
        return

    import tempfile
    import xgboost as xgb

    s3 = _s3()
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as f:
        s3.download_fileobj(BUCKET, f"{PREFIX}/model.ubj", f)
        model_path = f.name

    booster = xgb.Booster()
    booster.load_model(model_path)
    _MODEL = booster

    _RECENT = json.loads(
        s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/recent.json")["Body"].read())
    _META = json.loads(
        s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/metadata.json")["Body"].read())

    # Fail loudly on training/serving skew. An older xgboost reading a newer
    # model file drops base_score and returns raw tree sums, which look like
    # plausible small numbers rather than an error - the worst kind of bug.
    trained_with = _META.get("xgboost_version")
    if trained_with and trained_with != xgb.__version__:
        _MODEL = None
        raise RuntimeError(
            f"xgboost version skew: model trained with {trained_with}, "
            f"runtime has {xgb.__version__}. Rebuild the zip with "
            f"'xgboost=={trained_with}' (serving/build.sh reads this from "
            "metadata.json).")


def _history():
    """date -> (peak_mw, temp_avg_f), from the shipped recent.json."""
    return {
        d: (p, t) for d, p, t in zip(
            _RECENT["dates"], _RECENT["daily_peak_mw"], _RECENT["temp_avg_f"])
    }


def _lookup(hist, d, offset):
    key = (d - dt.timedelta(days=offset)).strftime("%Y-%m-%d")
    if key not in hist:
        raise ValueError(
            f"no history for {key} (need date-{offset}). Artifacts cover "
            f"{_RECENT['dates'][0]} to {_RECENT['dates'][-1]}; "
            "re-run scripts/export_model.py to refresh.")
    return hist[key]


def build_features(date_str, temp_max_f, temp_min_f, precip_in):
    """Assemble the 17 features in exactly config.FEATURE_COLS order."""
    d = dt.date.fromisoformat(date_str)
    hist = _history()

    temp_avg_f = (float(temp_max_f) + float(temp_min_f)) / 2.0
    bp = float(_META.get("balance_point_f", 65))

    peak_lag_1 = _lookup(hist, d, 1)[0]
    peak_lag_7 = _lookup(hist, d, 7)[0]
    peak_lag_14 = _lookup(hist, d, 14)[0]
    temp_lag_1 = _lookup(hist, d, 1)[1]

    # Trailing means over the N days ending the day before `date` - the same
    # shift-then-roll the training features use, so nothing leaks from `date`.
    roll_7 = float(np.mean([_lookup(hist, d, i)[0] for i in range(1, 8)]))
    roll_14 = float(np.mean([_lookup(hist, d, i)[0] for i in range(1, 15)]))

    values = {
        "temp_max_f": float(temp_max_f),
        "temp_min_f": float(temp_min_f),
        "temp_avg_f": temp_avg_f,
        "precip_in": float(precip_in),
        "hdd": max(0.0, bp - temp_avg_f),
        "cdd": max(0.0, temp_avg_f - bp),
        "temp_lag_1": temp_lag_1,
        "day_of_week": float(d.weekday()),
        "is_weekend": 1.0 if d.weekday() >= 5 else 0.0,
        "month": float(d.month),
        "day_of_year": float(d.timetuple().tm_yday),
        "is_holiday": float(_is_holiday(d)),
        "peak_lag_1": peak_lag_1,
        "peak_lag_7": peak_lag_7,
        "peak_lag_14": peak_lag_14,
        "peak_roll_mean_7": roll_7,
        "peak_roll_mean_14": roll_14,
    }
    cols = _META["feature_cols"]
    return np.array([[values[c] for c in cols]], dtype=np.float32), values


def predict(payload):
    import xgboost as xgb

    for field in ("date", "temp_max_f", "temp_min_f"):
        if field not in payload:
            raise ValueError(f"missing required field: {field}")

    X, values = build_features(
        payload["date"], payload["temp_max_f"], payload["temp_min_f"],
        payload.get("precip_in", 0.0))

    dmat = xgb.DMatrix(X, feature_names=_META["feature_cols"])
    pred = float(_MODEL.predict(dmat)[0])

    return {
        "date": payload["date"],
        "predicted_peak_mw": round(pred, 1),
        # Persistence is what the model has to beat; returning it lets a caller
        # see the model's contribution rather than trusting the number.
        "baseline_persistence_mw": round(values["peak_lag_1"], 1),
        "delta_vs_baseline_mw": round(pred - values["peak_lag_1"], 1),
        "model_run_id": _META["run_id"],
        "model_val_mae": _META["metrics"].get("val_mae"),
    }


def _response(code, body):
    return {
        "statusCode": code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    try:
        _load()
    except Exception as exc:  # cold-start failures are the confusing ones
        return _response(500, {"error": f"artifact load failed: {exc}"})

    path = (event.get("rawPath") or event.get("path") or "/").rstrip("/") or "/"
    method = (event.get("requestContext", {}).get("http", {}).get("method")
              or event.get("httpMethod") or "GET").upper()

    if path.endswith("/health") or (path == "/" and method == "GET"):
        return _response(200, {
            "status": "ok",
            "model_run_id": _META["run_id"],
            "metrics": _META["metrics"],
            "history_from": _RECENT["dates"][0],
            "history_to": _RECENT["dates"][-1],
        })

    if method != "POST":
        return _response(405, {"error": f"{method} not allowed on {path}"})

    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            import base64
            body = base64.b64decode(body).decode()
        payload = json.loads(body)
    except Exception as exc:
        return _response(400, {"error": f"invalid JSON body: {exc}"})

    try:
        return _response(200, predict(payload))
    except ValueError as exc:
        return _response(400, {"error": str(exc)})
    except Exception as exc:
        return _response(500, {"error": f"prediction failed: {exc}"})
