#!/usr/bin/env python
"""Promote the champion MLflow run into deployable serving artifacts.

Writes three small files that the Lambda needs and nothing else:

    artifacts/model.ubj      the champion booster
    artifacts/recent.json    trailing daily peaks + temps, for lag features
    artifacts/metadata.json  run id, metrics, feature order

Serving deliberately does not read the MLflow store or the feature CSV at
runtime - those are laptop-shaped. This is the handoff point between training
and inference.

    python scripts/export_model.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

ARTIFACTS = config.ROOT / "artifacts"

# Days of history to ship: enough for peak_lag_14 and peak_roll_mean_14 plus
# a little slack for gaps.
RECENT_DAYS = 21


def find_model_artifact(run_id):
    """Local path to a run's booster, or None if it never logged one.

    Champion-gated runs deliberately have no artifact, so the best-scoring run
    is often not the one holding a model.
    """
    import mlflow
    for candidate in ("model/model.ubj", "model"):
        try:
            p = Path(mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=candidate))
        except Exception:
            continue
        if p.is_file():
            return p
        found = next(p.rglob("*.ubj"), None)
        if found:
            return found
    return None


def champion_run(client, experiment_id):
    """Best val_mae among runs that actually shipped a model artifact."""
    runs = client.search_runs(
        [experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_mae ASC"],
        max_results=100,
    )
    scored = [r for r in runs if "val_mae" in r.data.metrics]
    if not scored:
        raise SystemExit("no finished run with a val_mae in the experiment")

    for r in scored:
        artifact = find_model_artifact(r.info.run_id)
        if artifact:
            return r, artifact

    raise SystemExit(
        f"{len(scored)} scored runs, none with a model artifact - they were all "
        "champion-gated out. Run: python scripts/train.py --always-log-model")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ARTIFACTS))
    args = ap.parse_args()

    import mlflow

    mlflow.set_tracking_uri(config.tracking_uri())
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"experiment {config.EXPERIMENT_NAME!r} not found - train first")

    run, model_path = champion_run(client, exp.experiment_id)
    metrics = run.data.metrics
    print(f"champion run: {run.info.run_id}  val_mae={metrics['val_mae']:.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- model ---
    shutil.copyfile(model_path, out / "model.ubj")
    print(f"wrote {out.name}/model.ubj ({(out / 'model.ubj').stat().st_size / 1024:.0f} KB)")

    # --- recent history for lag features ---
    df = pd.read_csv(config.DAILY_FEATURES, index_col=0, parse_dates=True).sort_index()
    tail = df.tail(RECENT_DAYS)
    recent = {
        "dates": [d.strftime("%Y-%m-%d") for d in tail.index],
        "daily_peak_mw": [float(v) for v in tail["daily_peak_mw"]],
        "temp_avg_f": [float(v) for v in tail["temp_avg_f"]],
    }
    (out / "recent.json").write_text(json.dumps(recent, indent=2))
    print(f"wrote {out.name}/recent.json ({len(tail)} days, "
          f"{tail.index.min():%Y-%m-%d} to {tail.index.max():%Y-%m-%d})")

    # --- metadata ---
    import xgboost as xgb_pkg
    import json as _json
    booster = xgb_pkg.Booster()
    booster.load_model(str(out / "model.ubj"))
    base_score = _json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]

    meta = {
        "run_id": run.info.run_id,
        # Serving MUST use this exact version. An older xgboost silently fails
        # to read base_score out of a newer model file and then returns raw
        # tree sums - predictions near zero instead of ~20,000 MW, with no error.
        "xgboost_version": xgb_pkg.__version__,
        "base_score": base_score,
        "experiment": config.EXPERIMENT_NAME,
        "feature_cols": config.FEATURE_COLS,
        "target": config.TARGET_COL,
        "balance_point_f": config.BALANCE_POINT_F,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "exported_at": pd.Timestamp.utcnow().isoformat(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out.name}/metadata.json")


if __name__ == "__main__":
    main()
