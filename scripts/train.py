#!/usr/bin/env python
"""Train the daily-peak forecaster and log it to MLflow.

Ports sections 5-6b of the notebook: chronological split, XGBoost, and the
persistence baseline that decides whether the result means anything.

Model artifacts are champion-gated. Metrics and params are logged on every
run because they are cheap; the ~750 KB booster is only written when the run
beats the best val_mae seen so far. Without that, a nightly retrain writes a
quarter of a gigabyte a year to say the same thing 365 times.

    python scripts/train.py
    python scripts/train.py --as-of 2026-08-30   # reproduce a past split
    python scripts/train.py --always-log-model   # bypass the champion gate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def load_features(as_of=None):
    if not config.DAILY_FEATURES.exists():
        raise SystemExit(
            f"{config.DAILY_FEATURES} missing. Run `python scripts/preprocess.py` first."
        )
    df = pd.read_csv(config.DAILY_FEATURES, index_col=0, parse_dates=True)
    if as_of:
        df = df[df.index <= pd.Timestamp(as_of)]
        if df.empty:
            raise SystemExit(f"no rows at or before {as_of}")
    return df


def chronological_split(df, weeks=None):
    """Hold out the most recent N weeks. Never shuffle: peak_lag_1 alone would
    let a shuffled model see next week while predicting last week."""
    weeks = config.VALIDATION_WEEKS if weeks is None else weeks
    split_date = df.index.max() - pd.Timedelta(weeks=weeks)
    train_df = df[df.index <= split_date]
    val_df = df[df.index > split_date]
    if val_df.empty or train_df.empty:
        raise SystemExit(f"split at {split_date:%Y-%m-%d} left an empty side")
    return train_df, val_df, split_date


def score(y_true, y_pred):
    from sklearn.metrics import (mean_absolute_error,
                                 mean_absolute_percentage_error,
                                 mean_squared_error)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def best_previous_mae(client, experiment_id):
    """Lowest val_mae logged so far, or None on a cold experiment."""
    runs = client.search_runs(
        [experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_mae ASC"],
        max_results=1,
    )
    if not runs or "val_mae" not in runs[0].data.metrics:
        return None
    return runs[0].data.metrics["val_mae"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", default=None,
                    help="truncate the dataset at this date (reproduce a past split)")
    ap.add_argument("--validation-weeks", type=int, default=None)
    ap.add_argument("--always-log-model", action="store_true",
                    help="skip the champion gate and always write the artifact")
    ap.add_argument("--run-name", default="xgb_daily_peak")
    args = ap.parse_args()

    import mlflow
    import mlflow.xgboost
    import xgboost as xgb

    df = load_features(args.as_of)
    train_df, val_df, split_date = chronological_split(df, args.validation_weeks)

    X_train, y_train = train_df[config.FEATURE_COLS], train_df[config.TARGET_COL]
    X_val, y_val = val_df[config.FEATURE_COLS], val_df[config.TARGET_COL]

    print(f"train: {X_train.shape[0]:,} days  {train_df.index.min():%Y-%m-%d} "
          f"to {train_df.index.max():%Y-%m-%d}")
    print(f"val:   {X_val.shape[0]:,} days  {val_df.index.min():%Y-%m-%d} "
          f"to {val_df.index.max():%Y-%m-%d}")

    mlflow.set_tracking_uri(config.tracking_uri())
    experiment = mlflow.set_experiment(config.EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()
    champion = best_previous_mae(client, experiment.experiment_id)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(config.XGB_PARAMS)
        mlflow.log_param("validation_weeks", args.validation_weeks or config.VALIDATION_WEEKS)
        mlflow.log_param("n_features", len(config.FEATURE_COLS))
        mlflow.log_param("train_start", f"{train_df.index.min():%Y-%m-%d}")
        mlflow.log_param("train_end", f"{train_df.index.max():%Y-%m-%d}")
        mlflow.log_param("split_date", f"{split_date:%Y-%m-%d}")
        mlflow.log_param("n_train", len(train_df))

        model = xgb.XGBRegressor(**config.XGB_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        m = score(y_val, preds)
        # Persistence: "tomorrow's peak equals today's peak". peak_lag_1 is also
        # the model's most important feature, so without this the score cannot
        # be distinguished from the model echoing yesterday.
        b = score(y_val, val_df["peak_lag_1"].values)

        for k, v in m.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in b.items():
            mlflow.log_metric(f"baseline_{k}", v)
        improvement = 1 - m["mae"] / b["mae"]
        mlflow.log_metric("mae_improvement_over_persistence", improvement)

        print(f"\n{'':12} {'model':>10} {'persistence':>12}")
        print(f"{'MAE  (MW)':12} {m['mae']:>10.2f} {b['mae']:>12.2f}")
        print(f"{'RMSE (MW)':12} {m['rmse']:>10.2f} {b['rmse']:>12.2f}")
        print(f"{'MAPE (%)':12} {m['mape']*100:>10.2f} {b['mape']*100:>12.2f}")
        print(f"\nMAE improvement over persistence: {improvement:.1%}")

        if improvement <= 0:
            print("WARNING: model does not beat persistence on MAE")

        # --- champion gate ---
        if args.always_log_model or champion is None or m["mae"] < champion:
            mlflow.xgboost.log_model(model, name="model")
            was = "cold start" if champion is None else f"beat {champion:.2f}"
            print(f"logged model artifact ({was})")
            mlflow.set_tag("champion", "true")
        else:
            print(f"skipped model artifact: val_mae {m['mae']:.2f} did not beat "
                  f"champion {champion:.2f} (~750 KB saved)")
            mlflow.set_tag("champion", "false")

        print(f"run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
