"""Daily NYISO forecasting pipeline.

    ingest_eia  ──┐
                  ├──> preprocess ──> train
    ingest_noaa ──┘

The two ingests are independent - they write different parquet files - so they
run in parallel. Everything downstream is deterministic and cheap, which is the
whole reason the data path was split at the raw/processed boundary.

Retries matter here: both upstream APIs are public, free, and occasionally
return 5xx. A transient failure at 11:00 should not mean no forecast today.
"""

from datetime import datetime, timedelta

from airflow import DAG

# BashOperator moved to the standard provider in Airflow 3; support both so the
# DAG parses on either major version.
try:
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.bash import BashOperator

# Project root as mounted inside the Airflow containers (see docker-compose.yml).
PROJECT_DIR = "/opt/airflow/project"
PYTHON = "python"

default_args = {
    "owner": "nyiso",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}

with DAG(
    dag_id="nyiso_daily",
    description="Ingest NYISO demand + NYC weather, rebuild features, retrain peak model",
    # 11:00 UTC = 07:00 EDT, comfortably after EIA publishes the prior day.
    schedule="0 11 * * *",
    start_date=datetime(2026, 9, 1),
    # Backfilling makes no sense here: ingest.py is incremental against a
    # high-water mark, so a catch-up run would just refetch the same window.
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["nyiso", "forecasting"],
) as dag:

    ingest_eia = BashOperator(
        task_id="ingest_eia",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} scripts/ingest.py --source eia",
    )

    ingest_noaa = BashOperator(
        task_id="ingest_noaa",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} scripts/ingest.py --source noaa",
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} scripts/preprocess.py",
    )

    train = BashOperator(
        task_id="train",
        # Champion-gated: logs metrics every run, writes the ~750 KB booster
        # only when it beats the best val_mae so far.
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} scripts/train.py",
    )

    [ingest_eia, ingest_noaa] >> preprocess >> train
