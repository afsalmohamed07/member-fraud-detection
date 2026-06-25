import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd


FILES_TO_DELETE = [
    "data/raw/combined_dataset.xlsx",
    "data/raw/combined_dataset.pkl",
    "data/raw/combined_dataset.parquet",
    "data/processed/processed_claims.pkl",
]


STEPS = [
    ("01 Merge datasets", "python -m scripts.00_merge_datasets"),
    ("02 Validate data", "python -m scripts.01_validate_data"),
    ("03 Preprocess data", "python -m scripts.02_preprocess_data"),
    ("04 Build baselines", "python -m scripts.03_build_baseline"),
    ("05 Build cost prediction baselines", "python -m scripts.10_build_cost_prediction_baselines"),
    ("06 Train CatBoost", "python -m scripts.06_train_catboost"),
    ("07 Train Isolation Forest", "python -m scripts.07_train_isolation"),
    ("08 Train pharmacy model", "python -m scripts.11_train_pharmacy_cost_model"),
    ("09 Train risk classifier", "python -m scripts.14_train_risk_classifier"),
    ("10 Build investigation artifacts", "python -m scripts.16_build_investigation_artifacts"),
    ("11 Build treatment intelligence", "python -m scripts.17_build_treatment_intelligence"),
]


def seconds_to_text(seconds):
    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.2f} min"

    hours = minutes / 60
    return f"{hours:.2f} hr"


def get_row_count():
    processed_path = Path("data/processed/processed_claims.pkl")
    raw_path = Path("data/raw/combined_dataset.pkl")

    try:
        if processed_path.exists():
            df = joblib.load(processed_path)
            return len(df)

        if raw_path.exists():
            df = pd.read_pickle(raw_path)
            return len(df)

    except Exception:
        return 0

    return 0


def cleanup_old_outputs():
    print("\nCleaning old rebuild outputs...")

    for file_path in FILES_TO_DELETE:
        path = Path(file_path)

        if path.exists():
            path.unlink()
            print(f"Deleted: {path}")


def run_step(name, command):
    print("\n" + "=" * 90)
    print(f"STARTING: {name}")
    print(command)
    print("=" * 90)

    start = time.perf_counter()

    result = subprocess.run(command, shell=True)

    elapsed = time.perf_counter() - start

    status = "SUCCESS" if result.returncode == 0 else "FAILED"

    print(f"\n{name} => {status}")
    print(f"Time taken: {seconds_to_text(elapsed)}")

    return {
        "step": name,
        "command": command,
        "status": status,
        "seconds": round(elapsed, 2),
        "time_text": seconds_to_text(elapsed),
    }


def print_summary(results, total_seconds, row_count):
    print("\n" + "=" * 90)
    print("FULL REBUILD TIME SUMMARY")
    print("=" * 90)

    print(f"Dataset rows processed: {row_count}")
    print(f"Total time: {seconds_to_text(total_seconds)}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['step']:<40} "
            f"{r['status']:<10} "
            f"{r['time_text']:>15}"
        )

    print("-" * 90)

    if row_count > 0:
        per_lakh = total_seconds / (row_count / 100000)
        estimate_12_lakh = per_lakh * 12

        print(f"Time per 1 lakh rows: {seconds_to_text(per_lakh)}")
        print(f"Estimated time for 12 lakh rows: {seconds_to_text(estimate_12_lakh)}")

    print("=" * 90)


def save_summary(results, total_seconds, row_count):
    output_dir = Path("output/rebuild_logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(results)

    df.loc[len(df)] = {
        "step": "TOTAL",
        "command": "",
        "status": "COMPLETED",
        "seconds": round(total_seconds, 2),
        "time_text": seconds_to_text(total_seconds),
    }

    if row_count > 0:
        per_lakh = total_seconds / (row_count / 100000)
        estimate_12_lakh = per_lakh * 12
    else:
        per_lakh = 0
        estimate_12_lakh = 0

    df.loc[len(df)] = {
        "step": "ROWS_PROCESSED",
        "command": "",
        "status": str(row_count),
        "seconds": 0,
        "time_text": "",
    }

    df.loc[len(df)] = {
        "step": "TIME_PER_1_LAKH_ROWS",
        "command": "",
        "status": "",
        "seconds": round(per_lakh, 2),
        "time_text": seconds_to_text(per_lakh),
    }

    df.loc[len(df)] = {
        "step": "ESTIMATED_TIME_FOR_12_LAKH_ROWS",
        "command": "",
        "status": "",
        "seconds": round(estimate_12_lakh, 2),
        "time_text": seconds_to_text(estimate_12_lakh),
    }

    csv_path = output_dir / f"full_rebuild_time_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print("\nSaved rebuild time summary:")
    print(csv_path)


def main():
    total_start = time.perf_counter()

    print("\nFULL REBUILD STARTED")
    print("One-click pipeline: raw data → processed data → baselines → models → artifacts")

    cleanup_old_outputs()

    results = []

    for name, command in STEPS:
        row = run_step(name, command)
        results.append(row)

        if row["status"] == "FAILED":
            total_seconds = time.perf_counter() - total_start
            row_count = get_row_count()

            print_summary(results, total_seconds, row_count)
            save_summary(results, total_seconds, row_count)

            print("\nPipeline stopped because one step failed.")
            sys.exit(1)

    total_seconds = time.perf_counter() - total_start
    row_count = get_row_count()

    print_summary(results, total_seconds, row_count)
    save_summary(results, total_seconds, row_count)

    print("\nFULL REBUILD COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()