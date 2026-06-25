import joblib
import pandas as pd
import numpy as np

from src.config_loader import ConfigLoader
from src.baseline_scoring import BaselineScorer
from src.risk_classifier import RiskClassifier


def main():
    config = ConfigLoader().get_config()
    processed_path = config["paths"]["processed_data"]

    print("\nLoading processed data...")
    df = joblib.load(processed_path)
    print(f"Rows loaded: {len(df)}")

    print("\nApplying baseline scoring...")
    baseline_scorer = BaselineScorer(config)
    df = baseline_scorer.score_all(df)

    fallback_cols = [
        "FINAL_EXPECTED_COST",
        "PATTERN_MEDIAN",
        "COST_MEDIAN_AMT",
        "ML_EXPECTED_COST",
    ]

    for col in fallback_cols:
        if col not in df.columns:
            df[col] = 0

    df["FINAL_EXPECTED_COST"] = pd.to_numeric(
        df["FINAL_EXPECTED_COST"],
        errors="coerce"
    ).fillna(0)

    df["PATTERN_MEDIAN"] = pd.to_numeric(
        df["PATTERN_MEDIAN"],
        errors="coerce"
    ).fillna(0)

    df["COST_MEDIAN_AMT"] = pd.to_numeric(
        df["COST_MEDIAN_AMT"],
        errors="coerce"
    ).fillna(0)

    df["ML_EXPECTED_COST"] = pd.to_numeric(
        df["ML_EXPECTED_COST"],
        errors="coerce"
    ).fillna(0)

    df["FINAL_EXPECTED_COST"] = np.where(
        df["FINAL_EXPECTED_COST"] > 0,
        df["FINAL_EXPECTED_COST"],
        np.where(
            df["PATTERN_MEDIAN"] > 0,
            df["PATTERN_MEDIAN"],
            np.where(
                df["COST_MEDIAN_AMT"] > 0,
                df["COST_MEDIAN_AMT"],
                df["ML_EXPECTED_COST"]
            )
        )
    )

    df["PA_EST_AMT_LC"] = pd.to_numeric(
        df["PA_EST_AMT_LC"],
        errors="coerce"
    ).fillna(0)

    df["FINAL_COST_RATIO"] = np.where(
        df["FINAL_EXPECTED_COST"] > 0,
        df["PA_EST_AMT_LC"] / df["FINAL_EXPECTED_COST"],
        1
    )

    required_cols = [
        "ML_EXPECTED_COST",
        "PATTERN_P75",
        "PATTERN_P90",
        "PATTERN_MAX",
        "SUPPORT_COUNT",
        "SIMILAR_CLAIMS_COUNT",
        "SIMILAR_REJECTION_RATE",
        "SIMILAR_COST_ABOVE_P90",
        "USAGE_RATIO",
        "GLOBAL_TREAT_USAGE_PCT",
        "PROVIDER_TREAT_USAGE_PCT",
        "ISOLATION_SCORE",
        "UNKNOWN_DIAGNOSIS_ANOMALY",
        "UNSEEN_TREATMENT_CODE_ANOMALY",
        "RARE_TREATMENT_ANOMALY",
        "RARE_DRUG_ANOMALY",
        "DURATION_OUTSIDE_RANGE_ANOMALY",
        "DOSAGE_OUTSIDE_RANGE_ANOMALY",
        "LICENSE_ANOMALY",
        "PROVIDER_BEHAVIOR_ANOMALY",
        "UPCODING_ANOMALY",
        "SIMILAR_CLAIMS_ANOMALY",
        "ISOLATION_ANOMALY",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    print("\nLabel distribution check:")
    temp_ratio = df["FINAL_COST_RATIO"]

    print("NORMAL:", int(((temp_ratio >= 0.5) & (temp_ratio < 1.2)).sum()))
    print("LOW_COST:", int((temp_ratio < 0.5).sum()))
    print("MEDIUM:", int(((temp_ratio >= 1.2) & (temp_ratio < 2)).sum()))
    print("HIGH:", int(((temp_ratio >= 2) & (temp_ratio < 5)).sum()))
    print("CRITICAL:", int((temp_ratio >= 5).sum()))

    classifier = RiskClassifier(config)

    print("\nTraining LightGBM risk classifier...\n")
    classifier.train(df)

    print("\nRisk classifier training completed.")


if __name__ == "__main__":
    main()