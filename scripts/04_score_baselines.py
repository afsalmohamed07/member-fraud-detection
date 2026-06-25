import pandas as pd
from pathlib import Path

from src.config_loader import ConfigLoader
from src.baseline_scoring import BaselineScorer


def main():
    print("\n======================================")
    print("BASELINE SCORING")
    print("======================================")

    config = ConfigLoader().get_config()

    processed_path = config["paths"]["processed_data"]
    output_path = "output/predictions/baseline_scored_claims.xlsx"

    print("\nProcessed path:", processed_path)
    print("Output path:", output_path)

    df = pd.read_pickle(processed_path)

    print("\nProcessed shape:", df.shape)

    scorer = BaselineScorer(config)
    scored_df = scorer.score_all(df)

    Path("output/predictions").mkdir(parents=True, exist_ok=True)

    scored_df.to_excel(output_path, index=False)

    print("\n======================================")
    print("BASELINE SCORING COMPLETED")
    print("======================================")

    print("Scored shape:", scored_df.shape)
    print("Saved:", output_path)

    anomaly_cols = [
        "COST_ANOMALY",
        "IQR_COST_ANOMALY",
        "PROVIDER_COST_ANOMALY",
        "RARE_TREATMENT_ANOMALY",
        "PHARMACY_MISSING_DRUG_ANOMALY",
        "NON_PHARMACY_WITH_DRUG_ANOMALY",
        "RARE_DRUG_ANOMALY",
        "DURATION_ANOMALY",
        "DOSAGE_ANOMALY",
        "DRUG_PRICE_ANOMALY",
        "DOCTOR_DRUG_DURATION_ANOMALY",
        "DOCTOR_DRUG_QTY_ANOMALY",
        "BASELINE_ANOMALY_COUNT",
    ]

    existing_cols = [col for col in anomaly_cols if col in scored_df.columns]
    missing_cols = [col for col in anomaly_cols if col not in scored_df.columns]

    print("\nExisting anomaly columns:")
    print(existing_cols)

    if missing_cols:
        print("\nMissing anomaly columns:")
        print(missing_cols)

    print("\nAnomaly counts:")
    print(scored_df[existing_cols].sum().sort_values(ascending=False))

    print("\nTop 20 output columns:")
    print(scored_df.columns[:20].tolist())


if __name__ == "__main__":
    main()