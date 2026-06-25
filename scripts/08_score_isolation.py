import pandas as pd
import joblib
from pathlib import Path

from src.config_loader import ConfigLoader
from src.isolation_model import IsolationModel


def main():
    print("\n======================================")
    print("ISOLATION FOREST SCORING")
    print("======================================")

    config = ConfigLoader().get_config()

    processed_path = config["paths"]["processed_data"]
    model_path = Path(config["paths"]["models_dir"]) / "isolation_forest.pkl"
    output_path = "output/predictions/isolation_scored_claims.xlsx"

    df = pd.read_pickle(processed_path)

    print("Processed shape:", df.shape)

    isolation = IsolationModel(config)
    isolation.model = joblib.load(model_path)

    scored_df = isolation.predict(df)

    Path("output/predictions").mkdir(parents=True, exist_ok=True)

    scored_df.to_excel(output_path, index=False)

    print("\nScored shape:", scored_df.shape)
    print("Saved:", output_path)

    print("\nIsolation anomaly count:")
    print(scored_df["ISOLATION_ANOMALY"].value_counts())


if __name__ == "__main__":
    main()