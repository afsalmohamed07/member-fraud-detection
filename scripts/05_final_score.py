import pandas as pd
from pathlib import Path

from src.config_loader import ConfigLoader
from src.risk_scoring import RiskScorer
from src.explanation_builder import ExplanationBuilder


def main():
    print("\n======================================")
    print("FINAL RISK SCORING")
    print("======================================")

    config = ConfigLoader().get_config()

    input_path = "output/predictions/baseline_scored_claims.xlsx"
    output_path = config["paths"]["prediction_output"]

    df = pd.read_excel(input_path)

    print("Input shape:", df.shape)

    scorer = RiskScorer(config)
    scored_df = scorer.score(df)

    explanation_builder = ExplanationBuilder(config)
    scored_df = explanation_builder.build(scored_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    scored_df.to_excel(output_path, index=False)

    print("Final scored shape:", scored_df.shape)
    print("Saved:", output_path)

    print("\nRisk label counts:")
    print(scored_df["FINAL_LABEL"].value_counts())


if __name__ == "__main__":
    main()