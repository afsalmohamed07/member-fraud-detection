import pandas as pd

from src.config_loader import ConfigLoader
from src.cost_prediction_baseline import CostPredictionBaselineBuilder


def main():
    print("\n======================================")
    print("BUILD COST PREDICTION BASELINES")
    print("======================================")

    config = ConfigLoader().get_config()

    processed_path = config["paths"]["processed_data"]

    df = pd.read_pickle(processed_path)

    print("Processed shape:", df.shape)

    builder = CostPredictionBaselineBuilder(config)

    baselines = builder.build(df)

    output_path = builder.save(baselines)

    print("\nSaved:", output_path)

    for level, obj in baselines.items():
        print(level, obj["table"].shape)


if __name__ == "__main__":
    main()