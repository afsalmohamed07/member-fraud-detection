import pandas as pd

from src.config_loader import ConfigLoader
from src.baseline_builder import BaselineBuilder


def main():
    print("\n======================================")
    print("BUILDING BASELINES")
    print("======================================")

    config = ConfigLoader().get_config()

    processed_path = config["paths"]["processed_data"]
    df = pd.read_pickle(processed_path)

    print("Processed shape:", df.shape)

    builder = BaselineBuilder(config)
    baselines = builder.build_all_baselines(df)
    builder.save_baselines(baselines)

    print("\nBaselines created:")

    for name, baseline_df in baselines.items():
        print(f"{name}: {baseline_df.shape}")

    print("\nSaved to:", config["paths"]["baselines_dir"])


if __name__ == "__main__":
    main()