import pandas as pd

from src.config_loader import ConfigLoader
from src.isolation_model import IsolationModel


def main():

    print("\n======================================")
    print("ISOLATION FOREST TRAINING")
    print("======================================")

    config = ConfigLoader().get_config()

    processed_path = config["paths"]["processed_data"]

    df = pd.read_pickle(processed_path)

    print("Processed shape:", df.shape)

    isolation_model = IsolationModel(config)

    model, feature_cols = isolation_model.train(df)

    model_path = isolation_model.save()

    print("\n======================================")
    print("TRAINING COMPLETED")
    print("======================================")

    print("\nFeature count:", len(feature_cols))

    print("\nModel saved at:")
    print(model_path)


if __name__ == "__main__":
    main()