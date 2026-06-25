import pandas as pd

from src.config_loader import ConfigLoader
from src.pharmacy_ml_model import PharmacyCostModel


def main():
    print("\n======================================")
    print("TRAIN PHARMACY DRUG COST MODEL")
    print("======================================")

    config = ConfigLoader().get_config()

    processed_path = config["paths"]["processed_data"]

    df = pd.read_pickle(processed_path)

    print("Processed shape:", df.shape)

    trainer = PharmacyCostModel(config)
    model, features = trainer.train(df)

    print("\nTraining completed")
    print("Features:", features)
    print("Saved pharmacy model:")
    print(trainer.model_path)


if __name__ == "__main__":
    main()