import joblib
from src.config_loader import ConfigLoader
from src.model_training import CostModelTrainer


def main():
    print("\n======================================")
    print("CATBOOST MODEL TRAINING")
    print("======================================")

    config = ConfigLoader().get_config()

    df = joblib.load(config["paths"]["processed_data"])

    print("Processed shape:", df.shape)

    trainer = CostModelTrainer(config)

    metrics = trainer.train_model(df)

    print("\nTraining completed successfully")
    print(metrics)


if __name__ == "__main__":
    main()