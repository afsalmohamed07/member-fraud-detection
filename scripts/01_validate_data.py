import pandas as pd
from pathlib import Path
from pprint import pprint

from src.config_loader import ConfigLoader
from src.validation import DataValidator


def load_raw_data(raw_path):
    raw_path = Path(raw_path)

    if raw_path.suffix.lower() == ".pkl":
        print(f"Loading pickle: {raw_path}")
        return pd.read_pickle(raw_path)

    if raw_path.suffix.lower() == ".parquet":
        print(f"Loading parquet: {raw_path}")
        return pd.read_parquet(raw_path)

    if raw_path.suffix.lower() in [".xlsx", ".xls"]:
        print(f"Loading excel: {raw_path}")
        return pd.read_excel(raw_path, engine="openpyxl")

    if raw_path.suffix.lower() == ".csv":
        print(f"Loading csv: {raw_path}")
        return pd.read_csv(raw_path, low_memory=False)

    raise FileNotFoundError(f"Unsupported raw data path: {raw_path}")


def main():
    print("\n======================================")
    print("LOADING CONFIG")
    print("======================================")

    config = ConfigLoader().get_config()
    raw_data_path = config["paths"]["raw_data"]

    print(f"\nRaw data path: {raw_data_path}")

    print("\n======================================")
    print("LOADING DATA")
    print("======================================")

    df = load_raw_data(raw_data_path)

    print(f"\nDataset shape: {df.shape}")

    print("\n======================================")
    print("RUNNING VALIDATION")
    print("======================================")

    validator = DataValidator(config)
    validation_result = validator.run_all_checks(df)

    print("\n======================================")
    print("VALIDATION RESULT")
    print("======================================")

    pprint(validation_result)

    print("\n======================================")
    print("VALIDATION COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()