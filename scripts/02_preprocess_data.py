from pathlib import Path
import pandas as pd

from src.config_loader import ConfigLoader
from src.preprocessing import ClaimsPreprocessor, load_input_data


def main():
    print("\n======================================")
    print("LOADING CONFIG")
    print("======================================")

    config = ConfigLoader().get_config()

    raw_path = config["paths"]["raw_data"]
    processed_path = config["paths"]["processed_data"]

    print("Raw data path:", raw_path)
    print("Processed data path:", processed_path)

    print("\n======================================")
    print("LOADING RAW DATA")
    print("======================================")

    df_raw = load_input_data(raw_path)

    print("Raw shape:", df_raw.shape)

    print("\n======================================")
    print("RUNNING PREPROCESSING")
    print("======================================")

    preprocessor = ClaimsPreprocessor(config)
    df_processed = preprocessor.preprocess(df_raw)

    raw_keep_cols = [
        "SERVICE_DT",
        "DOCTOR_LICENSE",
        "FACILITIES",
        "PROV_NAME",
        "DOC_NAME",
        "INS_TREAT_CODE",
        "CPT_CODE",
        "REJ_CODE",
        "REJ_DESC",
        "PBM_REJ_CODE",
        "PBM_REJ_DESC",
    ]

    for col in raw_keep_cols:
        if col in df_raw.columns and col not in df_processed.columns:
            df_processed[col] = df_raw[col].values

    print("Processed shape:", df_processed.shape)

    print("\n======================================")
    print("SAVING PROCESSED DATA")
    print("======================================")

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)

    # FULL REBUILD MODE:
    # overwrite processed file from current combined raw data only.
    df_processed.to_pickle(processed_path)

    print("Saved:", processed_path)

    print("\nSample rows:")
    print(df_processed.head())


if __name__ == "__main__":
    main()