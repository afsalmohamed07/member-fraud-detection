import pandas as pd
import numpy as np
from pathlib import Path

from src.duration_parser import DurationParser
from src.dosage_parser import DosageParser


class ClaimsPreprocessor:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]
        self.invalid_values = config["preprocessing"]["invalid_values"]

        self.duration_parser = DurationParser()
        self.dosage_parser = DosageParser()

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        return df

    def clean_text_column(self, series: pd.Series) -> pd.Series:
        return (
            series
            .astype(str)
            .str.strip()
            .str.upper()
            .replace(self.invalid_values, np.nan)
        )

    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        text_cols = [
            self.columns["diagnosis"],
            self.columns["treatment_code"],
            self.columns["drug_name"],
            self.columns["scientific_drug_name"],
            self.columns["dosage_desc"],
            self.columns["provider"],
            self.columns["doctor"],
            self.columns["service_type"],
            self.columns["category"],
            "FACILITIES",
            "DOCTOR_LICENSE",
        ]

        for col in text_cols:
            if col in df.columns:
                df[col] = self.clean_text_column(df[col])

        return df

    def clean_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        date_cols = [
            "SERVICE_DT",
            "INTM_DATE",
            "LMP_DT",
            "APPEAL_DATE",
            "PA_CR_DT",
            "PA_APPR_DT",
        ]

        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce",
                    dayfirst=True
                )

        return df

    def clean_amount_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        amount_cols = [
            self.columns["est_amount"],
            self.columns["appr_amount"],
            self.columns["rej_amount"],
        ]

        for col in amount_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def clean_quantity_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        qty_col = self.columns["quantity"]

        if qty_col in df.columns:
            df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

        return df

    def normalize_service_type(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        service_col = self.columns["service_type"]

        if service_col not in df.columns:
            return df

        service_map = {
            "OUT-PATIENT": "OP",
            "OUT PATIENT": "OP",
            "OUTPATIENT": "OP",
            "OP": "OP",
            "O/P": "OP",
            "IN-PATIENT": "IP",
            "IN PATIENT": "IP",
            "INPATIENT": "IP",
            "IP": "IP",
            "I/P": "IP",
        }

        df["SERVICE_TYPE_NORM"] = df[service_col].replace(service_map)

        return df

    def normalize_category(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        cat_col = self.columns["category"]

        if cat_col not in df.columns:
            return df

        df["PA_CATG_NORM"] = df[cat_col].copy()

        pharmacy_keywords = [
            "PHARMACY",
            "PHARMACY - LTC",
            "PHARMACY LTC",
        ]

        df.loc[
            df["PA_CATG_NORM"].isin(pharmacy_keywords),
            "PA_CATG_NORM"
        ] = "PHARMACY"

        return df

    def normalize_treatment_code(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        treat_col = self.columns["treatment_code"]

        if treat_col not in df.columns:
            return df

        df["PROV_TREAT_CODE_CLEAN"] = (
            df[treat_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace(self.invalid_values, np.nan)
        )

        return df

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        est_col = self.columns["est_amount"]
        appr_col = self.columns["appr_amount"]
        rej_col = self.columns["rej_amount"]

        df["APPROVAL_RATIO"] = np.where(
            df[est_col] > 0,
            df[appr_col] / df[est_col],
            np.nan
        )

        df["REJECTION_RATIO"] = np.where(
            df[est_col] > 0,
            df[rej_col] / df[est_col],
            np.nan
        )

        df["IS_FULLY_APPROVED"] = (
            (df[est_col] > 0) &
            (df[appr_col] >= df[est_col])
        ).astype(int)

        df["IS_FULLY_REJECTED"] = (
            (df[est_col] > 0) &
            (df[rej_col] >= df[est_col])
        ).astype(int)

        df["IS_PARTIAL_APPROVED"] = (
            (df[appr_col] > 0) &
            (df[appr_col] < df[est_col])
        ).astype(int)

        return df

    def add_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        duration_col = self.columns["drug_duration"]

        if duration_col in df.columns:
            df = self.duration_parser.process_duration_column(
                df,
                duration_col
            )

        return df

    def add_dosage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dosage_col = self.columns["dosage_desc"]

        if dosage_col in df.columns:
            df = self.dosage_parser.process_dosage_column(
                df,
                dosage_col
            )

        return df

    def select_model_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_cols = [
            self.columns["diagnosis"],
            self.columns["treatment_code"],
            "PROV_TREAT_CODE_CLEAN",

            self.columns["drug_name"],
            self.columns["scientific_drug_name"],
            self.columns["drug_duration"],
            "DURATION_RAW_CLEANED",
            "DURATION_DAYS",
            "DURATION_UNIT",

            self.columns["dosage_desc"],
            "DOSAGE_RAW_CLEANED",
            "DOSAGE_PER_DAY",
            "DOSAGE_NORMALIZED",
            "DOSAGE_TYPE",

            self.columns["quantity"],
            self.columns["provider"],
            self.columns["doctor"],
            self.columns["service_type"],
            "SERVICE_TYPE_NORM",
            self.columns["category"],
            "PA_CATG_NORM",

            "SERVICE_DT",
            "INTM_DATE",
            "LMP_DT",
            "APPEAL_DATE",
            "PA_CR_DT",
            "PA_APPR_DT",

            "FACILITIES",
            "DOCTOR_LICENSE",

            self.columns["est_amount"],
            self.columns["appr_amount"],
            self.columns["rej_amount"],

            "APPROVAL_RATIO",
            "REJECTION_RATIO",
            "IS_FULLY_APPROVED",
            "IS_FULLY_REJECTED",
            "IS_PARTIAL_APPROVED",
        ]

        existing_cols = [col for col in keep_cols if col in df.columns]

        return df[existing_cols].copy()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.clean_column_names(df)
        df = self.clean_text_columns(df)
        df = self.clean_date_columns(df)
        df = self.clean_amount_columns(df)
        df = self.clean_quantity_column(df)

        df = self.normalize_service_type(df)
        df = self.normalize_category(df)
        df = self.normalize_treatment_code(df)

        df = self.add_duration_features(df)
        df = self.add_dosage_features(df)

        df = self.create_basic_features(df)
        df = self.select_model_columns(df)

        return df


def load_input_data(path: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if path.suffix.lower() == ".pkl":
        return pd.read_pickle(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


if __name__ == "__main__":
    from src.config_loader import ConfigLoader

    loader = ConfigLoader()
    config = loader.get_config()

    raw_path = config["paths"]["raw_data"]
    processed_path = config["paths"]["processed_data"]

    df_raw = load_input_data(raw_path)

    preprocessor = ClaimsPreprocessor(config)
    df_processed = preprocessor.preprocess(df_raw)

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_pickle(processed_path)

    print("Preprocessing completed")
    print("Raw shape:", df_raw.shape)
    print("Processed shape:", df_processed.shape)
    print("Saved:", processed_path)