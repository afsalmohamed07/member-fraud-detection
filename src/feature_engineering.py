import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]

    def add_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        est_col = self.columns["est_amount"]
        appr_col = self.columns["appr_amount"]
        rej_col = self.columns["rej_amount"]

        df["APPROVAL_RATIO"] = np.where(
            df[est_col] > 0,
            df[appr_col] / df[est_col],
            0
        )

        df["REJECTION_RATIO"] = np.where(
            df[est_col] > 0,
            df[rej_col] / df[est_col],
            0
        )

        df["NET_APPROVED_GAP"] = df[est_col] - df[appr_col]

        return df

    def add_pharmacy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        qty_col = self.columns["quantity"]
        est_col = self.columns["est_amount"]

        df["UNIT_EST_COST"] = np.where(
            df[qty_col] > 0,
            df[est_col] / df[qty_col],
            np.nan
        )

        df["HAS_DRUG"] = df[self.columns["drug_name"]].notna().astype(int)

        return df

    def add_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ratio_cols = [
            "DIAG_TREAT_COST_RATIO",
            "DRUG_PRICE_RATIO",
            "DURATION_RATIO",
            "PROVIDER_TREAT_COST_RATIO"
        ]

        for col in ratio_cols:
            if col not in df.columns:
                df[col] = np.nan

        df["MAX_DEVIATION_RATIO"] = df[ratio_cols].max(axis=1)

        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_financial_features(df)
        df = self.add_pharmacy_features(df)
        df = self.add_baseline_features(df)

        return df