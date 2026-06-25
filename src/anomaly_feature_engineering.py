import numpy as np
import pandas as pd


class AnomalyFeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]

    def _safe_series(self, df: pd.DataFrame, col: str, default=0):
        if col in df.columns:
            return pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(default)

        return pd.Series(
            default,
            index=df.index
        )

    def _safe_divide(self, numerator, denominator):
        return np.where(
            denominator > 0,
            numerator / denominator,
            0
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        amount_col = self.columns["est_amount"]

        amount = pd.to_numeric(
            df.get(
                amount_col,
                pd.Series(0, index=df.index)
            ),
            errors="coerce"
        ).fillna(0)

        # =====================================================
        # COST RATIO FEATURES
        # =====================================================
        df["FEAT_COST_TO_MEDIAN_RATIO"] = self._safe_divide(
            amount,
            self._safe_series(df, "COST_MEDIAN_AMT")
        )

        df["FEAT_COST_TO_P90_RATIO"] = self._safe_divide(
            amount,
            self._safe_series(df, "COST_P90_AMT")
        )

        df["FEAT_COST_TO_IQR_LIMIT_RATIO"] = self._safe_divide(
            amount,
            self._safe_series(df, "COST_IQR_UPPER_LIMIT")
        )

        # =====================================================
        # DIAGNOSIS / TREATMENT / DRUG RARITY FEATURES
        # =====================================================
        df["FEAT_UNKNOWN_DIAGNOSIS"] = (
            self._safe_series(df, "UNKNOWN_DIAGNOSIS_ANOMALY")
            .astype(int)
        )

        df["FEAT_TREATMENT_USAGE_PCT"] = self._safe_series(
            df,
            "TREATMENT_USAGE_%"
        )

        df["FEAT_TREATMENT_RARITY_SCORE"] = (
            100 - df["FEAT_TREATMENT_USAGE_PCT"]
        ).clip(lower=0, upper=100)

        df["FEAT_UNSEEN_TREATMENT"] = (
            self._safe_series(df, "UNSEEN_TREATMENT_CODE_ANOMALY")
            .astype(int)
        )

        df["FEAT_RARE_TREATMENT"] = (
            self._safe_series(df, "RARE_TREATMENT_ANOMALY")
            .astype(int)
        )

        df["FEAT_DRUG_USAGE_PCT"] = self._safe_series(
            df,
            "DRUG_USAGE_%"
        )

        df["FEAT_DRUG_RARITY_SCORE"] = (
            100 - df["FEAT_DRUG_USAGE_PCT"]
        ).clip(lower=0, upper=100)

        df["FEAT_DRUG_NOT_IN_MASTER"] = (
            self._safe_series(df, "DRUG_NOT_IN_MASTER_ANOMALY")
            .astype(int)
        )

        df["FEAT_DRUG_NOT_USED_FOR_DIAGNOSIS"] = (
            self._safe_series(df, "DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY")
            .astype(int)
        )

        df["FEAT_RARE_DRUG"] = (
            self._safe_series(df, "RARE_DRUG_ANOMALY")
            .astype(int)
        )

        # =====================================================
        # DOCTOR BEHAVIOR FEATURES
        # =====================================================
        df["FEAT_DOCTOR_FACILITY_USAGE_PCT"] = self._safe_series(
            df,
            "DOCTOR_FACILITY_USAGE_%"
        )

        df["FEAT_DOCTOR_TREATMENT_USAGE_PCT"] = self._safe_series(
            df,
            "DOCTOR_TREATMENT_USAGE_%"
        )

        df["FEAT_DOCTOR_DRUG_USAGE_PCT"] = self._safe_series(
            df,
            "DOCTOR_DRUG_USAGE_%"
        )

        df["FEAT_DOCTOR_FACILITY_ANOMALY"] = (
            self._safe_series(df, "DOCTOR_FACILITY_ANOMALY")
            .astype(int)
        )

        df["FEAT_DOCTOR_TREATMENT_ANOMALY"] = (
            self._safe_series(df, "DOCTOR_TREATMENT_ANOMALY")
            .astype(int)
        )

        df["FEAT_DOCTOR_DRUG_ANOMALY"] = (
            self._safe_series(df, "DOCTOR_DRUG_ANOMALY")
            .astype(int)
        )

        # =====================================================
        # PROVIDER FEATURES
        # =====================================================
        df["FEAT_PROVIDER_SUPPORT_COUNT"] = self._safe_series(
            df,
            "PROVIDER_SUPPORT_COUNT"
        )

        df["FEAT_PROVIDER_COST_RATIO"] = self._safe_divide(
            amount,
            self._safe_series(df, "PROVIDER_MEDIAN_COST")
        )

        df["FEAT_PROVIDER_BEHAVIOR_ANOMALY"] = (
            self._safe_series(df, "PROVIDER_BEHAVIOR_ANOMALY")
            .astype(int)
        )

        # =====================================================
        # DURATION / DOSAGE FEATURES
        # =====================================================
        df["FEAT_DURATION_OUTSIDE_RANGE"] = (
            self._safe_series(df, "DURATION_OUTSIDE_RANGE_ANOMALY")
            .astype(int)
        )

        df["FEAT_EXTREME_DURATION"] = (
            self._safe_series(df, "EXTREME_DURATION_ANOMALY")
            .astype(int)
        )

        df["FEAT_DOSAGE_OUTSIDE_RANGE"] = (
            self._safe_series(df, "DOSAGE_OUTSIDE_RANGE_ANOMALY")
            .astype(int)
        )

        # =====================================================
        # STRUCTURE / POLICY FEATURES
        # =====================================================
        df["FEAT_PHARMACY_MISSING_DRUG"] = (
            self._safe_series(df, "PHARMACY_MISSING_DRUG_ANOMALY")
            .astype(int)
        )

        df["FEAT_NON_PHARMACY_WITH_DRUG"] = (
            self._safe_series(df, "NON_PHARMACY_WITH_DRUG_ANOMALY")
            .astype(int)
        )

        df["FEAT_LICENSE_ANOMALY"] = (
            self._safe_series(df, "LICENSE_ANOMALY")
            .astype(int)
        )

        # =====================================================
        # ISOLATION CONTEXT FEATURES
        # =====================================================
        df["FEAT_ISOLATION_CONTEXT_SCORE"] = self._safe_series(
            df,
            "ISOLATION_CONTEXT_SCORE"
        )

        df["FEAT_ISOLATION_CONTEXT_ANOMALY"] = (
            self._safe_series(df, "ISOLATION_CONTEXT_ANOMALY")
            .astype(int)
        )

        # =====================================================
        # FINAL CLEANUP
        # =====================================================
        feat_cols = [
            col for col in df.columns
            if col.startswith("FEAT_")
        ]

        for col in feat_cols:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).replace(
                [np.inf, -np.inf],
                0
            ).fillna(0)

        return df

    def get_feature_columns(self, df: pd.DataFrame):
        return [
            col for col in df.columns
            if col.startswith("FEAT_")
        ]