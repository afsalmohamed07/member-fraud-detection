import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class BaselineBuilder:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]
        self.baseline_config = config["baseline"]

    # =====================================================
    # COMMON
    # =====================================================
    def _amount_stats(self, df, group_cols, amount_col):
        out = (
            df.groupby(group_cols, dropna=False)
            .agg(
                SUPPORT_COUNT=(amount_col, "count"),
                MEDIAN_AMT=(amount_col, "median"),
                MEAN_AMT=(amount_col, "mean"),
                MIN_AMT=(amount_col, "min"),
                MAX_AMT=(amount_col, "max"),
                P75_AMT=(amount_col, lambda x: x.quantile(0.75)),
                P90_AMT=(amount_col, lambda x: x.quantile(0.90)),
                P95_AMT=(amount_col, lambda x: x.quantile(0.95)),
                Q1_AMT=(amount_col, lambda x: x.quantile(0.25)),
                Q3_AMT=(amount_col, lambda x: x.quantile(0.75)),
            )
            .reset_index()
        )

        out["IQR_AMT"] = out["Q3_AMT"] - out["Q1_AMT"]

        out["IQR_UPPER_LIMIT"] = (
            out["Q3_AMT"] +
            self.baseline_config["iqr_multiplier"] * out["IQR_AMT"]
        )

        out["BASELINE_CONFIDENCE"] = np.where(
            out["SUPPORT_COUNT"] >= 30,
            "HIGH",
            np.where(out["SUPPORT_COUNT"] >= 10, "MEDIUM", "LOW")
        )

        return out

    def _normalize_text_series(self, series):
        return (
            series.astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def _pharmacy_rows(self, df):
        drug_col = self.columns["drug_name"]

        return df[
            (df["PROV_TREAT_CODE_CLEAN"] == "PHARMACY") &
            (df[drug_col].notna()) &
            (df[drug_col].astype(str).str.strip() != "")
        ].copy()

    # =====================================================
    # EXISTING BASELINES
    # =====================================================
    def build_known_diagnosis_baseline(self, df):
        diag_col = self.columns["diagnosis"]

        return (
            df.groupby(
                ["SERVICE_TYPE_NORM", "PA_CATG_NORM", diag_col],
                dropna=False
            )
            .size()
            .reset_index(name="DIAG_CONTEXT_COUNT")
        )

    def build_cost_context_baseline(self, df):
        group_cols = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            self.columns["diagnosis"],
            "PROV_TREAT_CODE_CLEAN",
        ]

        return self._amount_stats(
            df=df,
            group_cols=group_cols,
            amount_col=self.columns["est_amount"]
        )

    def build_provider_context_cost_baseline(self, df):
        group_cols = [
            self.columns["provider"],
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            self.columns["diagnosis"],
            "PROV_TREAT_CODE_CLEAN",
        ]

        return self._amount_stats(
            df=df,
            group_cols=group_cols,
            amount_col=self.columns["est_amount"]
        )

    def build_diagnosis_treatment_usage_baseline(self, df):
        diag_col = self.columns["diagnosis"]
        treat_col = "PROV_TREAT_CODE_CLEAN"

        total = (
            df.groupby(diag_col, dropna=False)
            .agg(TOTAL_DIAG_ROWS=(treat_col, "count"))
            .reset_index()
        )

        out = (
            df.groupby([diag_col, treat_col], dropna=False)
            .agg(TREATMENT_USAGE_COUNT=(treat_col, "count"))
            .reset_index()
        )

        out = out.merge(total, on=diag_col, how="left")

        out["TREATMENT_USAGE_%"] = (
            out["TREATMENT_USAGE_COUNT"] /
            out["TOTAL_DIAG_ROWS"] * 100
        ).round(2)

        return out

    def build_diagnosis_drug_usage_baseline(self, df):
        diag_col = self.columns["diagnosis"]
        drug_col = self.columns["drug_name"]

        ph = self._pharmacy_rows(df)

        total = (
            ph.groupby(diag_col, dropna=False)
            .agg(TOTAL_DIAG_PHARMACY_ROWS=(drug_col, "count"))
            .reset_index()
        )

        out = (
            ph.groupby([diag_col, drug_col], dropna=False)
            .agg(DRUG_USAGE_COUNT=(drug_col, "count"))
            .reset_index()
        )

        out = out.merge(total, on=diag_col, how="left")

        out["DRUG_USAGE_%"] = (
            out["DRUG_USAGE_COUNT"] /
            out["TOTAL_DIAG_PHARMACY_ROWS"] * 100
        ).round(2)

        return out

    def build_drug_duration_baseline(self, df):
        drug_col = self.columns["drug_name"]

        ph = self._pharmacy_rows(df)
        ph = ph[ph["DURATION_DAYS"].notna()].copy()

        out = (
            ph.groupby(drug_col, dropna=False)
            .agg(
                SUPPORT_COUNT=("DURATION_DAYS", "count"),
                MEDIAN_DURATION=("DURATION_DAYS", "median"),
                P75_DURATION=("DURATION_DAYS", lambda x: x.quantile(0.75)),
                P90_DURATION=("DURATION_DAYS", lambda x: x.quantile(0.90)),
                P95_DURATION=("DURATION_DAYS", lambda x: x.quantile(0.95)),
                Q1_DURATION=("DURATION_DAYS", lambda x: x.quantile(0.25)),
                Q3_DURATION=("DURATION_DAYS", lambda x: x.quantile(0.75)),
                MAX_DURATION=("DURATION_DAYS", "max"),
            )
            .reset_index()
        )

        out["IQR_DURATION"] = out["Q3_DURATION"] - out["Q1_DURATION"]

        out["IQR_DURATION_UPPER_LIMIT"] = (
            out["Q3_DURATION"] +
            self.baseline_config["iqr_multiplier"] * out["IQR_DURATION"]
        )

        return out

    def build_drug_dosage_baseline(self, df):
        drug_col = self.columns["drug_name"]

        ph = self._pharmacy_rows(df)
        ph = ph[ph["DOSAGE_NORMALIZED"].notna()].copy()

        total = (
            ph.groupby(drug_col, dropna=False)
            .agg(TOTAL_DRUG_ROWS=("DOSAGE_NORMALIZED", "count"))
            .reset_index()
        )

        out = (
            ph.groupby([drug_col, "DOSAGE_NORMALIZED"], dropna=False)
            .agg(DOSAGE_USAGE_COUNT=("DOSAGE_NORMALIZED", "count"))
            .reset_index()
        )

        out = out.merge(total, on=drug_col, how="left")

        out["DOSAGE_USAGE_%"] = (
            out["DOSAGE_USAGE_COUNT"] /
            out["TOTAL_DRUG_ROWS"] * 100
        ).round(2)

        return out

    def build_drug_context_price_baseline(self, df):
        ph = self._pharmacy_rows(df)

        ph = ph[
            ph["DURATION_DAYS"].notna() &
            ph["DOSAGE_NORMALIZED"].notna()
        ].copy()

        group_cols = [
            self.columns["drug_name"],
            "DURATION_DAYS",
            "DOSAGE_NORMALIZED",
        ]

        return self._amount_stats(
            df=ph,
            group_cols=group_cols,
            amount_col=self.columns["est_amount"]
        )

    # =====================================================
    # EXISTING DOCTOR DRUG DURATION / QTY BASELINE
    # old doctor_drug_behavior.pkl
    # =====================================================
    def build_doctor_drug_duration_qty_baseline(self, df):
        doctor_col = self.columns["doctor"]
        drug_col = self.columns["drug_name"]

        ph = self._pharmacy_rows(df)

        out = (
            ph.groupby([doctor_col, drug_col], dropna=False)
            .agg(
                SUPPORT_COUNT=(drug_col, "count"),
                MEDIAN_DURATION=("DURATION_DAYS", "median"),
                P90_DURATION=("DURATION_DAYS", lambda x: x.quantile(0.90)),
                MEDIAN_QTY=(self.columns["quantity"], "median"),
                P90_QTY=(self.columns["quantity"], lambda x: x.quantile(0.90)),
            )
            .reset_index()
        )

        return out

    # =====================================================
    # ADVANCED DURATION RANGE
    # =====================================================
    def build_duration_range_baseline(self, df):
        drug_col = self.columns["drug_name"]

        ph = self._pharmacy_rows(df)
        ph = ph[ph["DURATION_DAYS"].notna()].copy()

        out = (
            ph.groupby(drug_col, dropna=False)
            .agg(
                DURATION_SUPPORT_COUNT=("DURATION_DAYS", "count"),
                MIN_DURATION=("DURATION_DAYS", "min"),
                MAX_DURATION=("DURATION_DAYS", "max"),
                MEDIAN_DURATION_ADV=("DURATION_DAYS", "median"),
                DURATION_LIST=(
                    "DURATION_DAYS",
                    lambda x: sorted(list(set(x.dropna().astype(float))))
                ),
            )
            .reset_index()
        )

        return out

    # =====================================================
    # ADVANCED DOSAGE RANGE
    # =====================================================
    def build_dosage_range_baseline(self, df):
        drug_col = self.columns["drug_name"]

        ph = self._pharmacy_rows(df)
        ph = ph[ph["DOSAGE_PER_DAY"].notna()].copy()

        out = (
            ph.groupby(drug_col, dropna=False)
            .agg(
                DOSAGE_SUPPORT_COUNT=("DOSAGE_PER_DAY", "count"),
                MIN_DOSAGE=("DOSAGE_PER_DAY", "min"),
                MAX_DOSAGE=("DOSAGE_PER_DAY", "max"),
                MEDIAN_DOSAGE=("DOSAGE_PER_DAY", "median"),
                DOSAGE_LIST=(
                    "DOSAGE_PER_DAY",
                    lambda x: sorted(list(set(x.dropna().astype(float))))
                ),
            )
            .reset_index()
        )

        return out

    # =====================================================
    # PROVIDER BEHAVIOR
    # =====================================================
    def build_provider_behavior_baseline(self, df):
        amount_col = self.columns["est_amount"]
        provider_col = self.columns["provider"]
        diag_col = self.columns["diagnosis"]

        out = (
            df.groupby([diag_col, provider_col], dropna=False)
            .agg(
                PROVIDER_SUPPORT_COUNT=(amount_col, "count"),
                PROVIDER_MEDIAN_COST=(amount_col, "median"),
                PROVIDER_MEAN_COST=(amount_col, "mean"),
                PROVIDER_P75_COST=(amount_col, lambda x: x.quantile(0.75)),
                PROVIDER_P90_COST=(amount_col, lambda x: x.quantile(0.90)),
                PROVIDER_MIN_COST=(amount_col, "min"),
                PROVIDER_MAX_COST=(amount_col, "max"),
            )
            .reset_index()
        )

        return out

    # =====================================================
    # DOCTOR FACILITY BEHAVIOR
    # DOC_NAME + FACILITIES
    # =====================================================
    def build_doctor_facility_behavior_baseline(self, df):
        doctor_col = self.columns["doctor"]

        if "FACILITIES" not in df.columns:
            return pd.DataFrame(
                columns=[
                    "DOC_NAME_KEY",
                    "FACILITY_KEY",
                    "DOCTOR_FACILITY_COUNT",
                    "DOCTOR_TOTAL_FACILITY_ROWS",
                    "DOCTOR_FACILITY_USAGE_%",
                    "DOCTOR_FACILITY_LIST",
                ]
            )

        temp = df.copy()

        temp["DOC_NAME_KEY"] = self._normalize_text_series(temp[doctor_col])
        temp["FACILITY_KEY"] = self._normalize_text_series(temp["FACILITIES"])

        total = (
            temp.groupby("DOC_NAME_KEY", dropna=False)
            .agg(DOCTOR_TOTAL_FACILITY_ROWS=("FACILITY_KEY", "count"))
            .reset_index()
        )

        out = (
            temp.groupby(["DOC_NAME_KEY", "FACILITY_KEY"], dropna=False)
            .agg(DOCTOR_FACILITY_COUNT=("FACILITY_KEY", "count"))
            .reset_index()
        )

        out = out.merge(total, on="DOC_NAME_KEY", how="left")

        out["DOCTOR_FACILITY_USAGE_%"] = (
            out["DOCTOR_FACILITY_COUNT"] /
            out["DOCTOR_TOTAL_FACILITY_ROWS"] * 100
        ).round(2)

        facility_list = (
            temp.groupby("DOC_NAME_KEY")["FACILITY_KEY"]
            .apply(lambda x: sorted(list(set(x.dropna()))))
            .reset_index(name="DOCTOR_FACILITY_LIST")
        )

        out = out.merge(facility_list, on="DOC_NAME_KEY", how="left")

        return out

    # =====================================================
    # DOCTOR TREATMENT BEHAVIOR
    # DOC_NAME + PROV_TREAT_CODE_CLEAN
    # =====================================================
    def build_doctor_treatment_behavior_baseline(self, df):
        doctor_col = self.columns["doctor"]
        treat_col = "PROV_TREAT_CODE_CLEAN"

        temp = df.copy()

        temp["DOC_NAME_KEY"] = self._normalize_text_series(temp[doctor_col])

        total = (
            temp.groupby("DOC_NAME_KEY", dropna=False)
            .agg(DOCTOR_TOTAL_TREATMENT_ROWS=(treat_col, "count"))
            .reset_index()
        )

        out = (
            temp.groupby(["DOC_NAME_KEY", treat_col], dropna=False)
            .agg(DOCTOR_TREATMENT_COUNT=(treat_col, "count"))
            .reset_index()
        )

        out = out.merge(total, on="DOC_NAME_KEY", how="left")

        out["DOCTOR_TREATMENT_USAGE_%"] = (
            out["DOCTOR_TREATMENT_COUNT"] /
            out["DOCTOR_TOTAL_TREATMENT_ROWS"] * 100
        ).round(2)

        return out

    # =====================================================
    # DOCTOR DRUG BEHAVIOR
    # DOC_NAME + PAT_DRUG_NAME
    # =====================================================
    def build_doctor_drug_behavior_baseline(self, df):
        doctor_col = self.columns["doctor"]
        drug_col = self.columns["drug_name"]

        temp = df.copy()

        temp = temp[
            temp[drug_col].notna() &
            (temp[drug_col].astype(str).str.strip() != "")
        ].copy()

        temp["DOC_NAME_KEY"] = self._normalize_text_series(temp[doctor_col])

        total = (
            temp.groupby("DOC_NAME_KEY", dropna=False)
            .agg(DOCTOR_TOTAL_DRUG_ROWS=(drug_col, "count"))
            .reset_index()
        )

        out = (
            temp.groupby(["DOC_NAME_KEY", drug_col], dropna=False)
            .agg(DOCTOR_DRUG_COUNT=(drug_col, "count"))
            .reset_index()
        )

        out = out.merge(total, on="DOC_NAME_KEY", how="left")

        out["DOCTOR_DRUG_USAGE_%"] = (
            out["DOCTOR_DRUG_COUNT"] /
            out["DOCTOR_TOTAL_DRUG_ROWS"] * 100
        ).round(2)

        return out

    # =====================================================
    # LICENSE BASELINE
    # =====================================================
    def build_license_baseline(self, df):
        doctor_col = self.columns["doctor"]

        if "DOCTOR_LICENSE" not in df.columns:
            return pd.DataFrame(
                columns=[
                    "DOC_NAME_LICENSE_KEY",
                    "SAMPLE_DOC_NAME",
                    "UNIQUE_LICENSE_COUNT",
                    "LICENSE_LIST",
                ]
            )

        temp = df.copy()

        temp["DOC_NAME_LICENSE_KEY"] = self._normalize_text_series(
            temp[doctor_col]
        )

        temp["DOCTOR_LICENSE"] = (
            temp["DOCTOR_LICENSE"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        out = (
            temp.groupby("DOC_NAME_LICENSE_KEY", dropna=False)
            .agg(
                SAMPLE_DOC_NAME=(doctor_col, "first"),
                UNIQUE_LICENSE_COUNT=("DOCTOR_LICENSE", lambda x: x.nunique()),
                LICENSE_LIST=("DOCTOR_LICENSE", lambda x: sorted(list(set(x)))),
            )
            .reset_index()
        )

        return out

    # =====================================================
    # BUILD ALL
    # =====================================================
    def build_all_baselines(self, df):
        return {
            # existing
            "known_diagnosis": self.build_known_diagnosis_baseline(df),
            "cost_context": self.build_cost_context_baseline(df),
            "provider_context_cost": self.build_provider_context_cost_baseline(df),
            "diagnosis_treatment_usage": self.build_diagnosis_treatment_usage_baseline(df),
            "diagnosis_drug_usage": self.build_diagnosis_drug_usage_baseline(df),
            "drug_duration": self.build_drug_duration_baseline(df),
            "drug_dosage": self.build_drug_dosage_baseline(df),
            "drug_context_price": self.build_drug_context_price_baseline(df),

            # old doctor drug duration/qty baseline
            "doctor_drug_behavior": self.build_doctor_drug_duration_qty_baseline(df),

            # advanced
            "duration_baseline": self.build_duration_range_baseline(df),
            "dosage_baseline": self.build_dosage_range_baseline(df),
            "provider_behavior": self.build_provider_behavior_baseline(df),
            "license_baseline": self.build_license_baseline(df),

            # new doctor behavior baselines
            "doctor_facility_behavior": self.build_doctor_facility_behavior_baseline(df),
            "doctor_treatment_behavior": self.build_doctor_treatment_behavior_baseline(df),
            "doctor_drug_behavior_new": self.build_doctor_drug_behavior_baseline(df),
        }

    # =====================================================
    # SAVE
    # =====================================================
    def save_baselines(self, baselines):
        baselines_dir = Path(self.config["paths"]["baselines_dir"])
        baselines_dir.mkdir(parents=True, exist_ok=True)

        for name, baseline_df in baselines.items():
            joblib.dump(
                baseline_df,
                baselines_dir / f"{name}.pkl"
            )

        return True