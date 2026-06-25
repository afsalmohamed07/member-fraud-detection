import pandas as pd
import joblib
from pathlib import Path
import re


class CostPredictionBaselineBuilder:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]

    def _create_treat_family(self, df):
        df = df.copy()

        if "PROV_TREAT_FAMILY" in df.columns:
            return df

        def family(code):
            code = str(code).upper().strip()
            code = re.sub(r"[^A-Z0-9]", "", code)

            if code.startswith("PHARMACY"):
                return "PHARMACY"

            match = re.match(r"^[A-Z]+", code)
            if match:
                return match.group(0)

            return "UNKNOWN"

        df["PROV_TREAT_FAMILY"] = df["PROV_TREAT_CODE_CLEAN"].apply(family)
        return df

    def _confidence(self, support):
        if support >= 30:
            return "HIGH"
        elif support >= 10:
            return "MEDIUM"
        elif support >= 3:
            return "LOW"
        else:
            return "VERY_LOW"

    def _build_level(self, df, group_cols):
        amount_col = self.columns["est_amount"]

        baseline = (
            df.groupby(group_cols, dropna=False)
            .agg(
                SUPPORT_COUNT=(amount_col, "count"),
                BASELINE_COST_MEDIAN=(amount_col, "median"),
                BASELINE_COST_MEAN=(amount_col, "mean"),
                BASELINE_COST_MIN=(amount_col, "min"),
                BASELINE_COST_MAX=(amount_col, "max"),
                BASELINE_COST_P75=(amount_col, lambda x: x.quantile(0.75)),
                BASELINE_COST_P90=(amount_col, lambda x: x.quantile(0.90)),
                BASELINE_COST_LIST=(amount_col, lambda x: list(x)),
            )
            .reset_index()
        )

        baseline["CONFIDENCE_LEVEL"] = baseline["SUPPORT_COUNT"].apply(
            self._confidence
        )

        return baseline

    def _build_pharmacy_drug_level(self, df):
        amount_col = self.columns["est_amount"]
        drug_col = self.columns["drug_name"]
        provider_col = self.columns["provider"]

        pharmacy_df = df[
            (df["PROV_TREAT_CODE_CLEAN"] == "PHARMACY") &
            (df[drug_col].notna())
        ].copy()

        baseline = (
            pharmacy_df.groupby([drug_col, provider_col], dropna=False)
            .agg(
                SUPPORT_COUNT=(amount_col, "count"),
                BASELINE_COST_MEDIAN=(amount_col, "median"),
                BASELINE_COST_MEAN=(amount_col, "mean"),
                BASELINE_COST_MIN=(amount_col, "min"),
                BASELINE_COST_MAX=(amount_col, "max"),
                BASELINE_COST_P75=(amount_col, lambda x: x.quantile(0.75)),
                BASELINE_COST_P90=(amount_col, lambda x: x.quantile(0.90)),
                BASELINE_COST_LIST=(amount_col, lambda x: list(x)),
            )
            .reset_index()
        )

        baseline["CONFIDENCE_LEVEL"] = baseline["SUPPORT_COUNT"].apply(
            self._confidence
        )

        return baseline

    def build(self, df):
        df = self._create_treat_family(df)

        provider_col = self.columns["provider"]

        generic_levels = {
            "level_1": [
                "SERVICE_TYPE_NORM",
                "PA_CATG_NORM",
                self.columns["diagnosis"],
                "PROV_TREAT_CODE_CLEAN",
                provider_col,
            ],
            "level_2": [
                "SERVICE_TYPE_NORM",
                "PA_CATG_NORM",
                self.columns["diagnosis"],
                "PROV_TREAT_FAMILY",
                provider_col,
            ],
            "level_2_5": [
                "SERVICE_TYPE_NORM",
                "PA_CATG_NORM",
                self.columns["diagnosis"],
                provider_col,
            ],
            "level_3": [
                "SERVICE_TYPE_NORM",
                "PA_CATG_NORM",
                "PROV_TREAT_FAMILY",
                provider_col,
            ],
            "level_4": [
                "SERVICE_TYPE_NORM",
                "PA_CATG_NORM",
                provider_col,
            ],
        }

        baseline_tables = {}

        for level, group_cols in generic_levels.items():
            baseline_tables[level] = {
                "group_cols": group_cols,
                "table": self._build_level(df, group_cols),
                "type": "generic_cost",
            }

        baseline_tables["pharmacy_drug_cost"] = {
            "group_cols": [self.columns["drug_name"], provider_col],
            "table": self._build_pharmacy_drug_level(df),
            "type": "pharmacy_cost",
        }

        return baseline_tables

    def save(self, baseline_tables):
        path = Path(self.config["paths"]["baselines_dir"])
        path.mkdir(parents=True, exist_ok=True)

        output_path = path / "cost_prediction_baselines.pkl"
        joblib.dump(baseline_tables, output_path)

        return output_path