import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from catboost import CatBoostRegressor


class PharmacyCostModel:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]

        self.model_path = (
            Path(config["paths"]["models_dir"]) /
            "catboost_pharmacy_drug_cost_model.pkl"
        )

        self.feature_path = (
            Path(config["paths"]["models_dir"]) /
            "catboost_pharmacy_drug_features.pkl"
        )

    def _clean_text(self, value):
        if pd.isna(value):
            return "UNKNOWN"

        value = str(value).upper().strip()
        value = " ".join(value.split())

        if value in ["", "NAN", "NONE", "NULL"]:
            return "UNKNOWN"

        if value.endswith(".0"):
            value = value[:-2]

        return value

    def _remove_outliers(self, df, amount_col):
        df = df.copy()

        if len(df) < 10:
            return df

        q1 = df[amount_col].quantile(0.25)
        q3 = df[amount_col].quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            return df

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return df[
            (df[amount_col] >= lower)
            &
            (df[amount_col] <= upper)
        ].copy()

    def _build_price_context_key(self, df):
        df = df.copy()

        for col in [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            self.columns["drug_name"],
            self.columns["provider"],
        ]:
            if col not in df.columns:
                df[col] = "UNKNOWN"

            df[col] = df[col].apply(self._clean_text)

        df["PRICE_CONTEXT_KEY"] = (
            df["SERVICE_TYPE_NORM"].astype(str)
            + "|"
            + df["PA_CATG_NORM"].astype(str)
            + "|"
            + df[self.columns["drug_name"]].astype(str)
            + "|"
            + df[self.columns["provider"]].astype(str)
        )

        return df

    def train(self, df):
        drug_col = self.columns["drug_name"]
        provider_col = self.columns["provider"]
        amount_col = self.columns["est_amount"]

        df = df.copy()

        pharmacy_df = df[
            (df["PA_CATG_NORM"] == "PHARMACY") |
            (df["PROV_TREAT_CODE_CLEAN"] == "PHARMACY")
        ].copy()

        pharmacy_df = pharmacy_df[
            pharmacy_df[drug_col].notna() &
            (pharmacy_df[drug_col].astype(str).str.strip() != "")
        ].copy()

        if provider_col not in pharmacy_df.columns:
            pharmacy_df[provider_col] = "UNKNOWN"

        pharmacy_df[amount_col] = pd.to_numeric(
            pharmacy_df[amount_col],
            errors="coerce"
        )

        pharmacy_df = pharmacy_df[
            pharmacy_df[amount_col].notna() &
            (pharmacy_df[amount_col] > 0)
        ].copy()

        pharmacy_df = self._build_price_context_key(pharmacy_df)

        pharmacy_df = self._remove_outliers(
            pharmacy_df,
            amount_col
        )

        feature_cols = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            drug_col,
            provider_col,
            "PRICE_CONTEXT_KEY",
        ]

        X = pharmacy_df[feature_cols].copy()
        y = np.log1p(pharmacy_df[amount_col])

        for col in feature_cols:
            X[col] = X[col].apply(self._clean_text)

        model = CatBoostRegressor(
            iterations=800,
            depth=8,
            learning_rate=0.04,
            loss_function="RMSE",
            random_seed=42,
            verbose=100
        )

        cat_features = feature_cols.copy()

        model.fit(
            X,
            y,
            cat_features=cat_features
        )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, self.model_path)
        joblib.dump(feature_cols, self.feature_path)

        return model, feature_cols

    def predict_row(self, row):
        model = joblib.load(self.model_path)
        feature_cols = joblib.load(self.feature_path)

        data = {}

        row_dict = dict(row)

        service = self._clean_text(
            row_dict.get("SERVICE_TYPE_NORM") or row_dict.get("SERVICE_TYPE")
        )

        category = self._clean_text(
            row_dict.get("PA_CATG_NORM") or row_dict.get("PA_CATG")
        )

        drug = self._clean_text(
            row_dict.get(self.columns["drug_name"])
        )

        provider = self._clean_text(
            row_dict.get(self.columns["provider"])
        )

        row_dict["SERVICE_TYPE_NORM"] = service
        row_dict["PA_CATG_NORM"] = category
        row_dict[self.columns["drug_name"]] = drug
        row_dict[self.columns["provider"]] = provider

        row_dict["PRICE_CONTEXT_KEY"] = (
            service
            + "|"
            + category
            + "|"
            + drug
            + "|"
            + provider
        )

        for col in feature_cols:
            data[col] = self._clean_text(
                row_dict.get(col, "UNKNOWN")
            )

        X = pd.DataFrame([data])

        pred_log = model.predict(X)[0]
        pred = np.expm1(pred_log)

        return float(pred)