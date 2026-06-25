import joblib
import numpy as np
import pandas as pd

from pathlib import Path


class MLPredictor:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]

        models_dir = Path(config["paths"]["models_dir"])

        self.p50_model = joblib.load(models_dir / "catboost_p50.pkl")
        self.p75_model = joblib.load(models_dir / "catboost_p75.pkl")
        self.p90_model = joblib.load(models_dir / "catboost_p90.pkl")

        self.feature_cols = joblib.load(models_dir / "catboost_features.pkl")
        self.cat_features = joblib.load(models_dir / "catboost_cat_features.pkl")

    def _clean_text(self, value):
        if value is None or pd.isna(value):
            return "UNKNOWN"

        value = str(value).upper().strip()
        value = " ".join(value.split())

        if value in ["", "NAN", "NONE", "NULL", "NA", "N/A"]:
            return "UNKNOWN"

        if value.endswith(".0"):
            value = value[:-2]

        return value

    def _clean_code(self, value):
        return self._clean_text(value).replace(" ", "")

    def _get_context_price_features(self, row):
        return {
            "CONTEXT_SUPPORT": float(row.get("PROVIDER_PEER_SUPPORT", 0) or 0),
            "CONTEXT_COMMON_PRICE": float(row.get("PROVIDER_MARKET_COMMON", 0) or 0),
            "CONTEXT_MIN_PRICE": float(row.get("PROVIDER_MARKET_MIN", 0) or 0),
            "CONTEXT_MAX_PRICE": float(row.get("PROVIDER_MARKET_MAX", 0) or 0),
        }

    def _prepare_input(self, row):
        service = self._clean_text(
            row.get("SERVICE_TYPE_NORM") or row.get("SERVICE_TYPE")
        )

        category = self._clean_text(
            row.get("PA_CATG_NORM") or row.get("PA_CATG")
        )

        diagnosis = self._clean_text(
            row.get(self.columns["diagnosis"]) or row.get("PA_PRIMARY_DIAG")
        )

        treatment = self._clean_code(
            row.get("PROV_TREAT_CODE_CLEAN") or row.get("PROV_TREAT_CODE")
        )

        family = self._clean_text(
            row.get("PROV_TREAT_FAMILY")
        )

        provider = self._clean_text(
            row.get("PROV_NAME")
        )

        price_context_key = (
            service
            + "|"
            + category
            + "|"
            + diagnosis
            + "|"
            + treatment
            + "|"
            + provider
        )

        data = {
            "SERVICE_TYPE_NORM": service,
            "PA_CATG_NORM": category,
            self.columns["diagnosis"]: diagnosis,
            "PROV_TREAT_CODE_CLEAN": treatment,
            "PROV_TREAT_FAMILY": family,
            "PROV_NAME": provider,
            "PRICE_CONTEXT_KEY": price_context_key,
        }

        data.update(self._get_context_price_features(row))

        X = pd.DataFrame([data])

        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0

        for col in self.cat_features:
            if col in X.columns:
                X[col] = X[col].fillna("UNKNOWN").astype(str)

        X = X[self.feature_cols]

        return X

    def predict_cost_row(self, row):
        X = self._prepare_input(row)

        p50 = float(np.expm1(self.p50_model.predict(X)[0]))
        p75 = float(np.expm1(self.p75_model.predict(X)[0]))
        p90 = float(np.expm1(self.p90_model.predict(X)[0]))

        if p75 < p50:
            p75 = p50

        if p90 < p75:
            p90 = p75

        return {
            "P50_COST": round(p50, 2),
            "P75_COST": round(p75, 2),
            "P90_COST": round(p90, 2),
            "ML_EXPECTED_COST": round(p50, 2),
            "ML_NORMAL_LOW": round(p50, 2),
            "ML_NORMAL_HIGH": round(p75, 2),
            "ML_HIGH_RISK_LIMIT": round(p90, 2),
        }