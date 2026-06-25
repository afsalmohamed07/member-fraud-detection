import re
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class CostModelTrainer:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]

    def _clean_text(self, value):
        if pd.isna(value):
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

    def _add_treat_family(self, df):
        df = df.copy()

        def family(code):
            code = str(code).upper().strip()
            code = re.sub(r"[^A-Z0-9]", "", code)

            if code.startswith("PHARMACY"):
                return "PHARMACY"

            match = re.match(r"^[A-Z]+", code)
            if match:
                return match.group(0)

            return "UNKNOWN"

        if "PROV_TREAT_FAMILY" not in df.columns:
            df["PROV_TREAT_FAMILY"] = df["PROV_TREAT_CODE_CLEAN"].apply(family)

        return df

    def _add_context_price_features(self, df, target_col):
        df = df.copy()

        context_cols = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            self.columns["diagnosis"],
            "PROV_TREAT_CODE_CLEAN",
            "PROV_NAME",
        ]

        grouped = (
            df.groupby(context_cols, dropna=False)[target_col]
            .agg(
                CONTEXT_SUPPORT="count",
                CONTEXT_COMMON_PRICE=lambda x: float(pd.Series(x).mode().iloc[0]),
                CONTEXT_MIN_PRICE="min",
                CONTEXT_MAX_PRICE="max",
            )
            .reset_index()
        )

        df = df.merge(
            grouped,
            on=context_cols,
            how="left"
        )

        for col in [
            "CONTEXT_SUPPORT",
            "CONTEXT_COMMON_PRICE",
            "CONTEXT_MIN_PRICE",
            "CONTEXT_MAX_PRICE",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def _train_quantile_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        cat_features,
        alpha,
    ):
        model = CatBoostRegressor(
            loss_function=f"Quantile:alpha={alpha}",
            eval_metric=f"Quantile:alpha={alpha}",
            iterations=800,
            depth=8,
            learning_rate=0.05,
            random_seed=42,
            verbose=100,
        )

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_test, y_test),
            use_best_model=True,
        )

        preds = np.expm1(model.predict(X_test))
        actual = np.expm1(y_test)

        mae = mean_absolute_error(actual, preds)

        return model, mae

    def train_model(self, df):
        df = df.copy()
        df = self._add_treat_family(df)

        target_col = self.columns["est_amount"]
        diag_col = self.columns["diagnosis"]

        required_cols = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            diag_col,
            "PROV_TREAT_CODE_CLEAN",
            "PROV_TREAT_FAMILY",
            "PROV_NAME",
            target_col,
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = "UNKNOWN"

        df["SERVICE_TYPE_NORM"] = df["SERVICE_TYPE_NORM"].apply(self._clean_text)
        df["PA_CATG_NORM"] = df["PA_CATG_NORM"].apply(self._clean_text)
        df[diag_col] = df[diag_col].apply(self._clean_text)
        df["PROV_TREAT_CODE_CLEAN"] = df["PROV_TREAT_CODE_CLEAN"].apply(self._clean_code)
        df["PROV_TREAT_FAMILY"] = df["PROV_TREAT_FAMILY"].apply(self._clean_text)
        df["PROV_NAME"] = df["PROV_NAME"].apply(self._clean_text)

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])
        df = df[df[target_col] > 0].copy()

        df["PRICE_CONTEXT_KEY"] = (
            df["SERVICE_TYPE_NORM"].astype(str)
            + "|"
            + df["PA_CATG_NORM"].astype(str)
            + "|"
            + df[diag_col].astype(str)
            + "|"
            + df["PROV_TREAT_CODE_CLEAN"].astype(str)
            + "|"
            + df["PROV_NAME"].astype(str)
        )

        df = self._add_context_price_features(df, target_col)

        feature_cols = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            diag_col,
            "PROV_TREAT_CODE_CLEAN",
            "PROV_TREAT_FAMILY",
            "PROV_NAME",
            "PRICE_CONTEXT_KEY",
            "CONTEXT_SUPPORT",
            "CONTEXT_COMMON_PRICE",
            "CONTEXT_MIN_PRICE",
            "CONTEXT_MAX_PRICE",
        ]

        cat_features = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            diag_col,
            "PROV_TREAT_CODE_CLEAN",
            "PROV_TREAT_FAMILY",
            "PROV_NAME",
            "PRICE_CONTEXT_KEY",
        ]

        X = df[feature_cols].copy()
        y = np.log1p(df[target_col])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        print("\nTraining P50 model...\n")
        p50_model, p50_mae = self._train_quantile_model(
            X_train, y_train, X_test, y_test, cat_features, alpha=0.50
        )

        print("\nTraining P75 model...\n")
        p75_model, p75_mae = self._train_quantile_model(
            X_train, y_train, X_test, y_test, cat_features, alpha=0.75
        )

        print("\nTraining P90 model...\n")
        p90_model, p90_mae = self._train_quantile_model(
            X_train, y_train, X_test, y_test, cat_features, alpha=0.90
        )

        models_dir = Path(self.config["paths"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(p50_model, models_dir / "catboost_p50.pkl")
        joblib.dump(p75_model, models_dir / "catboost_p75.pkl")
        joblib.dump(p90_model, models_dir / "catboost_p90.pkl")
        joblib.dump(feature_cols, models_dir / "catboost_features.pkl")
        joblib.dump(cat_features, models_dir / "catboost_cat_features.pkl")

        metrics = {
            "P50_MAE": float(p50_mae),
            "P75_MAE": float(p75_mae),
            "P90_MAE": float(p90_mae),
            "features": feature_cols,
        }

        with open(models_dir / "quantile_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics