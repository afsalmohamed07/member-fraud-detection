import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import IsolationForest

from src.anomaly_feature_engineering import AnomalyFeatureEngineer


class IsolationModel:
    def __init__(self, config):
        self.config = config

        self.model = None
        self.feature_cols = []

        self.models_dir = Path(config["paths"]["models_dir"])
        self.model_path = self.models_dir / "isolation_forest.pkl"
        self.meta_path = self.models_dir / "isolation_forest_meta.pkl"

        self.feature_engineer = AnomalyFeatureEngineer(config)

    def prepare_features(self, df):
        df = df.copy()

        df = self.feature_engineer.transform(df)

        feature_cols = self.feature_engineer.get_feature_columns(df)

        X = df[feature_cols].copy()

        for col in feature_cols:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            ).replace([np.inf, -np.inf], 0).fillna(0)

        return X, feature_cols

    def train(self, df):
        X, feature_cols = self.prepare_features(df)

        self.feature_cols = feature_cols

        model = IsolationForest(
            contamination=self.config["isolation_forest"]["contamination"],
            random_state=self.config["isolation_forest"]["random_state"],
            n_estimators=self.config["isolation_forest"].get("n_estimators", 200),
            n_jobs=-1
        )

        model.fit(X)

        self.model = model

        return model, feature_cols

    def predict(self, df):
        if self.model is None:
            self.load()

        df = df.copy()

        df = self.feature_engineer.transform(df)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[self.feature_cols].copy()

        for col in self.feature_cols:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            ).replace([np.inf, -np.inf], 0).fillna(0)

        anomaly_pred = self.model.predict(X)
        anomaly_score = self.model.decision_function(X)

        result_df = df.copy()

        result_df["ISOLATION_PRED"] = anomaly_pred
        result_df["ISOLATION_SCORE_RAW"] = anomaly_score

        result_df["ISOLATION_ANOMALY"] = (
            anomaly_pred == -1
        ).astype(int)

        result_df["ISOLATION_SCORE"] = (
            np.where(
                anomaly_score < 0,
                np.minimum(abs(anomaly_score) * 100, 100),
                0
            )
        ).round(2)

        return result_df

    def save(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, self.model_path)

        meta = {
            "feature_cols": self.feature_cols,
        }

        joblib.dump(meta, self.meta_path)

        return str(self.model_path)

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Isolation model missing: {self.model_path}"
            )

        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"Isolation metadata missing: {self.meta_path}"
            )

        self.model = joblib.load(self.model_path)

        meta = joblib.load(self.meta_path)
        self.feature_cols = meta["feature_cols"]

        return self