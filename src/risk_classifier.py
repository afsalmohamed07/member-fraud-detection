import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class RiskClassifier:
    def __init__(self, config):
        self.config = config

        self.models_dir = Path(
            config["paths"]["models_dir"]
        )

        self.model_path = (
            self.models_dir /
            "lightgbm_risk_classifier.pkl"
        )

        self.feature_path = (
            self.models_dir /
            "lightgbm_risk_features.pkl"
        )

    # =====================================================
    # LABEL CREATION
    # =====================================================
    def _create_labels(self, df):

        df = df.copy()

        ratio = np.where(
            df["FINAL_EXPECTED_COST"] > 0,
            df["PA_EST_AMT_LC"] / df["FINAL_EXPECTED_COST"],
            1
        )

        labels = []

        for r in ratio:

            if r >= 5:
                labels.append("CRITICAL")

            elif r >= 2:
                labels.append("HIGH")

            elif r >= 1.2:
                labels.append("MEDIUM")

            elif r < 0.5:
                labels.append("LOW_COST")

            else:
                labels.append("NORMAL")

        df["TARGET_LABEL"] = labels

        return df

    # =====================================================
    # FEATURE PREP
    # =====================================================
    def prepare_features(self, df):

        df = df.copy()

        feature_cols = [

            # cost intelligence
            "FINAL_EXPECTED_COST",
            "ML_EXPECTED_COST",
            "PATTERN_P75",
            "PATTERN_P90",
            "PATTERN_MAX",

            # ratios
            "FINAL_COST_RATIO",

            # support
            "SUPPORT_COUNT",

            # similar claims
            "SIMILAR_CLAIMS_COUNT",
            "SIMILAR_REJECTION_RATE",
            "SIMILAR_COST_ABOVE_P90",

            # upcoding
            "USAGE_RATIO",
            "GLOBAL_TREAT_USAGE_PCT",
            "PROVIDER_TREAT_USAGE_PCT",

            # isolation
            "ISOLATION_SCORE",

            # anomaly flags
            "UNKNOWN_DIAGNOSIS_ANOMALY",
            "UNSEEN_TREATMENT_CODE_ANOMALY",
            "RARE_TREATMENT_ANOMALY",
            "RARE_DRUG_ANOMALY",
            "DURATION_OUTSIDE_RANGE_ANOMALY",
            "DOSAGE_OUTSIDE_RANGE_ANOMALY",
            "LICENSE_ANOMALY",
            "PROVIDER_BEHAVIOR_ANOMALY",
            "UPCODING_ANOMALY",
            "SIMILAR_CLAIMS_ANOMALY",
            "ISOLATION_ANOMALY",
        ]

        for col in feature_cols:

            if col not in df.columns:
                df[col] = 0

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

        return df[feature_cols], feature_cols

    # =====================================================
    # TRAIN
    # =====================================================
    def train(self, df):

        df = self._create_labels(df)

        X, feature_cols = self.prepare_features(df)

        y = df["TARGET_LABEL"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        print(
            classification_report(
                y_test,
                preds
            )
        )

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(model, self.model_path)
        joblib.dump(feature_cols, self.feature_path)

        return model

    # =====================================================
    # LOAD
    # =====================================================
    def load(self):

        self.model = joblib.load(self.model_path)

        self.feature_cols = joblib.load(
            self.feature_path
        )

        return self

    # =====================================================
    # PREDICT
    # =====================================================
    def predict(self, df):

        if not hasattr(self, "model"):
            self.load()

        X = df.copy()

        for col in self.feature_cols:

            if col not in X.columns:
                X[col] = 0

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            ).fillna(0)

        X = X[self.feature_cols]

        pred = self.model.predict(X)

        probs = self.model.predict_proba(X)

        result = df.copy()

        result["ML_FINAL_LABEL"] = pred

        result["ML_FINAL_CONFIDENCE"] = (
            probs.max(axis=1) * 100
        ).round(2)

        return result