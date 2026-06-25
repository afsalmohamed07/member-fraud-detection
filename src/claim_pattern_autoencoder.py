import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow.keras.models import load_model


class ClaimPatternAutoencoder:

    def __init__(self, config):
        self.config = config

        model_dir = Path(config["paths"]["models_dir"])

        self.model = load_model(
            model_dir / "claim_pattern_autoencoder.h5",
            compile=False
        )

        self.scaler = joblib.load(
            model_dir / "claim_pattern_scaler.pkl"
        )

        self.threshold = joblib.load(
            model_dir / "claim_pattern_threshold.pkl"
        )

        self.feature_cols = joblib.load(
            model_dir / "claim_pattern_features.pkl"
        )

        self.prob_maps = joblib.load(
            model_dir / "claim_pattern_probability_maps.pkl"
        )

    def _safe_str(self, x):
        if pd.isna(x):
            return "UNKNOWN"

        value = str(x).strip().upper()

        if value in ["", "NAN", "NONE", "NULL", "NA", "N/A"]:
            return "UNKNOWN"

        return value

    def _safe_num(self, x):
        value = pd.to_numeric(
            x,
            errors="coerce"
        )

        if pd.isna(value):
            return 0

        return float(value)

    def _build_features(self, row):
        diag = self._safe_str(
            row.get("PA_PRIMARY_DIAG")
        )

        treat = self._safe_str(
            row.get("PROV_TREAT_CODE_CLEAN")
            or row.get("PROV_TREAT_CODE")
        )

        drug = self._safe_str(
            row.get("PAT_DRUG_NAME")
        )

        doctor = self._safe_str(
            row.get("DOC_NAME")
        )

        provider = self._safe_str(
            row.get("PROV_NAME")
        )

        duration = self._safe_num(
            row.get("DURATION_DAYS", 0)
        )

        dosage = self._safe_num(
            row.get("DOSAGE_PER_DAY", 0)
        )

        return {
            "DIAGNOSIS_TREATMENT_PROB":
                self.prob_maps["diag_treat_map"].get(
                    (diag, treat),
                    0
                ),

            "DIAGNOSIS_DRUG_PROB":
                self.prob_maps["diag_drug_map"].get(
                    (diag, drug),
                    0
                ),

            "DOCTOR_TREATMENT_PROB":
                self.prob_maps["doc_treat_map"].get(
                    (doctor, treat),
                    0
                ),

            "DOCTOR_DRUG_PROB":
                self.prob_maps["doc_drug_map"].get(
                    (doctor, drug),
                    0
                ),

            "PROVIDER_TREATMENT_PROB":
                self.prob_maps["prov_treat_map"].get(
                    (provider, treat),
                    0
                ),

            "DURATION_SCORE": duration,

            "DOSAGE_SCORE": dosage,
        }

    def predict(self, row):
        feature_dict = self._build_features(row)

        X = pd.DataFrame([feature_dict])

        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_cols]

        X_scaled = self.scaler.transform(X)

        reconstructed = self.model.predict(
            X_scaled,
            verbose=0
        )

        reconstruction_error = np.mean(
            np.square(X_scaled - reconstructed)
        )

        score_ratio = reconstruction_error / (
            self.threshold * 3
        )

        anomaly_score = min(
            100,
            round(score_ratio * 100, 2)
        )

        is_anomaly = (
            reconstruction_error > self.threshold * 3
        )

        return {
            "CLAIM_PATTERN_ANOMALY_SCORE": anomaly_score,
            "CLAIM_PATTERN_ANOMALY_FLAG": int(is_anomaly),
            "CLAIM_PATTERN_RECONSTRUCTION_ERROR": float(reconstruction_error),
        }