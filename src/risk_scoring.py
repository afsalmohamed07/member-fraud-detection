import pandas as pd
import numpy as np


class RiskScorer:
    def __init__(self, config):
        self.config = config

    def score(self, df: pd.DataFrame):
        df = df.copy()

        weights = {
            # diagnosis / treatment
            "UNKNOWN_DIAGNOSIS_ANOMALY": 10,
            "UNSEEN_TREATMENT_CODE_ANOMALY": 20,
            "RARE_TREATMENT_ANOMALY": 10,

            # cost
            "COST_ANOMALY_FINAL": 20,

            # pharmacy structure
            "PHARMACY_MISSING_DRUG_ANOMALY": 50,
            "NON_PHARMACY_WITH_DRUG_ANOMALY": 50,

            # drug
            "UNSEEN_DRUG_ANOMALY": 10,
            "RARE_DRUG_ANOMALY": 3,
            "DRUG_NOT_IN_MASTER_ANOMALY": 25,
            "DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY": 20,

            # duration / dosage
            "DURATION_OUTSIDE_RANGE_ANOMALY": 20,
            "EXTREME_DURATION_ANOMALY": 20,

            # provider / license
            "LICENSE_ANOMALY": 40,
            "PROVIDER_BEHAVIOR_ANOMALY": 10,

            # pharmacy price
            "PHARMACY_HIGH_COST_ANOMALY": 20,

            # new intelligence layers
            "SIMILAR_CLAIMS_ANOMALY": 10,
            "UPCODING_ANOMALY": 5,

            # ML support signal
            "ISOLATION_ANOMALY": 3,
            "ISOLATION_CONTEXT_ANOMALY": 5,
        }

        for col in weights:
            if col not in df.columns:
                df[col] = 0

            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        df["FINAL_RISK_SCORE"] = 0

        for col, weight in weights.items():
            df["FINAL_RISK_SCORE"] += df[col] * weight

        # =====================================================
        # COST RATIO BASED BOOST
        # =====================================================
        if "FINAL_EXPECTED_COST" in df.columns:
            df["FINAL_EXPECTED_COST"] = pd.to_numeric(
                df["FINAL_EXPECTED_COST"],
                errors="coerce"
            ).fillna(0)

            df["PA_EST_AMT_LC"] = pd.to_numeric(
                df["PA_EST_AMT_LC"],
                errors="coerce"
            ).fillna(0)

            df["FINAL_COST_RATIO"] = np.where(
                df["FINAL_EXPECTED_COST"] > 0,
                df["PA_EST_AMT_LC"] / df["FINAL_EXPECTED_COST"],
                1
            )

            df.loc[
                (df["FINAL_COST_RATIO"] >= 1.2)
                & (df["FINAL_COST_RATIO"] < 2),
                "FINAL_RISK_SCORE"
            ] += 5

            df.loc[
                (df["FINAL_COST_RATIO"] >= 2)
                & (df["FINAL_COST_RATIO"] < 5),
                "FINAL_RISK_SCORE"
            ] += 15

            df.loc[
                df["FINAL_COST_RATIO"] >= 5,
                "FINAL_RISK_SCORE"
            ] += 30

        # =====================================================
        # SIMILAR CLAIMS REJECTION RATE BOOST
        # =====================================================
        if "SIMILAR_REJECTION_RATE" in df.columns:
            df["SIMILAR_REJECTION_RATE"] = pd.to_numeric(
                df["SIMILAR_REJECTION_RATE"],
                errors="coerce"
            ).fillna(0)

            df.loc[
                (df["SIMILAR_REJECTION_RATE"] >= 25)
                & (df["SIMILAR_REJECTION_RATE"] < 50),
                "FINAL_RISK_SCORE"
            ] += 5

            df.loc[
                df["SIMILAR_REJECTION_RATE"] >= 50,
                "FINAL_RISK_SCORE"
            ] += 10

        # =====================================================
        # CLIP SCORE
        # =====================================================
        df["FINAL_RISK_SCORE"] = df["FINAL_RISK_SCORE"].clip(0, 100)

        # =====================================================
        # FINAL LABEL
        # =====================================================
        df["FINAL_LABEL"] = np.select(
            [
                df["FINAL_RISK_SCORE"] >= 80,
                df["FINAL_RISK_SCORE"] >= 60,
                df["FINAL_RISK_SCORE"] >= 30,
            ],
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
            ],
            default="NORMAL"
        )

        return df