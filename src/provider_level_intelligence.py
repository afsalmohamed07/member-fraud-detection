import joblib
from pathlib import Path


class ProviderLevelIntelligence:

    def __init__(self, config):

        models_dir = Path(config["paths"]["models_dir"]).parent

        artifact_path = (
            models_dir
            / "intelligence"
            / "provider_intelligence_artifact.pkl"
        )

        self.provider_artifact = joblib.load(artifact_path)

    def _clean_text(self, value):

        if value is None:
            return ""

        value = str(value).upper().strip()
        value = " ".join(value.split())

        if value in ["", "NAN", "NONE", "NULL", "UNKNOWN"]:
            return ""

        if value.endswith(".0"):
            value = value[:-2]

        return value

    def _clean_code(self, value):

        value = self._clean_text(value)
        value = value.replace(" ", "")

        return value

    def analyze(self, row):

        provider = self._clean_text(
            row.get("PROV_NAME", "")
        )

        diagnosis = self._clean_text(
            row.get("PA_PRIMARY_DIAG", "")
        )

        treatment = self._clean_code(
            row.get(
                "PROV_TREAT_CODE_CLEAN",
                row.get("PROV_TREAT_CODE", "")
            )
        )

        key = (
            provider,
            diagnosis,
            treatment,
        )

        info = self.provider_artifact.get(key)

        if not info:

            return {
                "PROVIDER_BEHAVIOR_ANOMALY": 0,

                "PROVIDER_MATCH_LEVEL": "NO_MATCH",

                "PROVIDER_SUPPORT": 0,

                "PROVIDER_REVIEW_RATE": 0,

                "PROVIDER_MEDIAN": 0,

                "PROVIDER_P90": 0,

                "PROVIDER_MAX": 0,

                "PROVIDER_TOP_REJECTION_CODE": "",

                "PROVIDER_TOP_REJECTION_REASON": "",

                "PROVIDER_EVIDENCE": {
                    "message": (
                        "No provider-level historical intelligence "
                        "found for this diagnosis-treatment pattern."
                    )
                },

                "PROVIDER_EXPLANATION": (
                    "No provider-level historical intelligence "
                    "found for this diagnosis-treatment pattern."
                )
            }

        support = int(
            info.get("MATCHING_PROVIDER_CLAIMS", 0)
        )

        review_count = int(
            info.get("MATCHING_REVIEW_CLAIMS", 0)
        )

        review_rate = float(
            info.get("REVIEW_RATE", 0)
        )

        median_cost = float(
            info.get("MEDIAN_COST", 0)
        )

        p90_cost = float(
            info.get("P90_COST", 0)
        )

        max_cost = float(
            info.get("MAX_COST", 0)
        )

        cross_doctor = int(
            info.get("CROSS_DOCTOR_COUNT", 0)
        )

        top_rej_code = info.get(
            "TOP_REJECTION_CODE",
            ""
        )

        top_rej_reason = info.get(
            "TOP_REJECTION_REASON",
            ""
        )

        explanation_parts = []

        explanation_parts.append(
            f"Same provider + diagnosis + treatment pattern "
            f"appeared {support} times in training history."
        )

        if review_count > 0:

            explanation_parts.append(
                f"{review_count} claims had rejection or "
                f"partial approval history "
                f"({round(review_rate, 2)}%)."
            )

        explanation_parts.append(
            f"Median cost is {median_cost}, "
            f"P90 cost is {p90_cost}, "
            f"maximum observed cost is {max_cost}."
        )

        if cross_doctor > 0:

            explanation_parts.append(
                f"This provider pattern was submitted by "
                f"{cross_doctor} different doctor(s)."
            )

        if top_rej_code:

            explanation_parts.append(
                f"Top rejection code is {top_rej_code}."
            )

        if top_rej_reason:

            explanation_parts.append(
                f"Top rejection reason is {top_rej_reason}."
            )

        full_explanation = " ".join(explanation_parts)

        anomaly_flag = 0

        if (
            support >= 5
            and (
                review_rate >= 40
                or cross_doctor >= 3
            )
        ):
            anomaly_flag = 1

        return {

            "PROVIDER_BEHAVIOR_ANOMALY": anomaly_flag,

            "PROVIDER_MATCH_LEVEL": info.get(
                "MATCH_LEVEL",
                "PROVIDER_DIAG_TREAT"
            ),

            "PROVIDER_SUPPORT": support,

            "PROVIDER_REVIEW_COUNT": review_count,

            "PROVIDER_REVIEW_RATE": review_rate,

            "PROVIDER_MEDIAN": median_cost,

            "PROVIDER_P90": p90_cost,

            "PROVIDER_MAX": max_cost,

            "PROVIDER_CROSS_DOCTOR_COUNT": cross_doctor,

            "PROVIDER_TOP_REJECTION_CODE": top_rej_code,

            "PROVIDER_TOP_REJECTION_REASON": top_rej_reason,

            "PROVIDER_EVIDENCE": info,

            "PROVIDER_EXPLANATION": full_explanation,
        }