import joblib
from pathlib import Path


class UpcodingDetector:

    def __init__(self, config):

        models_dir = Path(config["paths"]["models_dir"]).parent

        artifact_path = (
            models_dir
            / "intelligence"
            / "upcoding_artifact.pkl"
        )

        self.upcoding_artifact = joblib.load(artifact_path)

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

    def detect(self, row):

        diagnosis = self._clean_text(
            row.get("PA_PRIMARY_DIAG", "")
        )

        provider = self._clean_text(
            row.get("PROV_NAME", "")
        )

        treatment = self._clean_code(
            row.get(
                "PROV_TREAT_CODE_CLEAN",
                row.get("PROV_TREAT_CODE", "")
            )
        )

        key = (
            diagnosis,
            provider,
            treatment,
        )

        info = self.upcoding_artifact.get(key)

        if not info:

            return {
                "UPCODING_ANOMALY": 0,
                "UPCODING_CONFIDENCE": "LOW",
                "UPCODING_SUPPORT_STATUS": "NO_TRAINING_HISTORY",
                "GLOBAL_DIAG_TOTAL": 0,
                "GLOBAL_DIAG_TREAT_COUNT": 0,
                "GLOBAL_TREAT_USAGE_PCT": 0,
                "PROVIDER_DIAG_TOTAL": 0,
                "PROVIDER_DIAG_TREAT_COUNT": 0,
                "PROVIDER_TREAT_USAGE_PCT": 0,
                "USAGE_RATIO": 0,
                "DIAG_P75_COST": 0,
                "DIAG_P90_COST": 0,
                "TREATMENT_P75_COST": 0,
                "TREATMENT_P90_COST": 0,
                "UPCODING_EXPLANATION": (
                    "No training-history upcoding intelligence available "
                    "for this provider diagnosis-treatment pattern."
                ),
            }

        global_total = int(
            info.get("GLOBAL_DIAG_TOTAL", 0)
        )

        global_treat_count = int(
            info.get("GLOBAL_DIAG_TREAT_COUNT", 0)
        )

        provider_total = int(
            info.get("PROVIDER_DIAG_TOTAL", 0)
        )

        provider_treat_count = int(
            info.get("PROVIDER_DIAG_TREAT_COUNT", 0)
        )

        global_usage = float(
            info.get("GLOBAL_TREAT_USAGE_PCT", 0)
        )

        provider_usage = float(
            info.get("PROVIDER_TREAT_USAGE_PCT", 0)
        )

        usage_ratio = float(
            info.get("USAGE_RATIO", 0)
        )

        support_status = "VALID"

        if global_total < 30:
            support_status = "LOW_GLOBAL_DIAGNOSIS_HISTORY"

        elif global_treat_count < 3:
            support_status = "LOW_DIAGNOSIS_TREATMENT_HISTORY"

        elif provider_total < 5:
            support_status = "LOW_PROVIDER_DIAGNOSIS_HISTORY"

        elif provider_treat_count < 1:
            support_status = "NO_PROVIDER_TREATMENT_HISTORY"

        confidence = "LOW"

        if (
            global_total >= 100
            and global_treat_count >= 10
            and provider_total >= 20
        ):
            confidence = "HIGH"

        elif (
            global_total >= 50
            and global_treat_count >= 5
            and provider_total >= 10
        ):
            confidence = "MEDIUM"

        upcoding_flag = int(
            support_status == "VALID"
            and usage_ratio >= 3
        )

        if upcoding_flag == 1:

            explanation = (
                f"Provider treatment usage pattern differs from "
                f"training diagnosis-level treatment distribution. "
                f"Provider used this treatment {provider_treat_count} times "
                f"out of {provider_total} diagnosis claims "
                f"({provider_usage}%). "
                f"Across all providers, this treatment appeared "
                f"{global_treat_count} times out of {global_total} diagnosis claims "
                f"({global_usage}%). "
                f"Usage ratio is {usage_ratio}x."
            )

        elif support_status != "VALID":

            explanation = (
                f"Upcoding analysis is not conclusive due to {support_status}. "
                f"Global diagnosis claims: {global_total}, "
                f"global diagnosis-treatment claims: {global_treat_count}, "
                f"provider diagnosis claims: {provider_total}, "
                f"provider diagnosis-treatment claims: {provider_treat_count}."
            )

        else:

            explanation = (
                "No strong provider treatment usage deviation detected "
                "from training-history diagnosis-level treatment distribution."
            )

        return {
            "UPCODING_ANOMALY": upcoding_flag,
            "UPCODING_CONFIDENCE": confidence,
            "UPCODING_SUPPORT_STATUS": support_status,

            "GLOBAL_DIAG_TOTAL": global_total,
            "GLOBAL_DIAG_TREAT_COUNT": global_treat_count,
            "GLOBAL_TREAT_USAGE_PCT": global_usage,

            "PROVIDER_DIAG_TOTAL": provider_total,
            "PROVIDER_DIAG_TREAT_COUNT": provider_treat_count,
            "PROVIDER_TREAT_USAGE_PCT": provider_usage,

            "USAGE_RATIO": usage_ratio,

            "DIAG_P75_COST": info.get("DIAG_P75_COST", 0),
            "DIAG_P90_COST": info.get("DIAG_P90_COST", 0),

            "TREATMENT_P75_COST": info.get("TREATMENT_P75_COST", 0),
            "TREATMENT_P90_COST": info.get("TREATMENT_P90_COST", 0),

            "UPCODING_EXPLANATION": explanation,
        }