import joblib
import re
import pandas as pd
from pathlib import Path


class ProviderFingerprintAnalyzer:
    def __init__(self, config):
        artifact_path = (
            Path(config["paths"]["models_dir"]).parent
            / "intelligence"
            / "provider_fingerprint_artifact.pkl"
        )

        self.artifact = joblib.load(artifact_path)

    def _clean_text(self, value):
        if pd.isna(value):
            return ""

        value = str(value).upper().strip()

        if value in ["", "NAN", "NONE", "NULL", "UNKNOWN"]:
            return ""

        return value

    def _get_treatment_family(self, code):
        code = self._clean_text(code)
        code = re.sub(r"[^A-Z0-9]", "", code)

        if code.startswith("PHARMACY"):
            return "PHARMACY"

        match = re.match(r"^[A-Z]+", code)

        if match:
            return match.group(0)

        return "UNKNOWN"

    def analyze(self, row):
        provider = self._clean_text(row.get("PROV_NAME", ""))

        if provider not in self.artifact:
            return {
                "PROVIDER_FINGERPRINT_ANOMALY": 0,
                "PROVIDER_FINGERPRINT_EXPLANATION": (
                    "Provider fingerprint not available in training history."
                ),
            }

        fp = self.artifact[provider]

        current_category = self._clean_text(
            row.get("PA_CATG_NORM")
            or row.get("PA_CATG")
        )

        current_family = self._clean_text(
            row.get("PROV_TREAT_FAMILY")
            or self._get_treatment_family(
                row.get("PROV_TREAT_CODE_CLEAN")
                or row.get("PROV_TREAT_CODE")
            )
        )

        current_cost = pd.to_numeric(
            row.get("PA_EST_AMT_LC", 0),
            errors="coerce"
        )

        if pd.isna(current_cost):
            current_cost = 0

        reasons = []

        if (
            fp["TOP_CATEGORY"] != "UNKNOWN"
            and current_category
            and current_category != fp["TOP_CATEGORY"]
        ):
            reasons.append(
                f"provider most commonly handled {fp['TOP_CATEGORY']} category claims "
                f"in training history, but current claim category is {current_category}"
            )

        if (
            fp["TOP_TREATMENT_FAMILY"] != "UNKNOWN"
            and current_family
            and current_family != fp["TOP_TREATMENT_FAMILY"]
        ):
            reasons.append(
                f"provider most commonly used {fp['TOP_TREATMENT_FAMILY']} treatment family "
                f"in training history, but current treatment family is {current_family}"
            )
            if fp["UPPER_COST"] > 0 and current_cost > fp["UPPER_COST"] * 2:
                reasons.append(
                f"current cost {current_cost} is much higher than provider usual upper cost {fp['UPPER_COST']}"
            )

        anomaly = 1 if len(reasons) > 0 and fp["TOTAL_CLAIMS"] >= 10 else 0

        if anomaly:
            explanation = (
                f"Provider fingerprint mismatch detected. "
                f"Provider has {fp['TOTAL_CLAIMS']} historical claim(s). "
                + " | ".join(reasons)
            )
        else:
            explanation = (
                f"Provider fingerprint check passed or evidence is weak. "
                f"Provider historical claims: {fp['TOTAL_CLAIMS']}."
            )

        return {
            "PROVIDER_FINGERPRINT_ANOMALY": anomaly,
            "PROVIDER_FINGERPRINT_EXPLANATION": explanation,
            "PROVIDER_FINGERPRINT_TOTAL_CLAIMS": fp["TOTAL_CLAIMS"],
        }