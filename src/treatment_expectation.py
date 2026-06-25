import re
import joblib
from pathlib import Path


class TreatmentExpectationIntelligence:
    def __init__(self, config):
        self.config = config

        models_dir = Path(config["paths"]["models_dir"]).parent
        artifact_path = models_dir / "intelligence" / "treatment_intelligence_artifact.pkl"

        self.artifact = joblib.load(artifact_path)

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
        return re.sub(r"[^A-Z0-9]", "", self._clean_text(value))

    def _family(self, code):
        code = self._clean_code(code)

        if code == "":
            return "UNKNOWN"

        match = re.match(r"^[A-Z]+", code)

        if match:
            return match.group(0)

        return "NUMERIC"

    def analyze(self, row):
        service = self._clean_text(
            row.get("SERVICE_TYPE_NORM", row.get("SERVICE_TYPE", ""))
        )

        category = self._clean_text(
            row.get("PA_CATG_NORM", row.get("PA_CATG", ""))
        )

        diagnosis = self._clean_text(
            row.get("PA_PRIMARY_DIAG", "")
        )

        treatment = self._clean_code(
            row.get("PROV_TREAT_CODE_CLEAN", row.get("PROV_TREAT_CODE", ""))
        )

        family = self._family(treatment)

        key = (
            service,
            category,
            diagnosis,
        )

        info = self.artifact.get(key)

        if not info:
            return {
                "UNEXPECTED_TREATMENT_FOR_DIAGNOSIS": 0,
                "TREATMENT_EXPECTATION_STATUS": "NO_DIAGNOSIS_CONTEXT",
                "TREATMENT_CONTEXT_SUPPORT": 0,
                "TREATMENT_CODE_PROBABILITY": 0,
                "TREATMENT_FAMILY_PROBABILITY": 0,
                "TREATMENT_FAMILY": family,
                "TREATMENT_EXPECTATION_EXPLANATION": (
                    "No learned treatment distribution found for this service/category/diagnosis context."
                ),
                "EXPECTED_TREATMENT_FAMILIES": [],
                "EXPECTED_TREATMENT_CODES": [],
            }

        total = int(info.get("TOTAL_CONTEXT_CLAIMS", 0))

        code_probs = info.get("CODE_PROBS", {})
        family_probs = info.get("FAMILY_PROBS", {})

        code_probability = float(code_probs.get(treatment, 0))
        family_probability = float(family_probs.get(family, 0))

        top_families = info.get("TOP_TREATMENT_FAMILIES", [])
        top_codes = info.get("TOP_TREATMENT_CODES", [])

        unexpected = 0
        status = "EXPECTED"

        if total < 30:
            status = "LOW_SUPPORT_CONTEXT"

        elif code_probability > 0:
            status = "EXACT_TREATMENT_SEEN"

        elif code_probability == 0 and family_probability >= 0.10:
            status = "NEW_CODE_BUT_EXPECTED_FAMILY"

        elif code_probability == 0 and family_probability > 0 and family_probability < 0.10:
            status = "LOW_PROBABILITY_TREATMENT_FAMILY"
            unexpected = 1

        elif family_probability == 0:
            status = "UNEXPECTED_TREATMENT_FAMILY"
            unexpected = 1

        explanation = (
            f"Treatment {treatment} was evaluated under service {service}, category {category}, "
            f"diagnosis {diagnosis}. Training support for this diagnosis context is {total}. "
            f"Exact treatment probability is {round(code_probability * 100, 2)}%. "
            f"Treatment family {family} probability is {round(family_probability * 100, 2)}%."
        )

        if status == "NEW_CODE_BUT_EXPECTED_FAMILY":
            explanation += (
                " The exact treatment code was not seen, but its treatment family is common for this diagnosis. "
                "This should be treated as a new-pattern information signal, not a strong anomaly by itself."
            )

        elif status in ["LOW_PROBABILITY_TREATMENT_FAMILY", "UNEXPECTED_TREATMENT_FAMILY"]:
            explanation += (
                " The submitted treatment family is unusual for this diagnosis context and should be reviewed "
                "together with cost, provider behavior, and clinical context."
            )

        elif status == "LOW_SUPPORT_CONTEXT":
            explanation += (
                " The diagnosis context has low support, so treatment expectation evidence should be treated carefully."
            )

        return {
            "UNEXPECTED_TREATMENT_FOR_DIAGNOSIS": unexpected,
            "TREATMENT_EXPECTATION_STATUS": status,
            "TREATMENT_CONTEXT_SUPPORT": total,
            "TREATMENT_CODE_PROBABILITY": round(code_probability * 100, 2),
            "TREATMENT_FAMILY_PROBABILITY": round(family_probability * 100, 2),
            "TREATMENT_FAMILY": family,
            "TREATMENT_EXPECTATION_EXPLANATION": explanation,
            "EXPECTED_TREATMENT_FAMILIES": top_families,
            "EXPECTED_TREATMENT_CODES": top_codes,
        }