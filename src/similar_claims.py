import joblib
from pathlib import Path


class SimilarClaimsIntelligence:

    def __init__(self, config):

        self.config = config

        models_dir = Path(config["paths"]["models_dir"]).parent

        artifact_path = (
            models_dir
            / "intelligence"
            / "similar_claims_artifact.pkl"
        )

        self.similar_artifact = joblib.load(artifact_path)

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

    def _empty_result(self, explanation):

        return {
            "SIMILAR_CLAIMS_ANOMALY": 0,
            "SIMILAR_CLAIMS_STATUS": "NO_MATCH",
            "SIMILAR_CLAIMS_LEVEL_USED": "NO_MATCH",
            "SIMILAR_CLAIMS_COUNT": 0,
            "SIMILAR_REJECTED_COUNT": 0,
            "SIMILAR_REJECTION_RATE": 0,
            "SIMILAR_MEDIAN_COST": 0,
            "SIMILAR_P90_COST": 0,
            "SIMILAR_MAX_COST": 0,
            "SIMILAR_COST_ABOVE_P90": 0,
            "SIMILAR_TOP_REJ_CODE": "",
            "SIMILAR_TOP_REJ_DESC": "",
            "SIMILAR_CLAIMS_EXPLANATION": explanation,
            "SIMILAR_TOP_CONTEXTS": [],
        }

    def _build_keys(self, row):

        diagnosis = self._clean_text(
            row.get("PA_PRIMARY_DIAG", "")
        )

        treatment = self._clean_code(
            row.get(
                "PROV_TREAT_CODE_CLEAN",
                row.get("PROV_TREAT_CODE", "")
            )
        )

        provider = self._clean_text(
            row.get("PROV_NAME", "")
        )

        doctor = self._clean_text(
            row.get("DOC_NAME", "")
        )

        cpt_code = self._clean_code(
            row.get("CPT_CODE", "")
        )

        ins_code = self._clean_code(
            row.get("INS_TREAT_CODE", "")
        )

        keys = []

        if diagnosis and treatment and provider and doctor:
            keys.append((
                "DIAG_TREAT_PROVIDER_DOCTOR",
                (
                    diagnosis,
                    treatment,
                    provider,
                    doctor,
                )
            ))

        if diagnosis and treatment and provider:
            keys.append((
                "DIAG_TREAT_PROVIDER",
                (
                    diagnosis,
                    treatment,
                    provider,
                )
            ))

        if diagnosis and treatment:
            keys.append((
                "DIAG_TREAT",
                (
                    diagnosis,
                    treatment,
                )
            ))

        if diagnosis and cpt_code and ins_code and provider:
            keys.append((
                "DIAG_CPT_INS_PROVIDER",
                (
                    diagnosis,
                    cpt_code,
                    ins_code,
                    provider,
                )
            ))

        return keys

    def analyze(self, row):

        input_cost = float(
            row.get("PA_EST_AMT_LC", 0) or 0
        )

        keys = self._build_keys(row)

        best = None

        top_contexts = []

        for context_name, key_values in keys:

            lookup_key = (
                context_name,
                key_values,
            )

            info = self.similar_artifact.get(lookup_key)

            if not info:
                continue

            top_contexts.append(info)

            if best is None:
                best = info
                continue

            if (
                info.get("SIMILAR_CLAIMS_COUNT", 0)
                > best.get("SIMILAR_CLAIMS_COUNT", 0)
            ):
                best = info

        if best is None:
            return self._empty_result(
                "No similar claim context found in training history."
            )

        similar_count = int(
            best.get("SIMILAR_CLAIMS_COUNT", 0)
        )

        rejected_count = int(
            best.get("SIMILAR_REJECTED_COUNT", 0)
        )

        rejection_rate = float(
            best.get("SIMILAR_REJECTION_RATE", 0)
        )

        median_cost = float(
            best.get("SIMILAR_MEDIAN_COST", 0)
        )

        p90_cost = float(
            best.get("SIMILAR_P90_COST", 0)
        )

        max_cost = float(
            best.get("SIMILAR_MAX_COST", 0)
        )

        top_rej_code = best.get(
            "SIMILAR_TOP_REJ_CODE",
            ""
        )

        top_rej_desc = best.get(
            "SIMILAR_TOP_REJ_DESC",
            ""
        )

        cost_above_p90 = int(
            p90_cost > 0
            and input_cost > p90_cost
        )

        anomaly = 0
        status = "NORMAL_SIMILAR_HISTORY"

        if similar_count >= 5 and rejection_rate >= 50:
            anomaly = 1
            status = "HIGH_REJECTION_HISTORY"

        elif similar_count >= 5 and rejection_rate >= 25:
            anomaly = 1
            status = "WARNING_REJECTION_HISTORY"

        elif similar_count >= 5 and cost_above_p90 == 1:
            anomaly = 1
            status = "COST_ABOVE_SIMILAR_CLAIMS"

        explanation = (
            f"Best similar context is {best.get('CONTEXT_NAME', '')}. "
            f"Training history contains {similar_count} similar claims. "
            f"{rejected_count} claims had rejection or partial approval history "
            f"({round(rejection_rate, 2)}%). "
            f"Median cost is {median_cost}, P90 cost is {p90_cost}, "
            f"maximum observed cost is {max_cost}."
        )

        if top_rej_code:
            explanation += f" Top rejection code is {top_rej_code}."

        if top_rej_desc:
            explanation += f" Top rejection reason is {top_rej_desc}."

        if cost_above_p90 == 1:
            explanation += (
                f" Current submitted cost {input_cost} is above similar claims P90 {p90_cost}."
            )

        return {
            "SIMILAR_CLAIMS_ANOMALY": anomaly,
            "SIMILAR_CLAIMS_STATUS": status,
            "SIMILAR_CLAIMS_LEVEL_USED": best.get("CONTEXT_NAME", ""),
            "SIMILAR_CLAIMS_COUNT": similar_count,
            "SIMILAR_REJECTED_COUNT": rejected_count,
            "SIMILAR_REJECTION_RATE": rejection_rate,
            "SIMILAR_MEDIAN_COST": median_cost,
            "SIMILAR_P90_COST": p90_cost,
            "SIMILAR_MAX_COST": max_cost,
            "SIMILAR_COST_ABOVE_P90": cost_above_p90,
            "SIMILAR_TOP_REJ_CODE": top_rej_code,
            "SIMILAR_TOP_REJ_DESC": top_rej_desc,
            "SIMILAR_TOP_CONTEXTS": top_contexts[:5],
            "SIMILAR_CLAIMS_EXPLANATION": explanation,
        }