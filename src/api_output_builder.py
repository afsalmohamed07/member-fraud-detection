import pandas as pd


class APIOutputBuilder:
    def __init__(self, review_threshold=60):
        self.review_threshold = review_threshold

    def _num(self, value, default=0):
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    def _flag(self, result, key):
        return int(self._num(result.get(key), 0)) == 1

    def _add(self, summary, key, explanation):
        if explanation:
            summary[key] = str(explanation)

    def build(self, result: dict):
        risk_score = 0
        summary = {}

        actual = self._num(result.get("PA_EST_AMT_LC"))

        provider = result.get("PROV_NAME", "")
        diagnosis = result.get("PA_PRIMARY_DIAG", "")
        treatment = (
            result.get("PROV_TREAT_CODE_CLEAN")
            or result.get("PROV_TREAT_CODE")
            or ""
        )

        provider_support = self._num(result.get("PROVIDER_SUPPORT"))
        provider_median = self._num(result.get("PROVIDER_MEDIAN"))
        provider_p90 = self._num(result.get("PROVIDER_P90"))
        provider_mode = self._num(result.get("PROVIDER_MODE_COST"))
        provider_mode_count = self._num(result.get("PROVIDER_MODE_COUNT"))

        # =====================================================
        # COST INTELLIGENCE
        # =====================================================
        expected_cost = self._num(
            result.get("CATBOOST_EXPECTED_COST")
            or result.get("P50_COST")
            or result.get("p50_cost")
            or result.get("ML_EXPECTED_COST")
        )

        lower_limit = self._num(result.get("ML_NORMAL_LOW"))

        upper_limit = self._num(
            result.get("CATBOOST_NORMAL_UPPER")
            or result.get("P75_COST")
            or result.get("p75_cost")
            or result.get("ML_NORMAL_HIGH")
        )

        high_risk_limit = self._num(
            result.get("CATBOOST_HIGH_RISK_LIMIT")
            or result.get("P90_COST")
            or result.get("p90_cost")
            or result.get("ML_HIGH_RISK_LIMIT")
        )

        cost_msg = ""

        if expected_cost > 0:
            cost_msg += (
                f"CatBoost predicted expected treatment cost as "
                f"{round(expected_cost, 2)}. "
            )

            if lower_limit > 0 and upper_limit > 0:
                cost_msg += (
                    f"Model-predicted normal cost range is approximately "
                    f"{round(lower_limit, 2)} to {round(upper_limit, 2)}. "
                )
            elif upper_limit > 0:
                cost_msg += (
                    f"Model-predicted normal upper cost limit is approximately "
                    f"{round(upper_limit, 2)}. "
                )

            if high_risk_limit > 0:
                cost_msg += (
                    f"Model high-risk cost limit is approximately "
                    f"{round(high_risk_limit, 2)}. "
                )

        cost_msg += f"Current submitted cost is {actual}. "

        if high_risk_limit > 0 and actual > high_risk_limit:
            risk_score += 35
            cost_msg += (
                f"Current cost is above the model high-risk cost limit by "
                f"{round(actual / high_risk_limit, 2)}x."
            )

        elif upper_limit > 0 and actual > upper_limit:
            risk_score += 25
            cost_msg += (
                f"Current cost is above the model-predicted normal cost range by "
                f"{round(actual / upper_limit, 2)}x."
            )

        elif lower_limit > 0 and actual < lower_limit:
            risk_score += 3
            cost_msg += (
                "Current cost is below the model-predicted normal cost range. "
                "Low cost alone is treated as weak evidence."
            )

        elif expected_cost > 0 and actual < expected_cost * 0.5:
            risk_score += 3
            cost_msg += (
                "Current cost is below the model-predicted expected cost. "
                "Low cost alone is treated as weak evidence."
            )

        elif expected_cost > 0:
            cost_msg += "Current cost is within the model-predicted cost expectation."

        self._add(summary, "COST_INTELLIGENCE", cost_msg.strip())

        # =====================================================
        # PROVIDER PEER PRICING
        # =====================================================
        peer_msg = result.get("PROVIDER_PEER_PRICING")

        if peer_msg:
            self._add(summary, "PROVIDER_PEER_PRICING", peer_msg)

        # =====================================================
        # CLAIM PATTERN AUTOENCODER
        # Show only when anomaly flag is 1
        # =====================================================
        claim_pattern_score = self._num(
            result.get("CLAIM_PATTERN_ANOMALY_SCORE")
        )

        claim_pattern_flag = self._num(
            result.get("CLAIM_PATTERN_ANOMALY_FLAG")
        )

        self._add(
            summary,
            "CLAIM_PATTERN_MODEL",
            (
                f"Claim pattern autoencoder flag is {claim_pattern_flag}. "
                f"Pattern score is {claim_pattern_score}/100."
            )
        )

        if claim_pattern_flag == 1:
            risk_score += 12

        # =====================================================
        # INFO / NEW PATTERN RULES
        # =====================================================
        if self._flag(result, "UNKNOWN_DIAGNOSIS_ANOMALY"):
            risk_score += 2
            self._add(
                summary,
                "UNKNOWN_DIAGNOSIS_ANOMALY",
                "Diagnosis is not available in training history. Treated as a new-pattern signal, not strong fraud evidence by itself."
            )

        if self._flag(result, "UNSEEN_TREATMENT_CODE_ANOMALY"):
            risk_score += 3
            exp = result.get("TREATMENT_EXPECTATION_EXPLANATION")
            self._add(
                summary,
                "UNSEEN_TREATMENT_CODE_ANOMALY",
                exp if exp else (
                    f"Treatment {treatment} was not directly seen for diagnosis {diagnosis} in training history. "
                    "This is treated as new-pattern information unless supported by cost, provider, drug, or clinical mismatch signals."
                )
            )

        if self._flag(result, "DRUG_NOT_IN_MASTER_ANOMALY"):
            risk_score += 3
            self._add(
                summary,
                "DRUG_NOT_IN_MASTER_ANOMALY",
                "Drug is not available in the trained drug master/history artifact. This may indicate a new drug or master-data gap."
            )

        if self._flag(result, "UNSEEN_DRUG_ANOMALY"):
            risk_score += 3
            self._add(
                summary,
                "UNSEEN_DRUG_ANOMALY",
                "Drug was not seen in training history. Treated as new-pattern information, not strong anomaly by itself."
            )

        # =====================================================
        # PROVIDER FINGERPRINT
        # =====================================================
        if result.get("PROVIDER_FINGERPRINT_ANOMALY") == 1:
            risk_score += 12
            self._add(
                summary,
                "PROVIDER_FINGERPRINT_ANOMALY",
                result.get("PROVIDER_FINGERPRINT_EXPLANATION")
            )

        # =====================================================
        # TREATMENT EXPECTATION
        # =====================================================
        if self._flag(result, "UNEXPECTED_TREATMENT_FOR_DIAGNOSIS"):
            risk_score += 15
            exp = result.get("TREATMENT_EXPECTATION_EXPLANATION")
            self._add(
                summary,
                "UNEXPECTED_TREATMENT_FOR_DIAGNOSIS",
                exp if exp else (
                    f"Treatment {treatment} appears unusual for diagnosis {diagnosis} based on learned treatment distribution."
                )
            )

        # =====================================================
        # RARE TREATMENT / DRUG
        # =====================================================
        if self._flag(result, "RARE_TREATMENT_ANOMALY"):
            total_diag = self._num(result.get("TOTAL_DIAG_ROWS"))
            treatment_count = self._num(result.get("TREATMENT_USAGE_COUNT"))

            if total_diag >= 20 and treatment_count > 0:
                risk_score += 8
                msg = (
                    f"Treatment {treatment} is rare for diagnosis {diagnosis}. "
                    f"Diagnosis support is {int(total_diag)}, treatment usage count is {int(treatment_count)}."
                )
            else:
                risk_score += 2
                msg = (
                    f"Rare treatment signal exists, but support is weak. "
                    f"Diagnosis support is {int(total_diag)}, treatment usage count is {int(treatment_count)}."
                )

            self._add(summary, "RARE_TREATMENT_ANOMALY", msg)

        if self._flag(result, "DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY"):
            diag_pharmacy_total = self._num(result.get("TOTAL_DIAG_PHARMACY_ROWS"))

            if diag_pharmacy_total >= 20:
                risk_score += 12
                msg = (
                    f"Submitted drug is historically not associated with diagnosis {diagnosis}. "
                    f"Diagnosis pharmacy support is {int(diag_pharmacy_total)}."
                )
            else:
                risk_score += 3
                msg = (
                    f"Drug-diagnosis mismatch signal exists, but support is weak. "
                    f"Diagnosis pharmacy support is {int(diag_pharmacy_total)}."
                )

            self._add(summary, "DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY", msg)

        if self._flag(result, "RARE_DRUG_ANOMALY"):
            drug_count = self._num(result.get("DRUG_USAGE_COUNT"))
            drug_pct = self._num(result.get("DRUG_USAGE_%"))
            diag_pharmacy_total = self._num(result.get("TOTAL_DIAG_PHARMACY_ROWS"))

            if diag_pharmacy_total >= 20:
                risk_score += 6
                msg = (
                    f"Drug is rare for diagnosis {diagnosis}. "
                    f"Drug usage count is {int(drug_count)}, usage percentage is {round(drug_pct, 2)}%, "
                    f"diagnosis pharmacy support is {int(diag_pharmacy_total)}."
                )
            else:
                risk_score += 2
                msg = (
                    f"Rare drug signal exists, but support is weak. "
                    f"Diagnosis pharmacy support is {int(diag_pharmacy_total)}."
                )

            self._add(summary, "RARE_DRUG_ANOMALY", msg)

        # =====================================================
        # PHARMACY STRUCTURE
        # =====================================================
        if self._flag(result, "PHARMACY_MISSING_DRUG_ANOMALY"):
            risk_score += 8
            self._add(
                summary,
                "PHARMACY_MISSING_DRUG_ANOMALY",
                "Pharmacy claim does not contain valid drug information. This is a data completeness issue requiring review."
            )

        if self._flag(result, "NON_PHARMACY_WITH_DRUG_ANOMALY"):
            risk_score += 8
            self._add(
                summary,
                "NON_PHARMACY_WITH_DRUG_ANOMALY",
                "Drug information is present in a non-pharmacy claim category. This may indicate wrong category or coding issue."
            )

        # =====================================================
        # DURATION / DOSAGE
        # =====================================================
        if self._flag(result, "DURATION_OUTSIDE_RANGE_ANOMALY"):
            risk_score += 12
            duration = result.get("DURATION_DAYS")
            min_duration = result.get("MIN_DURATION")
            max_duration = result.get("MAX_DURATION")
            self._add(
                summary,
                "DURATION_OUTSIDE_RANGE_ANOMALY",
                f"Drug duration {duration} is outside expected range. Expected range is {min_duration} to {max_duration}."
            )

        if self._flag(result, "EXTREME_DURATION_ANOMALY"):
            risk_score += 15
            duration = result.get("DURATION_DAYS")
            self._add(
                summary,
                "EXTREME_DURATION_ANOMALY",
                f"Drug duration {duration} is extremely high and requires review."
            )

        if self._flag(result, "DOSAGE_OUTSIDE_RANGE_ANOMALY"):
            risk_score += 12
            dosage = result.get("DOSAGE_PER_DAY")
            min_dosage = result.get("MIN_DOSAGE")
            max_dosage = result.get("MAX_DOSAGE")
            self._add(
                summary,
                "DOSAGE_OUTSIDE_RANGE_ANOMALY",
                f"Drug dosage {dosage} is outside expected range. Expected range is {min_dosage} to {max_dosage}."
            )

        # =====================================================
        # LICENSE
        # =====================================================
        if self._flag(result, "LICENSE_ANOMALY"):
            risk_score += 20
            exp = result.get("LICENSE_EXPLANATION")
            self._add(
                summary,
                "LICENSE_ANOMALY",
                exp if exp else "Doctor license is missing, mismatched, unseen, or inconsistent for this doctor."
            )

        # =====================================================
        # DOCTOR BEHAVIOR
        # =====================================================
        if self._flag(result, "DOCTOR_FACILITY_ANOMALY"):
            support = self._num(result.get("DOCTOR_TOTAL_FACILITY_ROWS"))
            exp = result.get("DOCTOR_FACILITY_EXPLANATION")
            risk_score += 10 if support >= 5 else 3
            self._add(
                summary,
                "DOCTOR_FACILITY_ANOMALY",
                exp if exp else (
                    f"Doctor facility pattern is unusual. Doctor facility support is {int(support)}."
                )
            )

        if self._flag(result, "DOCTOR_TREATMENT_ANOMALY"):
            support = self._num(result.get("DOCTOR_TOTAL_TREATMENT_ROWS"))
            treatment_count = self._num(result.get("DOCTOR_TREATMENT_COUNT"))
            exp = result.get("DOCTOR_TREATMENT_EXPLANATION")
            risk_score += 10 if support >= 5 else 3
            self._add(
                summary,
                "DOCTOR_TREATMENT_ANOMALY",
                exp if exp else (
                    f"Doctor treatment pattern is unusual. Doctor total treatment support is {int(support)}, "
                    f"this treatment count is {int(treatment_count)}."
                )
            )

        # =====================================================
        # DOCTOR NEW DRUG PATTERN
        # Pharmacy/drug-present case only; not direct anomaly
        # =====================================================
        is_drug_present = (
            str(result.get("PAT_DRUG_NAME", "")).strip().upper()
            not in ["", "NAN", "NONE", "NULL", "UNKNOWN"]
        )

        is_pharmacy_case = (
            str(result.get("PA_CATG", "")).strip().upper() == "PHARMACY"
            or str(result.get("PA_CATG_NORM", "")).strip().upper() == "PHARMACY"
            or is_drug_present
        )

        if is_pharmacy_case and self._flag(result, "DOCTOR_DRUG_ANOMALY"):
            support = self._num(result.get("DOCTOR_TOTAL_DRUG_ROWS"))
            drug_count = self._num(result.get("DOCTOR_DRUG_COUNT"))
            exp = result.get("DOCTOR_DRUG_EXPLANATION")

            risk_score += 2

            self._add(
                summary,
                "DOCTOR_NEW_DRUG_PATTERN",
                exp if exp else (
                    f"This drug appears to be a new prescribing pattern for this doctor. "
                    f"Doctor historical drug support is {int(support)} and "
                    f"historical usage count for this drug is {int(drug_count)}. "
                    f"This is treated as informational evidence rather than a direct anomaly."
                )
            )

        # =====================================================
        # PROVIDER BEHAVIOR
        # =====================================================
        if self._flag(result, "PROVIDER_BEHAVIOR_ANOMALY"):
            exp = result.get("PROVIDER_EXPLANATION")

            if provider_support >= 5:
                risk_score += 12
                msg = exp if exp else (
                    f"For provider {provider}, diagnosis {diagnosis}, and treatment {treatment}, "
                    f"training history contains {int(provider_support)} matching claims. "
                    f"Provider median cost is {provider_median}, provider P90 cost is {provider_p90}. "
                    f"Current submitted amount is {actual}."
                )
            else:
                risk_score += 3
                msg = (
                    f"Provider behavior signal exists, but provider support is weak "
                    f"({int(provider_support)} matching records)."
                )

            self._add(summary, "PROVIDER_BEHAVIOR_ANOMALY", msg)

        if self._flag(result, "PHARMACY_HIGH_COST_ANOMALY"):
            risk_score += 10
            self._add(
                summary,
                "PHARMACY_HIGH_COST_ANOMALY",
                f"Pharmacy claim amount {actual} is above configured high-cost pharmacy threshold."
            )

        # =====================================================
        # UPCODING
        # =====================================================
        if self._flag(result, "UPCODING_ANOMALY"):
            exp = result.get("UPCODING_EXPLANATION")
            provider_total = self._num(result.get("PROVIDER_DIAG_TOTAL"))
            global_total = self._num(result.get("GLOBAL_DIAG_TOTAL"))

            risk_score += 12 if provider_total >= 5 and global_total >= 20 else 3

            self._add(
                summary,
                "UPCODING_ANOMALY",
                exp if exp else (
                    f"For diagnosis {diagnosis}, provider {provider} uses treatment {treatment} "
                    f"more frequently than the training-history diagnosis-level distribution."
                )
            )

        # =====================================================
        # SIMILAR CLAIMS
        # =====================================================
        if self._flag(result, "SIMILAR_CLAIMS_ANOMALY"):
            exp = result.get("SIMILAR_CLAIMS_EXPLANATION")
            count = self._num(result.get("SIMILAR_CLAIMS_COUNT"))

            risk_score += 12 if count >= 5 else 3

            self._add(
                summary,
                "SIMILAR_CLAIMS_ANOMALY",
                exp if exp else (
                    f"Similar claim context count is {int(count)}. "
                    "Similar training-history claims show rejection, partial approval, or abnormal cost behavior."
                )
            )

        # =====================================================
        # ISOLATION CONTEXT ONLY
        # =====================================================
        if self._flag(result, "ISOLATION_CONTEXT_ANOMALY"):
            exp = result.get("ISOLATION_CONTEXT_EXPLANATION")
            if exp:
                risk_score += 8
                self._add(summary, "ISOLATION_CONTEXT_ANOMALY", str(exp))

        # =====================================================
        # CATBOOST / PROVIDER CONFLICT
        # =====================================================
        p90 = self._num(result.get("P90_COST") or result.get("p90_cost"))

        if (
            provider_mode > 0
            and p90 > 0
            and provider_mode > p90
            and provider_mode_count >= 3
        ):
            risk_score += 5
            self._add(
                summary,
                "CATBOOST_PROVIDER_HISTORY_CONFLICT",
                f"CatBoost P90 is {p90}, but provider repeated historical charge is {provider_mode}. "
                f"Provider repeated charge count is {int(provider_mode_count)}, so provider evidence is treated as stronger."
            )

        # =====================================================
        # LIGHTGBM SUPPORTING SIGNAL
        # =====================================================
        final_label = str(result.get("FINAL_LABEL", "")).upper()
        final_conf = self._num(result.get("FINAL_CONFIDENCE"))

        if final_label in ["HIGH", "CRITICAL"]:
            risk_score += 15
            self._add(
                summary,
                "LIGHTGBM_HIGH_RISK",
                f"Risk classifier marked this claim as {final_label} with {final_conf}% confidence."
            )

        elif final_label == "MEDIUM":
            risk_score += 8
            self._add(
                summary,
                "LIGHTGBM_MEDIUM_RISK",
                f"Risk classifier marked this claim as MEDIUM with {final_conf}% confidence."
            )

        risk_score = int(min(max(risk_score, 0), 100))
        ml_model_status = 1 if risk_score >= self.review_threshold else 0

        return [
            {
                "ml_model_status": ml_model_status,
                "risk_score": risk_score,
            },
            {
                "summary": [
                    summary
                ]
            }
        ]