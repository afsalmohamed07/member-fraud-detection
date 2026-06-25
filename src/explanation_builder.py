import pandas as pd


class ExplanationBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, df: pd.DataFrame):
        df = df.copy()

        df["SHORT_EXPLANATION"] = df.apply(
            self._short_explanation,
            axis=1
        )

        df["DETAILED_EXPLANATION"] = df.apply(
            self._detailed_explanation,
            axis=1
        )

        return df

    # =====================================================
    # SHORT EXPLANATION
    # =====================================================
    def _short_explanation(self, row):
        reasons = []

        if row.get("UNKNOWN_DIAGNOSIS_ANOMALY", 0) == 1:
            reasons.append("Unknown diagnosis")

        if row.get("UNSEEN_TREATMENT_CODE_ANOMALY", 0) == 1:
            reasons.append("Unseen treatment")

        elif row.get("RARE_TREATMENT_ANOMALY", 0) == 1:
            reasons.append("Rare treatment")

        # =================================================
        # DRUG SHORT
        # =================================================
        if row.get("DRUG_NOT_IN_MASTER_ANOMALY", 0) == 1:
            reasons.append("Unknown drug")

        elif row.get("DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY", 0) == 1:
            reasons.append("Drug not used for diagnosis")

        elif row.get("RARE_DRUG_ANOMALY", 0) == 1:
            reasons.append("Rare drug")

        if row.get("DURATION_OUTSIDE_RANGE_ANOMALY", 0) == 1:
            reasons.append("Duration outside range")

        if row.get("EXTREME_DURATION_ANOMALY", 0) == 1:
            reasons.append("Extreme duration")

        if row.get("DOSAGE_OUTSIDE_RANGE_ANOMALY", 0) == 1:
            reasons.append("Dosage outside range")

        if row.get("NON_PHARMACY_WITH_DRUG_ANOMALY", 0) == 1:
            reasons.append("Drug under non-pharmacy")

        if row.get("PHARMACY_MISSING_DRUG_ANOMALY", 0) == 1:
            reasons.append("Drug missing in pharmacy")

        if row.get("LICENSE_ANOMALY", 0) == 1:
            reasons.append("License anomaly")

        if row.get("PROVIDER_BEHAVIOR_ANOMALY", 0) == 1:
            reasons.append("Provider behavior anomaly")

        if row.get("PHARMACY_HIGH_COST_ANOMALY", 0) == 1:
            reasons.append("High pharmacy cost")

        if row.get("COST_ANOMALY_FINAL", 0) == 1:
            reasons.append("Cost above historical range")

        if row.get("SIMILAR_CLAIMS_ANOMALY", 0) == 1:
            reasons.append("Similar claims rejection history")

        if row.get("UPCODING_ANOMALY", 0) == 1:
            reasons.append("Provider treatment pattern deviation")

        if row.get("ISOLATION_ANOMALY", 0) == 1:
            reasons.append("Isolation anomaly")

        clean_reasons = self._clean_list(reasons)

        if len(clean_reasons) == 0:
            return "No major anomaly detected"

        return " | ".join(clean_reasons[:3])

    # =====================================================
    # DETAILED EXPLANATION
    # =====================================================
    def _detailed_explanation(self, row):
        explanations = []

        # =================================================
        # DIAGNOSIS
        # =================================================
        if row.get("UNKNOWN_DIAGNOSIS_ANOMALY", 0) == 1:
            explanations.append(
                "Diagnosis combination not seen historically"
            )

        # =================================================
        # TREATMENT
        # =================================================
        if row.get("UNSEEN_TREATMENT_CODE_ANOMALY", 0) == 1:
            explanations.append(
                "Treatment code not seen historically for this diagnosis"
            )

        elif row.get("RARE_TREATMENT_ANOMALY", 0) == 1:
            usage = row.get("TREATMENT_USAGE_%", 0)

            explanations.append(
                f"Rare treatment for diagnosis "
                f"(usage {self._safe_round(usage)}%)"
            )

        # =================================================
        # DRUG
        # =================================================
        if row.get("DRUG_NOT_IN_MASTER_ANOMALY", 0) == 1:
            explanations.append(
                "Drug not found in historical drug master"
            )

        elif row.get("DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY", 0) == 1:
            explanations.append(
                "Drug exists historically, but not used for this diagnosis"
            )

        elif row.get("RARE_DRUG_ANOMALY", 0) == 1:
            usage = row.get("DRUG_USAGE_%", 0)

            explanations.append(
                f"Rare drug for diagnosis "
                f"(usage {self._safe_round(usage)}%)"
            )

        # =================================================
        # DURATION
        # =================================================
        if row.get("DURATION_OUTSIDE_RANGE_ANOMALY", 0) == 1:
            current = row.get("DURATION_DAYS")
            min_v = row.get("MIN_DURATION")
            max_v = row.get("MAX_DURATION")
            p90_v = row.get("P90_DURATION", None)

            if p90_v is not None and str(p90_v).lower() != "nan":
                explanations.append(
                    f"Input duration {self._safe_round(current)}D is above "
                    f"normal historical duration threshold "
                    f"(P90 {self._safe_round(p90_v)}D, "
                    f"historical min-max {self._safe_round(min_v)}D–{self._safe_round(max_v)}D)"
                )
            else:
                explanations.append(
                    f"Input duration {self._safe_round(current)}D is outside "
                    f"historical drug duration range "
                    f"{self._safe_round(min_v)}D–{self._safe_round(max_v)}D"
                )

        elif row.get("DURATION_UNSEEN_WITHIN_RANGE", 0) == 1:
            current = row.get("DURATION_DAYS")

            explanations.append(
                f"Duration {self._safe_round(current)}D not seen exactly "
                f"but within historical range"
            )

        elif row.get("DURATION_EXACT_SEEN", 0) == 1:
            current = row.get("DURATION_DAYS")

            explanations.append(
                f"Duration {self._safe_round(current)}D historically seen"
            )

        if row.get("EXTREME_DURATION_ANOMALY", 0) == 1:
            explanations.append(
                "Extremely large duration detected"
            )

        # =================================================
        # DOSAGE
        # =================================================
        if row.get("DOSAGE_OUTSIDE_RANGE_ANOMALY", 0) == 1:
            current = row.get("DOSAGE_PER_DAY")
            min_v = row.get("MIN_DOSAGE")
            max_v = row.get("MAX_DOSAGE")

            explanations.append(
                f"Dosage {self._safe_round(current)}/day outside "
                f"historical range {self._safe_round(min_v)}–{self._safe_round(max_v)}"
            )

        elif row.get("DOSAGE_UNSEEN_WITHIN_RANGE", 0) == 1:
            current = row.get("DOSAGE_PER_DAY")

            explanations.append(
                f"Dosage {self._safe_round(current)}/day not seen exactly "
                f"but within historical range"
            )

        elif row.get("DOSAGE_EXACT_SEEN", 0) == 1:
            current = row.get("DOSAGE_PER_DAY")

            explanations.append(
                f"Dosage {self._safe_round(current)}/day historically seen"
            )

        # =================================================
        # COST
        # =================================================
        if row.get("COST_ANOMALY_FINAL", 0) == 1:
            actual = row.get("PA_EST_AMT_LC")
            p75 = row.get("PATTERN_P75")
            p90 = row.get("PATTERN_P90")

            if (
                p90 is not None
                and str(p90).lower() != "nan"
                and actual > p90
            ):
                explanations.append(
                    f"Actual cost {self._safe_round(actual)} exceeds the historical P90 cost range "
                    f"({self._safe_round(p90)}) for the matched claim pattern"
                )

            elif (
                p75 is not None
                and str(p75).lower() != "nan"
                and actual > p75
            ):
                explanations.append(
                    f"Actual cost {self._safe_round(actual)} is above the historical P75 cost range "
                    f"({self._safe_round(p75)}) for the matched claim pattern"
                )

            else:
                explanations.append(
                    f"Actual cost {self._safe_round(actual)} deviates from the historical claim pattern"
                )
        # =================================================
        # PROVIDER
        # =================================================
        provider_exp = row.get(
            "PROVIDER_BEHAVIOR_EXPLANATION",
            ""
        )

        if provider_exp:
            explanations.append(provider_exp)
            
        # =================================================
        # SIMILAR CLAIMS
        # =================================================
        if row.get("SIMILAR_CLAIMS_ANOMALY", 0) == 1:

            count = row.get("SIMILAR_CLAIMS_COUNT", 0)
            rejected = row.get("SIMILAR_REJECTED_COUNT", 0)
            rej_rate = row.get("SIMILAR_REJECTION_RATE", 0)
            context = row.get("SIMILAR_CLAIMS_LEVEL_USED", "")

            explanations.append(
                f"{count} similar historical claims found using context [{context}]. "
                f"{rejected} claims were rejected or partially approved "
                f"({self._safe_round(rej_rate)}% rejection rate)" 
            )

        # =================================================
        # LICENSE
        # =================================================
        lic_exp = row.get(
            "LICENSE_EXPLANATION",
            ""
        )

        if lic_exp:
            explanations.append(lic_exp)

        # =================================================
        # STRUCTURE
        # =================================================
        if row.get("NON_PHARMACY_WITH_DRUG_ANOMALY", 0) == 1:
            explanations.append(
                "Drug present under non-pharmacy treatment"
            )

        if row.get("PHARMACY_MISSING_DRUG_ANOMALY", 0) == 1:
            explanations.append(
                "Pharmacy treatment without drug name"
            )

        # =================================================
        # PHARMACY COST
        # =================================================
        if row.get("PHARMACY_HIGH_COST_ANOMALY", 0) == 1:
            explanations.append(
                "Pharmacy cost extremely high"
            )
            
        # =================================================
        # UPCODING
        # =================================================
        if row.get("UPCODING_ANOMALY", 0) == 1:

            ratio = row.get("USAGE_RATIO", 0)
            provider_pct = row.get("PROVIDER_TREAT_USAGE_PCT", 0)
            global_pct = row.get("GLOBAL_TREAT_USAGE_PCT", 0)

            explanations.append(
                f"Provider treatment usage distribution differs from the historical diagnosis-level pattern."
                f"The provider uses this treatment {self._safe_round(ratio)}x more often "
                f"than the diagnosis-level historical average "
                f"(provider usage {self._safe_round(provider_pct)}% vs "
                f"global usage {self._safe_round(global_pct)}%)"
            )

        # =================================================
        # ISOLATION
        # =================================================
        if row.get("ISOLATION_ANOMALY", 0) == 1:
            explanations.append(
                "Overall claim pattern unusual compared to historical data"
            )

        clean_explanations = self._clean_list(explanations)

        if len(clean_explanations) == 0:
            return "No anomaly explanation generated"

        return " | ".join(clean_explanations)

    # =====================================================
    # HELPERS
    # =====================================================
    def _clean_list(self, items):
        clean_items = []

        for x in items:
            if x is None:
                continue

            text = str(x).strip()

            if text == "":
                continue

            if text.lower() == "nan":
                continue

            clean_items.append(text)

        return clean_items

    def _safe_round(self, value, digits=2):
        try:
            if pd.isna(value):
                return "NA"

            return round(float(value), digits)

        except Exception:
            return value