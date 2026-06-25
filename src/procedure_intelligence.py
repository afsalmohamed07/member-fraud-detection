import pandas as pd
import numpy as np


class ProviderLevelIntelligence:
    def __init__(self, history_df: pd.DataFrame):
        self.df = history_df.copy()
        self.date_col = "SERVICE_DT"
        self.window_days = 30

    def _clean_text(self, value):
        if pd.isna(value):
            return ""

        value = str(value).upper().strip()
        value = " ".join(value.split())

        if value in ["", "NAN", "NONE", "NULL", "UNKNOWN"]:
            return ""

        if value.endswith(".0"):
            value = value[:-2]

        return value

    def _clean_code(self, value):
        return self._clean_text(value).replace(" ", "")

    def _parse_date_value(self, value):
        if pd.isna(value):
            return pd.NaT

        if isinstance(value, pd.Timestamp):
            return value

        value = str(value).upper().strip()
        value = " ".join(value.split())

        parsed = pd.to_datetime(
            value,
            errors="coerce",
            dayfirst=True
        )

        return parsed

    def _prepare_df(self, df):
        df = df.copy()

        required_cols = [
            "PROV_NAME",
            "DOC_NAME",
            "PA_PRIMARY_DIAG",
            "PROV_TREAT_CODE",
            "PROV_TREAT_CODE_CLEAN",
            "INS_TREAT_CODE",
            "CPT_CODE",
            "PAT_DRUG_NAME",
            "PA_EST_AMT_LC",
            "PA_APPR_AMT_LC",
            "PA_REJ_AMT_LC",
            "REJ_CODE",
            "REJ_DESC",
            "PBM_REJ_CODE",
            "PBM_REJ_DESC",
            self.date_col,
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        text_cols = [
            "PROV_NAME",
            "DOC_NAME",
            "PA_PRIMARY_DIAG",
            "PAT_DRUG_NAME",
            "REJ_CODE",
            "REJ_DESC",
            "PBM_REJ_CODE",
            "PBM_REJ_DESC",
        ]

        code_cols = [
            "PROV_TREAT_CODE",
            "PROV_TREAT_CODE_CLEAN",
            "INS_TREAT_CODE",
            "CPT_CODE",
        ]

        for col in text_cols:
            df[col] = df[col].apply(self._clean_text)

        for col in code_cols:
            df[col] = df[col].apply(self._clean_code)

        df[self.date_col] = df[self.date_col].apply(self._parse_date_value)

        df["PA_EST_AMT_LC"] = pd.to_numeric(
            df["PA_EST_AMT_LC"],
            errors="coerce"
        ).fillna(0)

        df["PA_APPR_AMT_LC"] = pd.to_numeric(
            df["PA_APPR_AMT_LC"],
            errors="coerce"
        ).fillna(0)

        df["PA_REJ_AMT_LC"] = pd.to_numeric(
            df["PA_REJ_AMT_LC"],
            errors="coerce"
        ).fillna(0)

        df["TREATMENT_KEY"] = np.where(
            df["PROV_TREAT_CODE_CLEAN"].astype(str).str.strip() != "",
            df["PROV_TREAT_CODE_CLEAN"],
            df["PROV_TREAT_CODE"],
        )

        df["FULL_REJECT_FLAG"] = (
            (df["PA_APPR_AMT_LC"] <= 0)
            & (df["PA_REJ_AMT_LC"] > 0)
        ).astype(int)

        df["PARTIAL_APPROVAL_FLAG"] = (
            (df["PA_APPR_AMT_LC"] > 0)
            & (df["PA_APPR_AMT_LC"] < df["PA_EST_AMT_LC"])
        ).astype(int)

        df["REVIEW_FLAG"] = (
            (df["FULL_REJECT_FLAG"] == 1)
            | (df["PARTIAL_APPROVAL_FLAG"] == 1)
            | (df["PA_REJ_AMT_LC"] > 0)
        ).astype(int)

        return df

    def _recent_window(self, df, input_date):
        if pd.isna(input_date):
            return df.iloc[0:0].copy()

        start_date = input_date - pd.Timedelta(days=self.window_days)

        return df[
            (df[self.date_col] >= start_date)
            & (df[self.date_col] < input_date)
        ].copy()

    def _top_rejection_code(self, df):
        if df.empty:
            return ""

        for col in ["PBM_REJ_CODE", "REJ_CODE"]:
            if col not in df.columns:
                continue

            s = df[col].dropna().astype(str).str.upper().str.strip()
            s = s[~s.isin(["", "NAN", "NONE", "NULL", "UNKNOWN"])]

            if len(s) > 0:
                return s.value_counts().index[0]

        return ""

    def _cost_position(self, costs, current_cost):
        costs = pd.to_numeric(costs, errors="coerce").dropna()

        if len(costs) < 5:
            return "LIMITED_HISTORY"

        q3 = costs.quantile(0.75)
        p90 = costs.quantile(0.90)

        if current_cost > p90:
            return "ABOVE_RECENT_P90"

        if current_cost >= q3:
            return "UPPER_RECENT_BAND"

        return "NORMAL_RECENT_BAND"

    def analyze(self, row: dict):
        df = self._prepare_df(self.df)

        provider = self._clean_text(row.get("PROV_NAME", ""))
        doctor = self._clean_text(row.get("DOC_NAME", ""))
        diagnosis = self._clean_text(row.get("PA_PRIMARY_DIAG", ""))

        treatment = self._clean_code(
            row.get("PROV_TREAT_CODE_CLEAN", row.get("PROV_TREAT_CODE", ""))
        )

        ins_code = self._clean_code(row.get("INS_TREAT_CODE", ""))
        drug_name = self._clean_text(row.get("PAT_DRUG_NAME", ""))

        current_cost = pd.to_numeric(
            row.get("PA_EST_AMT_LC", 0),
            errors="coerce"
        )

        if pd.isna(current_cost):
            current_cost = 0

        input_date = self._parse_date_value(row.get(self.date_col))
        recent_df = self._recent_window(df, input_date)

        findings = []

        evidence = {
            "Provider": provider,
            "Diagnosis": diagnosis,
            "Treatment": treatment,
            "Insurance Code": ins_code,
            "Input Service Date": str(row.get(self.date_col)),
            "Parsed Input Date": str(input_date),
            "Recent Window": f"Last {self.window_days} days before input service date",
            "Recent Window Total Claims": int(len(recent_df)),
            "Recent Provider Claims": 0,
            "Match Level Used": "NO_MATCH",
            "Matching Provider Claims": 0,
            "Matching Review Claims": 0,
            "Matching Full Reject Claims": 0,
            "Matching Partial Approval Claims": 0,
            "Top Rejection Code": "Not available",
            "Current Cost Position": "Not available",
        }

        if recent_df.empty:
            findings.append(
                "No recent claim history available before the input service date."
            )
            return {
                "PROVIDER_FINDINGS": findings,
                "PROVIDER_EVIDENCE": evidence,
                "PROVIDER_REVIEW_SUMMARY": findings[0],
            }

        provider_df = recent_df[
            recent_df["PROV_NAME"] == provider
        ].copy()

        evidence["Recent Provider Claims"] = int(len(provider_df))

        if provider_df.empty:
            findings.append(
                "No recent provider history found for the current claim provider."
            )
            return {
                "PROVIDER_FINDINGS": findings,
                "PROVIDER_EVIDENCE": evidence,
                "PROVIDER_REVIEW_SUMMARY": findings[0],
            }

        level_1 = provider_df[
            (provider_df["PA_PRIMARY_DIAG"] == diagnosis)
            & (provider_df["TREATMENT_KEY"] == treatment)
            & (provider_df["INS_TREAT_CODE"] == ins_code)
        ].copy()

        level_2 = provider_df[
            (provider_df["PA_PRIMARY_DIAG"] == diagnosis)
            & (provider_df["TREATMENT_KEY"] == treatment)
        ].copy()

        level_3 = provider_df[
            provider_df["PA_PRIMARY_DIAG"] == diagnosis
        ].copy()

        level_4 = provider_df[
            provider_df["TREATMENT_KEY"] == treatment
        ].copy()

        if len(level_1) > 0:
            matched_df = level_1
            match_level = "PROVIDER_DIAG_TREAT_INS"
        elif len(level_2) > 0:
            matched_df = level_2
            match_level = "PROVIDER_DIAG_TREAT"
            findings.append(
                "No insurance-code-specific recent match found. Using provider diagnosis-treatment history."
            )
        elif len(level_3) > 0:
            matched_df = level_3
            match_level = "PROVIDER_DIAG"
            findings.append(
                "No provider diagnosis-treatment recent match found. Using provider diagnosis-level history."
            )
        elif len(level_4) > 0:
            matched_df = level_4
            match_level = "PROVIDER_TREAT"
            findings.append(
                "No provider diagnosis match found. Using provider treatment-level history."
            )
        else:
            findings.append(
                "Current claim does not match recent provider diagnosis or treatment history."
            )
            return {
                "PROVIDER_FINDINGS": findings,
                "PROVIDER_EVIDENCE": evidence,
                "PROVIDER_REVIEW_SUMMARY": " ".join(findings),
            }

        evidence["Match Level Used"] = match_level
        evidence["Matching Provider Claims"] = int(len(matched_df))

        review_count = int(matched_df["REVIEW_FLAG"].sum())
        full_reject_count = int(matched_df["FULL_REJECT_FLAG"].sum())
        partial_count = int(matched_df["PARTIAL_APPROVAL_FLAG"].sum())

        evidence["Matching Review Claims"] = review_count
        evidence["Matching Full Reject Claims"] = full_reject_count
        evidence["Matching Partial Approval Claims"] = partial_count

        top_rej = self._top_rejection_code(
            matched_df[matched_df["REVIEW_FLAG"] == 1]
        )

        if top_rej:
            evidence["Top Rejection Code"] = top_rej

        provider_review_rate = (
            provider_df["REVIEW_FLAG"].sum() / len(provider_df)
            if len(provider_df) > 0
            else 0
        )

        matched_review_rate = (
            review_count / len(matched_df)
            if len(matched_df) > 0
            else 0
        )

        if (
            len(matched_df) >= 3
            and review_count >= 1
            and matched_review_rate > max(provider_review_rate * 1.5, 0.10)
        ):
            msg = (
                f"Current claim matches recent provider submissions with elevated "
                f"rejection or partial approval activity under match level {match_level}."
            )

            if top_rej:
                msg += f" Most frequent rejection code observed: {top_rej}."

            findings.append(msg)

        if len(matched_df) >= 5:
            findings.append(
                f"Current claim matches a repeatedly submitted recent provider context "
                f"({match_level}) with {len(matched_df)} matching claims."
            )

        provider_diag_df = provider_df[
            provider_df["PA_PRIMARY_DIAG"] == diagnosis
        ].copy()

        if len(provider_diag_df) >= 5:
            treatment_counts = provider_diag_df["TREATMENT_KEY"].value_counts()

            if len(treatment_counts) > 0:
                top_treatment = treatment_counts.index[0]
                top_count = int(treatment_counts.iloc[0])
                dominance_ratio = top_count / len(provider_diag_df)

                if top_treatment == treatment and dominance_ratio >= 0.60:
                    findings.append(
                        f"For diagnosis {diagnosis}, this provider's recent submissions are concentrated toward treatment {treatment} "
                        f"({top_count} of {len(provider_diag_df)} recent diagnosis-level claims)."
                    )

        cost_position = self._cost_position(
            matched_df["PA_EST_AMT_LC"],
            current_cost
        )

        evidence["Current Cost Position"] = cost_position

        if cost_position == "ABOVE_RECENT_P90":
            findings.append(
                f"Current claim amount is above the recent P90 pricing range for matched provider context {match_level}."
            )
        elif cost_position == "UPPER_RECENT_BAND":
            findings.append(
                f"Current claim amount falls within the upper recent pricing band for matched provider context {match_level}."
            )

        if len(matched_df) <= 2 and review_count > 0:
            findings.append(
                "Current context has limited recent provider history, but the matching submissions include rejection or partial approval."
            )

        if doctor:
            doctor_df = matched_df[
                matched_df["DOC_NAME"] == doctor
            ].copy()

            if len(doctor_df) >= 3:
                doc_review_count = int(doctor_df["REVIEW_FLAG"].sum())

                if doc_review_count > 0:
                    findings.append(
                        f"Current claim matches recent provider-doctor submissions in the same context, "
                        f"with {doc_review_count} reviewed or partially approved matching claims."
                    )

        if drug_name:
            drug_df = matched_df[
                matched_df["PAT_DRUG_NAME"] == drug_name
            ].copy()

            if len(drug_df) >= 3:
                drug_review_count = int(drug_df["REVIEW_FLAG"].sum())

                if drug_review_count > 0:
                    findings.append(
                        f"Current drug {drug_name} appears in recent matching provider submissions, "
                        f"with {drug_review_count} reviewed or partially approved claims."
                    )

        if len(findings) == 0:
            findings.append(
                "No significant provider-level operational review pattern observed for this current claim context in recent history."
            )

        return {
            "PROVIDER_FINDINGS": findings,
            "PROVIDER_EVIDENCE": evidence,
            "PROVIDER_REVIEW_SUMMARY": " ".join(findings),
        }