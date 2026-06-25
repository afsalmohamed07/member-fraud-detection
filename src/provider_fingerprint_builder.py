import re
import pandas as pd


class ProviderFingerprintBuilder:

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

    def _top_or_unknown(self, series):
        s = (
            series
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        s = s[~s.isin(["", "NAN", "NONE", "NULL", "UNKNOWN"])]

        if len(s) == 0:
            return "UNKNOWN"

        return s.value_counts().index[0]

    def build(self, df):
        df = df.copy()

        if "PROV_TREAT_FAMILY" not in df.columns:
            df["PROV_TREAT_FAMILY"] = df["PROV_TREAT_CODE_CLEAN"].apply(
                self._get_treatment_family
            )

        if "PA_CATG_NORM" not in df.columns:
            df["PA_CATG_NORM"] = df["PA_CATG"].apply(self._clean_text)

        df["PROV_NAME"] = df["PROV_NAME"].apply(self._clean_text)
        df["PA_EST_AMT_LC"] = pd.to_numeric(
            df["PA_EST_AMT_LC"],
            errors="coerce"
        ).fillna(0)

        artifact = {}

        for provider, g in df.groupby("PROV_NAME", dropna=False):
            if provider == "":
                continue

            total = len(g)

            if total == 0:
                continue

            drug_usage_pct = round(
                (
                    g["PAT_DRUG_NAME"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .mean()
                ) * 100,
                2
            )

            artifact[provider] = {
                "TOTAL_CLAIMS": int(total),
                "TOP_CATEGORY": self._top_or_unknown(g["PA_CATG_NORM"]),
                "TOP_TREATMENT_FAMILY": self._top_or_unknown(g["PROV_TREAT_FAMILY"]),
                "USUAL_COST": round(float(g["PA_EST_AMT_LC"].median()), 2),
                "UPPER_COST": round(float(g["PA_EST_AMT_LC"].quantile(0.90)), 2),
                "DRUG_USAGE_PCT": drug_usage_pct,
            }

        return artifact