import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class BaselineScorer:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]
        self.min_support = config["baseline"]["min_support"]
        self.base_dir = Path(config["paths"]["baselines_dir"])

        self.known_diagnosis = self._load_baseline("known_diagnosis.pkl")
        self.cost_context = self._load_baseline("cost_context.pkl")
        self.provider_context_cost = self._load_baseline("provider_context_cost.pkl")
        self.diagnosis_treatment_usage = self._load_baseline("diagnosis_treatment_usage.pkl")
        self.diagnosis_drug_usage = self._load_baseline("diagnosis_drug_usage.pkl")
        self.drug_duration = self._load_baseline("drug_duration.pkl")
        self.drug_dosage = self._load_baseline("drug_dosage.pkl")
        self.drug_context_price = self._load_baseline("drug_context_price.pkl")
        self.doctor_drug_behavior = self._load_baseline("doctor_drug_behavior.pkl")

        self.doctor_facility_behavior = self._load_optional_baseline("doctor_facility_behavior.pkl")
        self.doctor_treatment_behavior = self._load_optional_baseline("doctor_treatment_behavior.pkl")
        self.doctor_drug_behavior_new = self._load_optional_baseline("doctor_drug_behavior_new.pkl")

        self.duration_baseline = self._load_optional_baseline("duration_baseline.pkl")
        self.dosage_baseline = self._load_optional_baseline("dosage_baseline.pkl")
        self.provider_behavior = self._load_optional_baseline("provider_behavior.pkl")
        self.license_baseline = self._load_optional_baseline("license_baseline.pkl")

    def _load_baseline(self, file_name):
        path = self.base_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"Baseline file missing: {path}")

        df = joblib.load(path)
        print(f"Loaded baseline: {file_name} -> {df.shape}")
        return df

    def _load_optional_baseline(self, file_name):
        path = self.base_dir / file_name

        if not path.exists():
            print(f"Optional baseline not found: {file_name}")
            return None

        df = joblib.load(path)
        print(f"Loaded optional baseline: {file_name} -> {df.shape}")
        return df

    def _safe_merge(self, left, right, on, name):
        before = len(left)

        dup_count = right.duplicated(subset=on).sum()
        if dup_count > 0:
            print(f"WARNING: {name} duplicate keys: {dup_count}")
            right = right.drop_duplicates(subset=on, keep="first").copy()

        out = left.merge(right, on=on, how="left")

        if len(out) != before:
            raise RuntimeError(
                f"{name} merge row count changed. Before={before}, After={len(out)}"
            )

        return out

    def _drug_related_mask(self, df):
        drug_col = self.columns["drug_name"]

        drug_present = (
            df[drug_col].notna() &
            (df[drug_col].astype(str).str.strip() != "")
        )

        return (
            (df["PROV_TREAT_CODE_CLEAN"] == "PHARMACY") |
            (df["PA_CATG_NORM"] == "PHARMACY") |
            drug_present
        )

    def _doctor_key(self, series):
        return (
            series.astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # =====================================================
    # DOCTOR FACILITY BEHAVIOR
    # =====================================================
    def score_doctor_facility_behavior(self, df):
        df = df.copy()

        doctor_col = self.columns["doctor"]

        if doctor_col not in df.columns:
            df["DOCTOR_FACILITY_ANOMALY"] = 0
            df["DOCTOR_FACILITY_STATUS"] = "DOCTOR_COLUMN_MISSING"
            df["DOCTOR_FACILITY_EXPLANATION"] = ""
            return df

        if (
            self.doctor_facility_behavior is None
            or self.doctor_facility_behavior.empty
            or "FACILITIES" not in df.columns
        ):
            df["DOCTOR_FACILITY_ANOMALY"] = 0
            df["DOCTOR_FACILITY_STATUS"] = "NOT_AVAILABLE"
            df["DOCTOR_FACILITY_EXPLANATION"] = ""
            return df

        df["DOC_NAME_KEY"] = self._doctor_key(df[doctor_col])

        df["FACILITY_KEY"] = (
            df["FACILITIES"]
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        base = self.doctor_facility_behavior.copy()

        doctor_total = (
            base[["DOC_NAME_KEY", "DOCTOR_TOTAL_FACILITY_ROWS"]]
            .drop_duplicates("DOC_NAME_KEY")
            .copy()
        )

        pair_base = base[
            [
                "DOC_NAME_KEY",
                "FACILITY_KEY",
                "DOCTOR_FACILITY_COUNT",
                "DOCTOR_FACILITY_USAGE_%"
            ]
        ].copy()

        df = self._safe_merge(
            df,
            doctor_total,
            on=["DOC_NAME_KEY"],
            name="doctor_facility_total"
        )

        df = self._safe_merge(
            df,
            pair_base,
            on=["DOC_NAME_KEY", "FACILITY_KEY"],
            name="doctor_facility_pair"
        )

        df["DOCTOR_TOTAL_FACILITY_ROWS"] = df["DOCTOR_TOTAL_FACILITY_ROWS"].fillna(0)
        df["DOCTOR_FACILITY_COUNT"] = df["DOCTOR_FACILITY_COUNT"].fillna(0)
        df["DOCTOR_FACILITY_USAGE_%"] = df["DOCTOR_FACILITY_USAGE_%"].fillna(0)

        df["DOCTOR_FACILITY_ANOMALY"] = (
            (df["DOCTOR_TOTAL_FACILITY_ROWS"] >= self.min_support) &
            (df["DOCTOR_FACILITY_COUNT"] == 0)
        ).astype(int)

        df["DOCTOR_FACILITY_STATUS"] = "NORMAL_FACILITY"

        df.loc[
            df["DOCTOR_TOTAL_FACILITY_ROWS"] == 0,
            "DOCTOR_FACILITY_STATUS"
        ] = "NO_DOCTOR_FACILITY_HISTORY"

        df.loc[
            df["DOCTOR_FACILITY_ANOMALY"] == 1,
            "DOCTOR_FACILITY_STATUS"
        ] = "UNSEEN_FACILITY_FOR_DOCTOR"

        df["DOCTOR_FACILITY_EXPLANATION"] = ""

        df.loc[
            df["DOCTOR_FACILITY_STATUS"] == "UNSEEN_FACILITY_FOR_DOCTOR",
            "DOCTOR_FACILITY_EXPLANATION"
        ] = "Doctor exists historically but this facility is not seen for this doctor"

        return df

    # =====================================================
    # DOCTOR TREATMENT BEHAVIOR
    # =====================================================
    def score_doctor_treatment_behavior(self, df):
        df = df.copy()

        doctor_col = self.columns["doctor"]

        if doctor_col not in df.columns:
            df["DOCTOR_TREATMENT_ANOMALY"] = 0
            df["DOCTOR_TREATMENT_STATUS"] = "DOCTOR_COLUMN_MISSING"
            df["DOCTOR_TREATMENT_EXPLANATION"] = ""
            return df

        if self.doctor_treatment_behavior is None or self.doctor_treatment_behavior.empty:
            df["DOCTOR_TREATMENT_ANOMALY"] = 0
            df["DOCTOR_TREATMENT_STATUS"] = "NOT_AVAILABLE"
            df["DOCTOR_TREATMENT_EXPLANATION"] = ""
            return df

        df["DOC_NAME_KEY"] = self._doctor_key(df[doctor_col])

        base = self.doctor_treatment_behavior.copy()

        doctor_total = (
            base[["DOC_NAME_KEY", "DOCTOR_TOTAL_TREATMENT_ROWS"]]
            .drop_duplicates("DOC_NAME_KEY")
            .copy()
        )

        pair_base = base[
            [
                "DOC_NAME_KEY",
                "PROV_TREAT_CODE_CLEAN",
                "DOCTOR_TREATMENT_COUNT",
                "DOCTOR_TREATMENT_USAGE_%"
            ]
        ].copy()

        df = self._safe_merge(
            df,
            doctor_total,
            on=["DOC_NAME_KEY"],
            name="doctor_treatment_total"
        )

        df = self._safe_merge(
            df,
            pair_base,
            on=["DOC_NAME_KEY", "PROV_TREAT_CODE_CLEAN"],
            name="doctor_treatment_pair"
        )

        df["DOCTOR_TOTAL_TREATMENT_ROWS"] = df["DOCTOR_TOTAL_TREATMENT_ROWS"].fillna(0)
        df["DOCTOR_TREATMENT_COUNT"] = df["DOCTOR_TREATMENT_COUNT"].fillna(0)
        df["DOCTOR_TREATMENT_USAGE_%"] = df["DOCTOR_TREATMENT_USAGE_%"].fillna(0)

        df["DOCTOR_TREATMENT_ANOMALY"] = (
            (df["DOCTOR_TOTAL_TREATMENT_ROWS"] >= self.min_support) &
            (df["DOCTOR_TREATMENT_COUNT"] == 0)
        ).astype(int)

        df["DOCTOR_TREATMENT_STATUS"] = "NORMAL_TREATMENT"

        df.loc[
            df["DOCTOR_TOTAL_TREATMENT_ROWS"] == 0,
            "DOCTOR_TREATMENT_STATUS"
        ] = "NO_DOCTOR_TREATMENT_HISTORY"

        df.loc[
            df["DOCTOR_TREATMENT_ANOMALY"] == 1,
            "DOCTOR_TREATMENT_STATUS"
        ] = "UNSEEN_TREATMENT_FOR_DOCTOR"

        df["DOCTOR_TREATMENT_EXPLANATION"] = ""

        df.loc[
            df["DOCTOR_TREATMENT_STATUS"] == "UNSEEN_TREATMENT_FOR_DOCTOR",
            "DOCTOR_TREATMENT_EXPLANATION"
        ] = "Doctor exists historically but has not performed this treatment"

        return df

    # =====================================================
    # DOCTOR DRUG BEHAVIOR
    # =====================================================
    def score_doctor_drug_behavior(self, df):
        df = df.copy()

        doctor_col = self.columns["doctor"]
        drug_col = self.columns["drug_name"]

        if doctor_col not in df.columns:
            df["DOCTOR_DRUG_ANOMALY"] = 0
            df["DOCTOR_DRUG_STATUS"] = "DOCTOR_COLUMN_MISSING"
            df["DOCTOR_DRUG_EXPLANATION"] = ""
            return df

        if self.doctor_drug_behavior_new is None or self.doctor_drug_behavior_new.empty:
            df["DOCTOR_DRUG_ANOMALY"] = 0
            df["DOCTOR_DRUG_STATUS"] = "NOT_AVAILABLE"
            df["DOCTOR_DRUG_EXPLANATION"] = ""
            return df

        df["DOC_NAME_KEY"] = self._doctor_key(df[doctor_col])

        base = self.doctor_drug_behavior_new.copy()

        doctor_total = (
            base[["DOC_NAME_KEY", "DOCTOR_TOTAL_DRUG_ROWS"]]
            .drop_duplicates("DOC_NAME_KEY")
            .copy()
        )

        pair_base = base[
            [
                "DOC_NAME_KEY",
                drug_col,
                "DOCTOR_DRUG_COUNT",
                "DOCTOR_DRUG_USAGE_%"
            ]
        ].copy()

        df = self._safe_merge(
            df,
            doctor_total,
            on=["DOC_NAME_KEY"],
            name="doctor_drug_total"
        )

        df = self._safe_merge(
            df,
            pair_base,
            on=["DOC_NAME_KEY", drug_col],
            name="doctor_drug_pair"
        )

        df["DOCTOR_TOTAL_DRUG_ROWS"] = df["DOCTOR_TOTAL_DRUG_ROWS"].fillna(0)
        df["DOCTOR_DRUG_COUNT"] = df["DOCTOR_DRUG_COUNT"].fillna(0)
        df["DOCTOR_DRUG_USAGE_%"] = df["DOCTOR_DRUG_USAGE_%"].fillna(0)

        # =====================================================
        # NEW LOGIC
        # Same treatment + unseen drug = anomaly
        # New treatment + new drug = new pattern only
        # =====================================================

        treatment_seen = (
            df["DOCTOR_TREATMENT_COUNT"] > 0
        )

        unseen_drug = (
            df["DOCTOR_DRUG_COUNT"] == 0
        )

        enough_history = (
            df["DOCTOR_TOTAL_DRUG_ROWS"] >= self.min_support
        )

        df["DOCTOR_DRUG_ANOMALY"] = (
            enough_history
            &
            treatment_seen
            &
            unseen_drug
        ).astype(int)

        df["DOCTOR_DRUG_STATUS"] = "NORMAL_DRUG"

        df.loc[
            df["DOCTOR_TOTAL_DRUG_ROWS"] == 0,
            "DOCTOR_DRUG_STATUS"
        ] = "NO_DOCTOR_DRUG_HISTORY"

        df.loc[
            df["DOCTOR_DRUG_ANOMALY"] == 1,
            "DOCTOR_DRUG_STATUS"
        ] = "UNSEEN_DRUG_FOR_EXISTING_TREATMENT"

        df["DOCTOR_DRUG_EXPLANATION"] = ""

        df.loc[
            df["DOCTOR_DRUG_STATUS"] == "UNSEEN_DRUG_FOR_DOCTOR",
            "DOCTOR_DRUG_EXPLANATION"
        ] = "Doctor exists historically but has not prescribed this drug"

        return df

    # =====================================================
    # ISOLATION CONTEXT EXPLAINER
    # =====================================================
    def score_isolation_context(self, df):
        df = df.copy()

        diag_col = self.columns["diagnosis"]

        combo_base = (
            self.known_diagnosis[
                ["SERVICE_TYPE_NORM", "PA_CATG_NORM"]
            ]
            .drop_duplicates()
            .copy()
        )

        combo_base["SERVICE_CATEGORY_SEEN"] = 1

        df = self._safe_merge(
            df,
            combo_base,
            on=["SERVICE_TYPE_NORM", "PA_CATG_NORM"],
            name="service_category_seen"
        )

        df["SERVICE_CATEGORY_SEEN"] = (
            df["SERVICE_CATEGORY_SEEN"]
            .fillna(0)
            .astype(int)
        )

        diag_base = (
            self.known_diagnosis[
                ["PA_CATG_NORM", diag_col]
            ]
            .drop_duplicates()
            .copy()
        )

        diag_base["CATEGORY_DIAGNOSIS_SEEN"] = 1

        df = self._safe_merge(
            df,
            diag_base,
            on=["PA_CATG_NORM", diag_col],
            name="category_diagnosis_seen"
        )

        df["CATEGORY_DIAGNOSIS_SEEN"] = (
            df["CATEGORY_DIAGNOSIS_SEEN"]
            .fillna(0)
            .astype(int)
        )

        treat_base = (
            self.diagnosis_treatment_usage[
                [diag_col, "PROV_TREAT_CODE_CLEAN"]
            ]
            .drop_duplicates()
            .copy()
        )

        treat_base["DIAGNOSIS_TREATMENT_SEEN"] = 1

        df = self._safe_merge(
            df,
            treat_base,
            on=[diag_col, "PROV_TREAT_CODE_CLEAN"],
            name="diagnosis_treatment_seen"
        )

        df["DIAGNOSIS_TREATMENT_SEEN"] = (
            df["DIAGNOSIS_TREATMENT_SEEN"]
            .fillna(0)
            .astype(int)
        )

        category_treat = (
            self.cost_context[
                ["PA_CATG_NORM", "PROV_TREAT_CODE_CLEAN"]
            ]
            .drop_duplicates()
            .copy()
        )

        category_treat["CATEGORY_TREATMENT_SEEN"] = 1

        df = self._safe_merge(
            df,
            category_treat,
            on=["PA_CATG_NORM", "PROV_TREAT_CODE_CLEAN"],
            name="category_treatment_seen"
        )

        df["CATEGORY_TREATMENT_SEEN"] = (
            df["CATEGORY_TREATMENT_SEEN"]
            .fillna(0)
            .astype(int)
        )

        df["ISOLATION_CONTEXT_SCORE"] = 0

        context_cols = [
            "SERVICE_CATEGORY_SEEN",
            "CATEGORY_DIAGNOSIS_SEEN",
            "DIAGNOSIS_TREATMENT_SEEN",
            "CATEGORY_TREATMENT_SEEN",
        ]

        for col in context_cols:
            df["ISOLATION_CONTEXT_SCORE"] += (
                df[col] == 0
            ).astype(int)

        if "DOCTOR_TREATMENT_ANOMALY" in df.columns:
            df["ISOLATION_CONTEXT_SCORE"] += (
                df["DOCTOR_TREATMENT_ANOMALY"] == 1
            ).astype(int)

        if "DOCTOR_DRUG_ANOMALY" in df.columns:
            df["ISOLATION_CONTEXT_SCORE"] += (
                df["DOCTOR_DRUG_ANOMALY"] == 1
            ).astype(int)

        if "DOCTOR_FACILITY_ANOMALY" in df.columns:
            df["ISOLATION_CONTEXT_SCORE"] += (
                df["DOCTOR_FACILITY_ANOMALY"] == 1
            ).astype(int)

        df["ISOLATION_CONTEXT_ANOMALY"] = (
            df["ISOLATION_CONTEXT_SCORE"] >= 2
        ).astype(int)

        def build_reason(row):
            reasons = []

            if row["SERVICE_CATEGORY_SEEN"] == 0:
                reasons.append("New service-category combination")

            if row["CATEGORY_DIAGNOSIS_SEEN"] == 0:
                reasons.append("Diagnosis unseen for category")

            if row["DIAGNOSIS_TREATMENT_SEEN"] == 0:
                reasons.append("Treatment unseen for diagnosis")

            if row["CATEGORY_TREATMENT_SEEN"] == 0:
                reasons.append("Treatment unseen for category")

            if row.get("DOCTOR_TREATMENT_ANOMALY", 0) == 1:
                reasons.append("Doctor never performed this treatment")

            if row.get("DOCTOR_DRUG_ANOMALY", 0) == 1:
                reasons.append("Doctor never prescribed this drug")

            if row.get("DOCTOR_FACILITY_ANOMALY", 0) == 1:
                reasons.append("Doctor unusual facility usage")

            return " | ".join(reasons)

        df["ISOLATION_CONTEXT_EXPLANATION"] = (
            df.apply(build_reason, axis=1)
        )

        return df

    # =====================================================
    # UNKNOWN DIAGNOSIS
    # =====================================================
    def score_known_diagnosis(self, df):
        df = df.copy()

        diag_col = self.columns["diagnosis"]

        key_cols = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            diag_col
        ]

        base = self.known_diagnosis[
            key_cols + ["DIAG_CONTEXT_COUNT"]
        ].copy()

        df = self._safe_merge(
            df,
            base,
            on=key_cols,
            name="known_diagnosis"
        )

        df["DIAG_CONTEXT_COUNT"] = df["DIAG_CONTEXT_COUNT"].fillna(0)

        df["UNKNOWN_DIAGNOSIS_ANOMALY"] = (
            df["DIAG_CONTEXT_COUNT"] == 0
        ).astype(int)

        return df

    # =====================================================
    # COST CONTEXT
    # =====================================================
    def score_cost_context(self, df):
        df = df.copy()

        diag_col = self.columns["diagnosis"]
        amount_col = self.columns["est_amount"]

        key_cols = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            diag_col,
            "PROV_TREAT_CODE_CLEAN",
        ]

        base = self.cost_context[
            key_cols + [
                "SUPPORT_COUNT",
                "MEDIAN_AMT",
                "P75_AMT",
                "P90_AMT",
                "P95_AMT",
                "IQR_UPPER_LIMIT",
                "BASELINE_CONFIDENCE",
            ]
        ].copy()

        base = base.rename(
            columns={
                "SUPPORT_COUNT": "COST_SUPPORT_COUNT",
                "MEDIAN_AMT": "COST_MEDIAN_AMT",
                "P75_AMT": "COST_P75_AMT",
                "P90_AMT": "COST_P90_AMT",
                "P95_AMT": "COST_P95_AMT",
                "IQR_UPPER_LIMIT": "COST_IQR_UPPER_LIMIT",
                "BASELINE_CONFIDENCE": "COST_BASELINE_CONFIDENCE",
            }
        )

        df = self._safe_merge(
            df,
            base,
            on=key_cols,
            name="cost_context"
        )

        df["COST_RATIO"] = np.where(
            df["COST_MEDIAN_AMT"] > 0,
            df[amount_col] / df["COST_MEDIAN_AMT"],
            np.nan
        )

        df["COST_ANOMALY"] = (
            (df[amount_col] > df["COST_P90_AMT"]) &
            (df["COST_SUPPORT_COUNT"] >= self.min_support)
        ).fillna(False).astype(int)

        df["IQR_COST_ANOMALY"] = (
            (df[amount_col] > df["COST_IQR_UPPER_LIMIT"]) &
            (df["COST_SUPPORT_COUNT"] >= self.min_support)
        ).fillna(False).astype(int)

        return df

    # =====================================================
    # TREATMENT SEEN / RARE
    # =====================================================
    def score_rare_treatment(self, df):
        df = df.copy()

        diag_col = self.columns["diagnosis"]
        key_cols = [diag_col, "PROV_TREAT_CODE_CLEAN"]

        base = self.diagnosis_treatment_usage[
            key_cols + [
                "TREATMENT_USAGE_COUNT",
                "TOTAL_DIAG_ROWS",
                "TREATMENT_USAGE_%"
            ]
        ].copy()

        df = self._safe_merge(
            df,
            base,
            on=key_cols,
            name="diagnosis_treatment_usage"
        )

        df["TREATMENT_USAGE_COUNT"] = df["TREATMENT_USAGE_COUNT"].fillna(0)
        df["TOTAL_DIAG_ROWS"] = df["TOTAL_DIAG_ROWS"].fillna(0)
        df["TREATMENT_USAGE_%"] = df["TREATMENT_USAGE_%"].fillna(0)

        df["UNSEEN_TREATMENT_CODE_ANOMALY"] = (
            df["TREATMENT_USAGE_COUNT"] == 0
        ).astype(int)

        df["RARE_TREATMENT_ANOMALY"] = (
            (df["TREATMENT_USAGE_COUNT"] > 0) &
            (df["TREATMENT_USAGE_%"] < 1) &
            (df["TOTAL_DIAG_ROWS"] >= self.min_support)
        ).astype(int)

        return df

    # =====================================================
    # PHARMACY STRUCTURE RULES
    # =====================================================
    def score_pharmacy_rules(self, df):
        df = df.copy()

        drug_col = self.columns["drug_name"]
        pharmacy_mask = df["PROV_TREAT_CODE_CLEAN"] == "PHARMACY"

        drug_present = (
            df[drug_col].notna() &
            (df[drug_col].astype(str).str.strip() != "")
        )

        df["PHARMACY_MISSING_DRUG_ANOMALY"] = (
            pharmacy_mask & (~drug_present)
        ).astype(int)

        df["NON_PHARMACY_WITH_DRUG_ANOMALY"] = (
            (~pharmacy_mask) & drug_present
        ).astype(int)

        return df

    # =====================================================
    # DRUG MASTER + DIAGNOSIS DRUG LOGIC
    # =====================================================
    def score_rare_drug(self, df):
        df = df.copy()

        diag_col = self.columns["diagnosis"]
        drug_col = self.columns["drug_name"]

        key_cols = [diag_col, drug_col]

        base = self.diagnosis_drug_usage[
            key_cols + [
                "DRUG_USAGE_COUNT",
                "TOTAL_DIAG_PHARMACY_ROWS",
                "DRUG_USAGE_%"
            ]
        ].copy()

        df = self._safe_merge(
            df,
            base,
            on=key_cols,
            name="diagnosis_drug_usage"
        )

        drug_master = (
            self.diagnosis_drug_usage
            .groupby(drug_col, dropna=False)
            .agg(DRUG_MASTER_COUNT=("DRUG_USAGE_COUNT", "sum"))
            .reset_index()
        )

        df = self._safe_merge(
            df,
            drug_master,
            on=[drug_col],
            name="drug_master_check"
        )

        pharmacy_mask = self._drug_related_mask(df)

        df["DRUG_USAGE_%"] = df["DRUG_USAGE_%"].fillna(0)
        df["DRUG_USAGE_COUNT"] = df["DRUG_USAGE_COUNT"].fillna(0)
        df["TOTAL_DIAG_PHARMACY_ROWS"] = df["TOTAL_DIAG_PHARMACY_ROWS"].fillna(0)
        df["DRUG_MASTER_COUNT"] = df["DRUG_MASTER_COUNT"].fillna(0)

        drug_present = (
            df[drug_col].notna() &
            (df[drug_col].astype(str).str.strip() != "")
        )

        df["DRUG_NOT_IN_MASTER_ANOMALY"] = (
            pharmacy_mask &
            drug_present &
            (df["DRUG_MASTER_COUNT"] == 0)
        ).astype(int)

        df["DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY"] = (
            pharmacy_mask &
            drug_present &
            (df["DRUG_MASTER_COUNT"] > 0) &
            (df["DRUG_USAGE_COUNT"] == 0)
        ).astype(int)

        df["UNSEEN_DRUG_ANOMALY"] = (
            df["DRUG_NOT_IN_MASTER_ANOMALY"] == 1
        ).astype(int)

        df["RARE_DRUG_ANOMALY"] = (
            pharmacy_mask &
            drug_present &
            (df["DRUG_USAGE_COUNT"] > 0) &
            (df["DRUG_USAGE_COUNT"] < 10) &
            (df["DRUG_USAGE_%"] < 0.5) &
            (df["TOTAL_DIAG_PHARMACY_ROWS"] >= self.min_support)
        ).astype(int)

        df.loc[
            df["DRUG_USAGE_COUNT"] >= 10,
            "RARE_DRUG_ANOMALY"
        ] = 0

        return df

    # =====================================================
    # ADVANCED DURATION
    # =====================================================
    def score_duration_advanced(self, df):
        df = df.copy()

        drug_col = self.columns["drug_name"]

        if self.duration_baseline is None:
            df["DURATION_EXACT_SEEN"] = 0
            df["DURATION_UNSEEN_WITHIN_RANGE"] = 0
            df["DURATION_OUTSIDE_RANGE_ANOMALY"] = 0
            df["EXTREME_DURATION_ANOMALY"] = (
                df["DURATION_DAYS"].notna() &
                (df["DURATION_DAYS"] > 365)
            ).astype(int)
            return df

        base = self.duration_baseline.copy()

        df = self._safe_merge(
            df,
            base,
            on=[drug_col],
            name="duration_advanced"
        )

        pharmacy_mask = self._drug_related_mask(df)

        def exact_seen(row):
            duration = row.get("DURATION_DAYS")
            values = row.get("DURATION_LIST")

            if pd.isna(duration) or not isinstance(values, list):
                return 0

            return int(float(duration) in values)

        df["DURATION_EXACT_SEEN"] = df.apply(exact_seen, axis=1)

        df["DURATION_UNSEEN_WITHIN_RANGE"] = (
            pharmacy_mask &
            df["DURATION_DAYS"].notna() &
            (df["DURATION_EXACT_SEEN"] == 0) &
            (df["DURATION_DAYS"] >= df["MIN_DURATION"]) &
            (df["DURATION_DAYS"] <= df["MAX_DURATION"])
        ).fillna(False).astype(int)

        if "P90_DURATION" in df.columns:
            upper_limit = df["P90_DURATION"]
        else:
            upper_limit = df["MAX_DURATION"]

        df["DURATION_OUTSIDE_RANGE_ANOMALY"] = (
            pharmacy_mask &
            df["DURATION_DAYS"].notna() &
            (
                (df["DURATION_DAYS"] < df["MIN_DURATION"]) |
                (df["DURATION_DAYS"] > upper_limit)
            )
        ).fillna(False).astype(int)

        df["EXTREME_DURATION_ANOMALY"] = (
            df["DURATION_DAYS"].notna() &
            (df["DURATION_DAYS"] > 365)
        ).astype(int)

        return df

    # =====================================================
    # ADVANCED DOSAGE
    # =====================================================
    def score_dosage_advanced(self, df):
        df = df.copy()

        drug_col = self.columns["drug_name"]

        if self.dosage_baseline is None:
            df["DOSAGE_EXACT_SEEN"] = 0
            df["DOSAGE_UNSEEN_WITHIN_RANGE"] = 0
            df["DOSAGE_OUTSIDE_RANGE_ANOMALY"] = 0
            return df

        base = self.dosage_baseline.copy()

        df = self._safe_merge(
            df,
            base,
            on=[drug_col],
            name="dosage_advanced"
        )

        pharmacy_mask = self._drug_related_mask(df)

        def exact_seen(row):
            dosage = row.get("DOSAGE_PER_DAY")
            values = row.get("DOSAGE_LIST")

            if pd.isna(dosage) or not isinstance(values, list):
                return 0

            return int(float(dosage) in values)

        df["DOSAGE_EXACT_SEEN"] = df.apply(exact_seen, axis=1)

        df["DOSAGE_UNSEEN_WITHIN_RANGE"] = (
            pharmacy_mask &
            df["DOSAGE_PER_DAY"].notna() &
            (df["DOSAGE_EXACT_SEEN"] == 0) &
            (df["DOSAGE_PER_DAY"] >= df["MIN_DOSAGE"]) &
            (df["DOSAGE_PER_DAY"] <= df["MAX_DOSAGE"])
        ).fillna(False).astype(int)

        df["DOSAGE_OUTSIDE_RANGE_ANOMALY"] = (
            pharmacy_mask &
            df["DOSAGE_PER_DAY"].notna() &
            (
                (df["DOSAGE_PER_DAY"] < df["MIN_DOSAGE"]) |
                (df["DOSAGE_PER_DAY"] > df["MAX_DOSAGE"])
            )
        ).fillna(False).astype(int)

        return df

    # =====================================================
    # LICENSE
    # =====================================================
    def score_license(self, df):
        df = df.copy()

        default_cols = {
            "DOCTOR_NOT_IN_LICENSE_BASELINE": 0,
            "MULTIPLE_LICENSE_ANOMALY": 0,
            "MISSING_LICENSE_ANOMALY": 0,
            "UNSEEN_LICENSE_ANOMALY": 0,
            "LICENSE_ANOMALY": 0,
            "LICENSE_EXPLANATION": "",
        }

        if self.license_baseline is None or "DOCTOR_LICENSE" not in df.columns:
            for col, val in default_cols.items():
                df[col] = val
            return df

        doctor_col = self.columns["doctor"]

        if doctor_col not in df.columns:
            for col, val in default_cols.items():
                df[col] = val
            df["LICENSE_EXPLANATION"] = "Doctor column missing for license check"
            return df

        base = self.license_baseline.copy()

        required_base_cols = [
            "DOC_NAME_LICENSE_KEY",
            "UNIQUE_LICENSE_COUNT",
            "LICENSE_LIST",
        ]

        if base.empty or any(col not in base.columns for col in required_base_cols):
            for col, val in default_cols.items():
                df[col] = val
            df["LICENSE_EXPLANATION"] = "License baseline unavailable"
            return df

        df["DOC_NAME_LICENSE_KEY"] = self._doctor_key(df[doctor_col])

        df = self._safe_merge(
            df,
            base,
            on=["DOC_NAME_LICENSE_KEY"],
            name="license_baseline"
        )

        input_license = (
            df["DOCTOR_LICENSE"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df["DOCTOR_NOT_IN_LICENSE_BASELINE"] = (
            df["UNIQUE_LICENSE_COUNT"].isna()
        ).astype(int)

        df["UNIQUE_LICENSE_COUNT"] = df["UNIQUE_LICENSE_COUNT"].fillna(0)

        df["MULTIPLE_LICENSE_ANOMALY"] = (
            df["UNIQUE_LICENSE_COUNT"] > 1
        ).astype(int)

        df["MISSING_LICENSE_ANOMALY"] = (
            input_license.isna() |
            input_license.isin(["", "NAN", "NONE", "UNKNOWN"])
        ).astype(int)

        def unseen_license(row):
            if row.get("DOCTOR_NOT_IN_LICENSE_BASELINE", 0) == 1:
                return 0

            lic = str(row.get("DOCTOR_LICENSE", "")).strip().upper()
            lic_list = row.get("LICENSE_LIST")

            if not isinstance(lic_list, list):
                return 0

            normalized_list = [
                str(x).strip().upper()
                for x in lic_list
            ]

            return int(lic not in normalized_list)

        df["UNSEEN_LICENSE_ANOMALY"] = df.apply(unseen_license, axis=1)

        df["LICENSE_ANOMALY"] = (
            (df["MULTIPLE_LICENSE_ANOMALY"] == 1) |
            (df["MISSING_LICENSE_ANOMALY"] == 1) |
            (df["UNSEEN_LICENSE_ANOMALY"] == 1)
        ).astype(int)

        def explanation(row):
            reasons = []

            if row.get("DOCTOR_NOT_IN_LICENSE_BASELINE", 0) == 1:
                reasons.append("License history not available for this doctor")

            if row.get("MULTIPLE_LICENSE_ANOMALY", 0) == 1:
                reasons.append("Doctor has multiple licenses historically")

            if row.get("MISSING_LICENSE_ANOMALY", 0) == 1:
                reasons.append("Doctor license is missing")

            if row.get("UNSEEN_LICENSE_ANOMALY", 0) == 1:
                reasons.append("Doctor license is mismatch / unseen for this doctor")

            return " | ".join(reasons)

        df["LICENSE_EXPLANATION"] = df.apply(explanation, axis=1)

        return df

    # =====================================================
    # PROVIDER BEHAVIOR
    # =====================================================
    def score_provider_behavior(self, df):
        df = df.copy()

        if self.provider_behavior is None:
            df["PROVIDER_BEHAVIOR_ANOMALY"] = 0
            df["PROVIDER_CHARGE_CLASS"] = "UNKNOWN"
            df["PROVIDER_BEHAVIOR_EXPLANATION"] = ""
            return df

        diag_col = self.columns["diagnosis"]
        provider_col = self.columns["provider"]
        amount_col = self.columns["est_amount"]

        key_cols = [diag_col, provider_col]
        base = self.provider_behavior.copy()

        rename_map = {
            "SUPPORT_COUNT": "PROVIDER_SUPPORT_COUNT",
            "MEDIAN_COST": "PROVIDER_MEDIAN_COST",
            "P90_COST": "PROVIDER_P90_COST",
        }

        for old, new in rename_map.items():
            if old in base.columns and new not in base.columns:
                base = base.rename(columns={old: new})

        required_cols = [
            "PROVIDER_SUPPORT_COUNT",
            "PROVIDER_MEDIAN_COST",
            "PROVIDER_P90_COST",
        ]

        if any(col not in base.columns for col in required_cols):
            df["PROVIDER_BEHAVIOR_ANOMALY"] = 0
            df["PROVIDER_CHARGE_CLASS"] = "UNKNOWN"
            df["PROVIDER_BEHAVIOR_EXPLANATION"] = "Provider baseline columns unavailable"
            return df

        df = self._safe_merge(
            df,
            base,
            on=key_cols,
            name="provider_behavior"
        )

        df["PROVIDER_SUPPORT_COUNT"] = df["PROVIDER_SUPPORT_COUNT"].fillna(0)
        df["PROVIDER_MEDIAN_COST"] = df["PROVIDER_MEDIAN_COST"].fillna(0)
        df["PROVIDER_P90_COST"] = df["PROVIDER_P90_COST"].fillna(0)

        df["PROVIDER_BEHAVIOR_ANOMALY"] = 0
        df["PROVIDER_CHARGE_CLASS"] = "UNKNOWN"
        df["PROVIDER_BEHAVIOR_EXPLANATION"] = ""

        high_mask = (
            df["PROVIDER_SUPPORT_COUNT"] >= self.min_support
        ) & (
            df[amount_col] > df["PROVIDER_P90_COST"]
        )

        low_mask = (
            df["PROVIDER_SUPPORT_COUNT"] >= self.min_support
        ) & (
            df[amount_col] < df["PROVIDER_MEDIAN_COST"] * 0.5
        )

        normal_mask = (
            df["PROVIDER_SUPPORT_COUNT"] >= self.min_support
        ) & (
            df[amount_col] >= df["PROVIDER_MEDIAN_COST"] * 0.5
        ) & (
            df[amount_col] <= df["PROVIDER_P90_COST"]
        )

        df.loc[high_mask, "PROVIDER_BEHAVIOR_ANOMALY"] = 1
        df.loc[high_mask, "PROVIDER_CHARGE_CLASS"] = "CURRENT_HIGH_FOR_PROVIDER"

        df.loc[low_mask, "PROVIDER_BEHAVIOR_ANOMALY"] = 1
        df.loc[low_mask, "PROVIDER_CHARGE_CLASS"] = "CURRENT_LOW_FOR_PROVIDER"

        df.loc[normal_mask, "PROVIDER_CHARGE_CLASS"] = "NORMAL_FOR_PROVIDER"

        def provider_exp(row):
            if row.get("PROVIDER_CHARGE_CLASS") == "CURRENT_HIGH_FOR_PROVIDER":
                return "Provider current amount is higher than provider historical P90 for this diagnosis"

            if row.get("PROVIDER_CHARGE_CLASS") == "CURRENT_LOW_FOR_PROVIDER":
                return "Provider current amount is unusually lower than provider historical median for this diagnosis"

            if row.get("PROVIDER_CHARGE_CLASS") == "NORMAL_FOR_PROVIDER":
                return "Provider current amount is within normal provider range"

            return ""

        df["PROVIDER_BEHAVIOR_EXPLANATION"] = df.apply(provider_exp, axis=1)

        return df

    # =====================================================
    # PHARMACY HIGH COST SIMPLE RULE
    # =====================================================
    def score_pharmacy_high_cost(self, df):
        df = df.copy()

        amount_col = self.columns["est_amount"]
        pharmacy_mask = self._drug_related_mask(df)

        df["PHARMACY_HIGH_COST_ANOMALY"] = (
            pharmacy_mask &
            (df[amount_col] > 1000)
        ).astype(int)

        return df

    # =====================================================
    # RUN ALL
    # =====================================================
    def score_all(self, df):
        df = df.copy()

        print("\nStarting baseline scoring...")
        print("Input shape:", df.shape)

        steps = [
            ("known_diagnosis", self.score_known_diagnosis),
            ("cost_context", self.score_cost_context),
            ("rare_treatment", self.score_rare_treatment),
            ("pharmacy_rules", self.score_pharmacy_rules),
            ("rare_drug", self.score_rare_drug),
            ("duration_advanced", self.score_duration_advanced),
            ("dosage_advanced", self.score_dosage_advanced),
            ("license", self.score_license),

            ("doctor_facility_behavior", self.score_doctor_facility_behavior),
            ("doctor_treatment_behavior", self.score_doctor_treatment_behavior),
            ("doctor_drug_behavior", self.score_doctor_drug_behavior),

            ("provider_behavior", self.score_provider_behavior),
            ("isolation_context", self.score_isolation_context),

            ("pharmacy_high_cost", self.score_pharmacy_high_cost),
        ]

        for step_name, func in steps:
            before = df.shape
            print(f"\nRunning step: {step_name}")
            df = func(df)
            after = df.shape
            print(f"{step_name}: {before} -> {after}")

        anomaly_cols = [
            "UNKNOWN_DIAGNOSIS_ANOMALY",
            "COST_ANOMALY",
            "IQR_COST_ANOMALY",
            "UNSEEN_TREATMENT_CODE_ANOMALY",
            "RARE_TREATMENT_ANOMALY",

            "PHARMACY_MISSING_DRUG_ANOMALY",
            "NON_PHARMACY_WITH_DRUG_ANOMALY",

            "DRUG_NOT_IN_MASTER_ANOMALY",
            "DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY",
            "UNSEEN_DRUG_ANOMALY",
            "RARE_DRUG_ANOMALY",

            "DURATION_OUTSIDE_RANGE_ANOMALY",
            "EXTREME_DURATION_ANOMALY",

            "DOSAGE_OUTSIDE_RANGE_ANOMALY",

            "MULTIPLE_LICENSE_ANOMALY",
            "MISSING_LICENSE_ANOMALY",
            "UNSEEN_LICENSE_ANOMALY",
            "LICENSE_ANOMALY",

            "DOCTOR_FACILITY_ANOMALY",
            "DOCTOR_TREATMENT_ANOMALY",
            "DOCTOR_DRUG_ANOMALY",

            "PROVIDER_BEHAVIOR_ANOMALY",

            "ISOLATION_CONTEXT_ANOMALY",

            "PHARMACY_HIGH_COST_ANOMALY",
        ]

        for col in anomaly_cols:
            if col not in df.columns:
                df[col] = 0

            df[col] = df[col].fillna(0).astype(int)

        df["BASELINE_ANOMALY_COUNT"] = df[anomaly_cols].sum(axis=1)

        print("\nBaseline scoring finished.")
        print("Final shape:", df.shape)

        return df