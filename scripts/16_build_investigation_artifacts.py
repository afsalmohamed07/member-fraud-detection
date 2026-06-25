import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config_loader import ConfigLoader


MIN_SIMILAR_SUPPORT = 2


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).upper().strip()
    value = " ".join(value.split())

    if value in ["", "NAN", "NONE", "NULL", "UNKNOWN"]:
        return ""

    if value.endswith(".0"):
        value = value[:-2]

    return value


def clean_code(value):
    return clean_text(value).replace(" ", "")


def prepare_df(df):
    df = df.copy()

    required_cols = [
        "PROV_NAME",
        "DOC_NAME",
        "PA_PRIMARY_DIAG",
        "PROV_TREAT_CODE",
        "PROV_TREAT_CODE_CLEAN",
        "CPT_CODE",
        "INS_TREAT_CODE",
        "PAT_DRUG_NAME",
        "DRUG_DURATION",
        "DOSAGE_DESC",
        "PA_EST_AMT_LC",
        "PA_APPR_AMT_LC",
        "PA_REJ_AMT_LC",
        "REJ_CODE",
        "REJ_DESC",
        "PBM_REJ_CODE",
        "PBM_REJ_DESC",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    text_cols = [
        "PROV_NAME",
        "DOC_NAME",
        "PA_PRIMARY_DIAG",
        "PAT_DRUG_NAME",
        "DRUG_DURATION",
        "DOSAGE_DESC",
        "REJ_CODE",
        "REJ_DESC",
        "PBM_REJ_CODE",
        "PBM_REJ_DESC",
    ]

    code_cols = [
        "PROV_TREAT_CODE",
        "PROV_TREAT_CODE_CLEAN",
        "CPT_CODE",
        "INS_TREAT_CODE",
    ]

    for col in text_cols:
        df[col] = df[col].map(clean_text)

    for col in code_cols:
        df[col] = df[col].map(clean_code)

    df["TREATMENT_KEY"] = np.where(
        df["PROV_TREAT_CODE_CLEAN"].astype(str).str.strip() != "",
        df["PROV_TREAT_CODE_CLEAN"],
        df["PROV_TREAT_CODE"],
    )

    for col in ["PA_EST_AMT_LC", "PA_APPR_AMT_LC", "PA_REJ_AMT_LC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

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

    df["TOP_REJ_CODE_SOURCE"] = np.where(
        df["PBM_REJ_CODE"].astype(str).str.strip() != "",
        df["PBM_REJ_CODE"],
        df["REJ_CODE"],
    )

    df["TOP_REJ_DESC_SOURCE"] = np.where(
        df["PBM_REJ_DESC"].astype(str).str.strip() != "",
        df["PBM_REJ_DESC"],
        df["REJ_DESC"],
    )

    return df


def _top_mode_map(df, group_cols, value_col):
    review_df = df[
        (df["REVIEW_FLAG"] == 1)
        & (df[value_col].astype(str).str.strip() != "")
    ]

    if review_df.empty:
        return {}

    temp = (
        review_df
        .groupby(group_cols + [value_col], dropna=False)
        .size()
        .reset_index(name="COUNT")
    )

    temp = temp.sort_values(
        group_cols + ["COUNT"],
        ascending=[True] * len(group_cols) + [False]
    )

    temp = temp.drop_duplicates(group_cols)

    result = {}

    for _, row in temp.iterrows():
        key = tuple(row[col] for col in group_cols)
        result[key] = row[value_col]

    return result


def build_provider_artifact(df):
    group_cols = [
        "PROV_NAME",
        "PA_PRIMARY_DIAG",
        "TREATMENT_KEY",
    ]

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            MATCHING_PROVIDER_CLAIMS=("PA_EST_AMT_LC", "size"),
            MATCHING_REVIEW_CLAIMS=("REVIEW_FLAG", "sum"),
            MATCHING_FULL_REJECT_CLAIMS=("FULL_REJECT_FLAG", "sum"),
            MATCHING_PARTIAL_APPROVAL_CLAIMS=("PARTIAL_APPROVAL_FLAG", "sum"),
            CROSS_DOCTOR_COUNT=("DOC_NAME", "nunique"),
            MEDIAN_COST=("PA_EST_AMT_LC", "median"),
            P90_COST=("PA_EST_AMT_LC", lambda x: x.quantile(0.90)),
            MAX_COST=("PA_EST_AMT_LC", "max"),
        )
        .reset_index()
    )

    top_code_map = _top_mode_map(df, group_cols, "TOP_REJ_CODE_SOURCE")
    top_desc_map = _top_mode_map(df, group_cols, "TOP_REJ_DESC_SOURCE")

    artifact = {}

    for _, row in agg.iterrows():
        key = tuple(row[col] for col in group_cols)
        total = int(row["MATCHING_PROVIDER_CLAIMS"])
        review_count = int(row["MATCHING_REVIEW_CLAIMS"])

        artifact[key] = {
            "MATCH_LEVEL": "PROVIDER_DIAG_TREAT",
            "MATCHING_PROVIDER_CLAIMS": total,
            "MATCHING_REVIEW_CLAIMS": review_count,
            "MATCHING_FULL_REJECT_CLAIMS": int(row["MATCHING_FULL_REJECT_CLAIMS"]),
            "MATCHING_PARTIAL_APPROVAL_CLAIMS": int(row["MATCHING_PARTIAL_APPROVAL_CLAIMS"]),
            "REVIEW_RATE": round((review_count / total) * 100, 2) if total else 0,
            "CROSS_DOCTOR_COUNT": int(row["CROSS_DOCTOR_COUNT"]),
            "TOP_REJECTION_CODE": top_code_map.get(key, ""),
            "TOP_REJECTION_REASON": top_desc_map.get(key, ""),
            "MEDIAN_COST": round(float(row["MEDIAN_COST"]), 2),
            "P90_COST": round(float(row["P90_COST"]), 2),
            "MAX_COST": round(float(row["MAX_COST"]), 2),
        }

    return artifact


def build_similar_claims_artifact(df):
    artifact = {}

    context_sets = [
        (
            "DIAG_TREAT_PROVIDER_DOCTOR",
            ["PA_PRIMARY_DIAG", "TREATMENT_KEY", "PROV_NAME", "DOC_NAME"],
        ),
        (
            "DIAG_TREAT_PROVIDER",
            ["PA_PRIMARY_DIAG", "TREATMENT_KEY", "PROV_NAME"],
        ),
        (
            "DIAG_TREAT",
            ["PA_PRIMARY_DIAG", "TREATMENT_KEY"],
        ),
        (
            "DIAG_CPT_INS_PROVIDER",
            ["PA_PRIMARY_DIAG", "CPT_CODE", "INS_TREAT_CODE", "PROV_NAME"],
        ),
    ]

    for context_name, cols in context_sets:
        print(f"Building similar context: {context_name}")

        agg = (
            df.groupby(cols, dropna=False)
            .agg(
                SIMILAR_CLAIMS_COUNT=("PA_EST_AMT_LC", "size"),
                SIMILAR_REJECTED_COUNT=("REVIEW_FLAG", "sum"),
                SIMILAR_MEDIAN_COST=("PA_EST_AMT_LC", "median"),
                SIMILAR_P90_COST=("PA_EST_AMT_LC", lambda x: x.quantile(0.90)),
                SIMILAR_MAX_COST=("PA_EST_AMT_LC", "max"),
            )
            .reset_index()
        )

        agg = agg[agg["SIMILAR_CLAIMS_COUNT"] >= MIN_SIMILAR_SUPPORT]

        top_code_map = _top_mode_map(df, cols, "TOP_REJ_CODE_SOURCE")
        top_desc_map = _top_mode_map(df, cols, "TOP_REJ_DESC_SOURCE")

        for _, row in agg.iterrows():
            key_values = tuple(row[col] for col in cols)
            total = int(row["SIMILAR_CLAIMS_COUNT"])
            rejected = int(row["SIMILAR_REJECTED_COUNT"])

            artifact[(context_name, key_values)] = {
                "CONTEXT_NAME": context_name,
                "CONTEXT_FIELDS": cols,
                "SIMILAR_CLAIMS_COUNT": total,
                "SIMILAR_REJECTED_COUNT": rejected,
                "SIMILAR_REJECTION_RATE": round((rejected / total) * 100, 2) if total else 0,
                "SIMILAR_MEDIAN_COST": round(float(row["SIMILAR_MEDIAN_COST"]), 2),
                "SIMILAR_P90_COST": round(float(row["SIMILAR_P90_COST"]), 2),
                "SIMILAR_MAX_COST": round(float(row["SIMILAR_MAX_COST"]), 2),
                "SIMILAR_TOP_REJ_CODE": top_code_map.get(key_values, ""),
                "SIMILAR_TOP_REJ_DESC": top_desc_map.get(key_values, ""),
            }

    return artifact


def build_upcoding_artifact(df):
    group_cols = [
        "PA_PRIMARY_DIAG",
        "PROV_NAME",
        "TREATMENT_KEY",
    ]

    provider_treat = (
        df.groupby(group_cols, dropna=False)
        .agg(
            PROVIDER_DIAG_TREAT_COUNT=("PA_EST_AMT_LC", "size"),
            TREATMENT_P75_COST=("PA_EST_AMT_LC", lambda x: x.quantile(0.75)),
            TREATMENT_P90_COST=("PA_EST_AMT_LC", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )

    provider_diag = (
        df.groupby(["PA_PRIMARY_DIAG", "PROV_NAME"], dropna=False)
        .size()
        .reset_index(name="PROVIDER_DIAG_TOTAL")
    )

    global_diag = (
        df.groupby("PA_PRIMARY_DIAG", dropna=False)
        .agg(
            GLOBAL_DIAG_TOTAL=("PA_EST_AMT_LC", "size"),
            DIAG_P75_COST=("PA_EST_AMT_LC", lambda x: x.quantile(0.75)),
            DIAG_P90_COST=("PA_EST_AMT_LC", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )

    global_diag_treat = (
        df.groupby(["PA_PRIMARY_DIAG", "TREATMENT_KEY"], dropna=False)
        .size()
        .reset_index(name="GLOBAL_DIAG_TREAT_COUNT")
    )

    temp = provider_treat.merge(
        provider_diag,
        on=["PA_PRIMARY_DIAG", "PROV_NAME"],
        how="left"
    )

    temp = temp.merge(
        global_diag,
        on="PA_PRIMARY_DIAG",
        how="left"
    )

    temp = temp.merge(
        global_diag_treat,
        on=["PA_PRIMARY_DIAG", "TREATMENT_KEY"],
        how="left"
    )

    temp["GLOBAL_DIAG_TREAT_COUNT"] = temp["GLOBAL_DIAG_TREAT_COUNT"].fillna(0)

    temp["GLOBAL_TREAT_USAGE_PCT"] = np.where(
        temp["GLOBAL_DIAG_TOTAL"] > 0,
        (temp["GLOBAL_DIAG_TREAT_COUNT"] / temp["GLOBAL_DIAG_TOTAL"]) * 100,
        0
    )

    temp["PROVIDER_TREAT_USAGE_PCT"] = np.where(
        temp["PROVIDER_DIAG_TOTAL"] > 0,
        (temp["PROVIDER_DIAG_TREAT_COUNT"] / temp["PROVIDER_DIAG_TOTAL"]) * 100,
        0
    )

    temp["USAGE_RATIO"] = np.where(
        temp["GLOBAL_TREAT_USAGE_PCT"] > 0,
        temp["PROVIDER_TREAT_USAGE_PCT"] / temp["GLOBAL_TREAT_USAGE_PCT"],
        0
    )

    artifact = {}

    for _, row in temp.iterrows():
        key = (
            row["PA_PRIMARY_DIAG"],
            row["PROV_NAME"],
            row["TREATMENT_KEY"],
        )

        artifact[key] = {
            "GLOBAL_DIAG_TOTAL": int(row["GLOBAL_DIAG_TOTAL"]),
            "GLOBAL_DIAG_TREAT_COUNT": int(row["GLOBAL_DIAG_TREAT_COUNT"]),
            "GLOBAL_TREAT_USAGE_PCT": round(float(row["GLOBAL_TREAT_USAGE_PCT"]), 2),
            "PROVIDER_DIAG_TOTAL": int(row["PROVIDER_DIAG_TOTAL"]),
            "PROVIDER_DIAG_TREAT_COUNT": int(row["PROVIDER_DIAG_TREAT_COUNT"]),
            "PROVIDER_TREAT_USAGE_PCT": round(float(row["PROVIDER_TREAT_USAGE_PCT"]), 2),
            "USAGE_RATIO": round(float(row["USAGE_RATIO"]), 2),
            "DIAG_P75_COST": round(float(row["DIAG_P75_COST"]), 2),
            "DIAG_P90_COST": round(float(row["DIAG_P90_COST"]), 2),
            "TREATMENT_P75_COST": round(float(row["TREATMENT_P75_COST"]), 2),
            "TREATMENT_P90_COST": round(float(row["TREATMENT_P90_COST"]), 2),
        }

    return artifact


def main():
    print("\n======================================")
    print("BUILDING INVESTIGATION ARTIFACTS")
    print("======================================")

    config = ConfigLoader().get_config()

    df = joblib.load(config["paths"]["processed_data"])
    print("Processed shape:", df.shape)

    df = prepare_df(df)

    output_dir = Path(config["paths"]["models_dir"]).parent / "intelligence"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nBuilding provider artifact...")
    provider_artifact = build_provider_artifact(df)

    print("\nBuilding similar claims artifact...")
    similar_artifact = build_similar_claims_artifact(df)

    print("\nBuilding upcoding artifact...")
    upcoding_artifact = build_upcoding_artifact(df)

    joblib.dump(provider_artifact, output_dir / "provider_intelligence_artifact.pkl")
    joblib.dump(similar_artifact, output_dir / "similar_claims_artifact.pkl")
    joblib.dump(upcoding_artifact, output_dir / "upcoding_artifact.pkl")

    print("Provider artifact:", len(provider_artifact))
    print("Similar claims artifact:", len(similar_artifact))
    print("Upcoding artifact:", len(upcoding_artifact))

    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()