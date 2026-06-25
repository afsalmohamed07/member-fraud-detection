
import pandas as pd
import numpy as np


# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_excel(
    "data/raw/claims.xlsx",
    engine="openpyxl"
)


# =====================================================
# IMPORTANT COLUMNS
# =====================================================
feature_cols = [

    "PA_PRIMARY_DIAG",

    "PROV_TREAT_CODE",
    "INS_TREAT_CODE",
    "CPT_CODE",

    "PAT_DRUG_NAME",

    "PROV_NAME",
    "DOC_NAME",

    "DRUG_DURATION",
    "DOSAGE_DESC",

    "SERVICE_TYPE",
    "PA_CATG",
]


financial_cols = [
    "PA_EST_AMT_LC",
    "PA_APPR_AMT_LC",
    "PA_REJ_AMT_LC",
]


required_cols = feature_cols + financial_cols


missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise Exception(f"Missing columns: {missing}")


# =====================================================
# CLEANING
# =====================================================
for col in feature_cols:

    df[col] = (
        df[col]
        .astype(str)
        .fillna("UNKNOWN")
        .str.upper()
        .str.strip()
    )


for col in financial_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)


# =====================================================
# REJECTION FLAG
# =====================================================
df["REJECTION_FLAG"] = (
    (df["PA_REJ_AMT_LC"] > 0) |
    (
        (df["PA_EST_AMT_LC"] > 0) &
        (df["PA_APPR_AMT_LC"] < df["PA_EST_AMT_LC"])
    )
).astype(int)


# =====================================================
# FEATURE IMPACT ANALYSIS
# =====================================================
results = []


for feature in feature_cols:

    temp = df.copy()

    grp = temp.groupby(feature).agg(

        TOTAL_CLAIMS=(
            "REJECTION_FLAG",
            "count"
        ),

        REJECTED_CLAIMS=(
            "REJECTION_FLAG",
            "sum"
        ),

        MEDIAN_REJ_AMT=(
            "PA_REJ_AMT_LC",
            "median"
        ),

        AVG_REJ_AMT=(
            "PA_REJ_AMT_LC",
            "mean"
        ),

    ).reset_index()


    grp["REJECTION_RATE"] = (
        grp["REJECTED_CLAIMS"] /
        grp["TOTAL_CLAIMS"]
    ) * 100


    # remove weak groups
    grp = grp[
        grp["TOTAL_CLAIMS"] >= 20
    ]


    if len(grp) == 0:
        continue


    # =====================================================
    # FEATURE IMPACT SCORE
    # =====================================================

    rejection_rate_std = grp["REJECTION_RATE"].std()

    median_rej_std = grp["MEDIAN_REJ_AMT"].std()

    unique_values = grp[feature].nunique()


    impact_score = (
        rejection_rate_std * 0.7
        +
        median_rej_std * 0.3
    )


    results.append({

        "FEATURE": feature,

        "UNIQUE_VALUES": unique_values,

        "AVG_REJECTION_RATE": round(
            grp["REJECTION_RATE"].mean(),
            2
        ),

        "REJECTION_RATE_STD": round(
            rejection_rate_std,
            2
        ),

        "MEDIAN_REJ_STD": round(
            median_rej_std,
            2
        ),

        "FEATURE_IMPACT_SCORE": round(
            impact_score,
            2
        ),
    })


# =====================================================
# FINAL OUTPUT
# =====================================================
result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    by="FEATURE_IMPACT_SCORE",
    ascending=False
)


print("\n===================================")
print("SIMILARITY FEATURE IMPORTANCE")
print("===================================\n")

print(result_df)


# =====================================================
# SAVE
# =====================================================
result_df.to_excel(
    "similarity_feature_analysis.xlsx",
    index=False
)

print("\nSaved:")
print("similarity_feature_analysis.xlsx")