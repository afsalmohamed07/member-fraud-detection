import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_excel(
    "data/raw/claims.xlsx",
    engine="openpyxl"
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================
required_cols = [
    "PA_PRIMARY_DIAG",
    "CPT_CODE",
    "INS_TREAT_CODE",
    "PROV_TREAT_CODE",
    "PAT_DRUG_NAME",
    "DRUG_DURATION",
    "DOSAGE_DESC",
    "PROV_NAME",
    "DOC_NAME",
    "SERVICE_TYPE",
    "PA_CATG",

    "PA_EST_AMT_LC",
    "PA_APPR_AMT_LC",
    "PA_REJ_AMT_LC",

    "REJ_CODE",
    "REJ_DESC",
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise Exception(f"Missing columns: {missing}")


# =========================================================
# CLEANING
# =========================================================
for col in required_cols:
    df[col] = (
        df[col]
        .astype(str)
        .fillna("UNKNOWN")
        .str.upper()
        .str.strip()
    )

for col in [
    "PA_EST_AMT_LC",
    "PA_APPR_AMT_LC",
    "PA_REJ_AMT_LC",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)


# =========================================================
# CREATE REJECTION FLAG
# =========================================================
df["REJECTION_FLAG"] = (
    (df["PA_REJ_AMT_LC"] > 0) |
    (
        (df["PA_EST_AMT_LC"] > 0) &
        (df["PA_APPR_AMT_LC"] < df["PA_EST_AMT_LC"])
    )
).astype(int)


# =========================================================
# KEEP ONLY REJECTED CLAIMS
# =========================================================
rej_df = df[df["REJECTION_FLAG"] == 1].copy()

print("Rejected claims:", len(rej_df))


# =========================================================
# BUILD TRANSACTIONS
# =========================================================
feature_cols = [
    "PA_PRIMARY_DIAG",
    "CPT_CODE",
    "INS_TREAT_CODE",
    "PROV_TREAT_CODE",
    "PAT_DRUG_NAME",
    "DRUG_DURATION",
    "DOSAGE_DESC",
    "PROV_NAME",
    "DOC_NAME",
    "SERVICE_TYPE",
    "PA_CATG",
]

transactions = []

for _, row in rej_df.iterrows():

    items = []

    for col in feature_cols:

        value = row[col]

        if value not in [
            "",
            "UNKNOWN",
            "NAN",
            "NONE",
        ]:
            items.append(f"{col}={value}")

    transactions.append(items)


# =========================================================
# ONE HOT ENCODE
# =========================================================
te = TransactionEncoder()

te_array = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    te_array,
    columns=te.columns_
)


# =========================================================
# FP GROWTH
# =========================================================
frequent_patterns = fpgrowth(
    basket,
    min_support=0.01,
    use_colnames=True
)

print("Frequent patterns found:", len(frequent_patterns))


# =========================================================
# ASSOCIATION RULES
# =========================================================
rules = association_rules(
    frequent_patterns,
    metric="confidence",
    min_threshold=0.6
)

if len(rules) == 0:
    print("No strong rules found")
    exit()


# =========================================================
# FORMAT
# =========================================================
rules["antecedents"] = rules["antecedents"].apply(
    lambda x: ", ".join(list(x))
)

rules["consequents"] = rules["consequents"].apply(
    lambda x: ", ".join(list(x))
)

rules = rules.sort_values(
    by=["confidence", "lift"],
    ascending=False
)


# =========================================================
# FIND HISTORICAL METRICS
# =========================================================
output_rows = []

for _, rule in rules.iterrows():

    pattern_items = rule["antecedents"].split(", ")

    temp = rej_df.copy()

    valid = True

    for item in pattern_items:

        if "=" not in item:
            continue

        col, val = item.split("=", 1)

        if col not in temp.columns:
            valid = False
            break

        temp = temp[temp[col] == val]

    if not valid or len(temp) == 0:
        continue

    total_claims = len(temp)

    median_rej_amt = temp["PA_REJ_AMT_LC"].median()

    median_rej_pct = (
        (
            temp["PA_REJ_AMT_LC"] /
            temp["PA_EST_AMT_LC"].replace(0, np.nan)
        ) * 100
    ).median()

    top_rej_code = (
        temp["REJ_CODE"]
        .value_counts()
        .index[0]
        if len(temp["REJ_CODE"].value_counts()) > 0
        else ""
    )

    top_rej_desc = (
        temp["REJ_DESC"]
        .value_counts()
        .index[0]
        if len(temp["REJ_DESC"].value_counts()) > 0
        else ""
    )

    output_rows.append({

        "PATTERN": rule["antecedents"],

        "SUPPORT": round(rule["support"], 4),

        "CONFIDENCE": round(rule["confidence"], 4),

        "LIFT": round(rule["lift"], 4),

        "TOTAL_REJECTED_CLAIMS": total_claims,

        "MEDIAN_REJECTED_AMOUNT": round(
            median_rej_amt,
            2
        ),

        "MEDIAN_REJECTION_PERCENT": round(
            median_rej_pct,
            2
        ),

        "TOP_REJ_CODE": top_rej_code,

        "TOP_REJ_DESC": top_rej_desc,
    })


# =========================================================
# FINAL OUTPUT
# =========================================================
output_df = pd.DataFrame(output_rows)

output_df = output_df.drop_duplicates()

output_df = output_df.sort_values(
    by=[
        "CONFIDENCE",
        "TOTAL_REJECTED_CLAIMS",
    ],
    ascending=False
)

output_df.to_excel(
    "rejection_pattern_analysis.xlsx",
    index=False
)

print("\nSaved:")
print("rejection_pattern_analysis.xlsx")

print("\nTop Patterns:")
print(output_df.head(20))