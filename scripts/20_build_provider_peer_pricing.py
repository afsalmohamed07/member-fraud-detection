import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from src.config_loader import ConfigLoader


def safe(x):
    if pd.isna(x):
        return "UNKNOWN"

    value = str(x).strip().upper()

    if value in ["", "NAN", "NONE", "NULL", "NA", "N/A"]:
        return "UNKNOWN"

    return value


def remove_outliers(g, amount_col="PA_EST_AMT_LC"):
    g = g.copy()

    g = g[
        pd.to_numeric(
            g[amount_col],
            errors="coerce"
        ).fillna(0) > 0
    ]

    if len(g) < 1:
        return g

    q1 = g[amount_col].quantile(0.25)
    q3 = g[amount_col].quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        return g

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return g[
        (g[amount_col] >= lower)
        &
        (g[amount_col] <= upper)
    ]


def build_context_artifact(df, group_cols, context_type):
    artifact = {}

    grouped = df.groupby(
        group_cols,
        dropna=False
    )

    for key, g in grouped:
        g = remove_outliers(g)

        if len(g) < 1:
            continue

        provider_prices = {}

        for provider, pg in g.groupby(
            "PROV_NAME",
            dropna=False
        ):
            prices = (
                pg["PA_EST_AMT_LC"]
                .dropna()
                .astype(float)
                .tolist()
            )

            if len(prices) > 0:
                provider_prices[provider] = prices

        all_prices = (
            g["PA_EST_AMT_LC"]
            .dropna()
            .astype(float)
            .tolist()
        )

        if len(all_prices) == 0:
            continue

        if not isinstance(key, tuple):
            key = (key,)

        artifact[(context_type,) + key] = {
            "provider_prices": provider_prices,
            "all_prices": all_prices,
            "support": int(len(g)),
            "provider_count": int(len(provider_prices)),
        }

    return artifact


def main():
    print("\n======================================")
    print("BUILD PROVIDER PEER PRICING")
    print("======================================")

    config = ConfigLoader().get_config()

    df = joblib.load(
        config["paths"]["processed_data"]
    )

    required_cols = [
        "SERVICE_TYPE_NORM",
        "PA_CATG_NORM",
        "PA_PRIMARY_DIAG",
        "PROV_TREAT_CODE_CLEAN",
        "PAT_DRUG_NAME",
        "PROV_NAME",
        "PA_EST_AMT_LC",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = "UNKNOWN"

    text_cols = [
        "SERVICE_TYPE_NORM",
        "PA_CATG_NORM",
        "PA_PRIMARY_DIAG",
        "PROV_TREAT_CODE_CLEAN",
        "PAT_DRUG_NAME",
        "PROV_NAME",
    ]

    for col in text_cols:
        df[col] = df[col].apply(safe)
        
        df["SERVICE_TYPE_NORM"] = df["SERVICE_TYPE_NORM"].replace({
        "OUT-PATIENT": "OP",
        "OUTPATIENT": "OP",
        "OUT PATIENT": "OP",
        "IN-PATIENT": "IP",
        "INPATIENT": "IP",
        "IN PATIENT": "IP",
    })

    df["PA_EST_AMT_LC"] = pd.to_numeric(
        df["PA_EST_AMT_LC"],
        errors="coerce"
    ).fillna(0)

    df = df[
        df["PA_EST_AMT_LC"] > 0
    ].copy()

    pharmacy_df = df[
        df["PA_CATG_NORM"] == "PHARMACY"
    ].copy()

    non_pharmacy_df = df[
        df["PA_CATG_NORM"] != "PHARMACY"
    ].copy()

    artifact = {}

    print("\nBuilding pharmacy peer pricing...")

    pharmacy_artifact = build_context_artifact(
        pharmacy_df,
        group_cols=[
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            "PAT_DRUG_NAME",
        ],
        context_type="PHARMACY"
    )

    artifact.update(pharmacy_artifact)

    print("Pharmacy contexts:", len(pharmacy_artifact))

    print("\nBuilding non-pharmacy peer pricing...")

    non_pharmacy_artifact = build_context_artifact(
        non_pharmacy_df,
        group_cols=[
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            "PA_PRIMARY_DIAG",
            "PROV_TREAT_CODE_CLEAN",
        ],
        context_type="NON_PHARMACY"
    )

    artifact.update(non_pharmacy_artifact)

    print("Non-pharmacy contexts:", len(non_pharmacy_artifact))

    output_dir = Path(
        "artifacts/intelligence"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "provider_peer_pricing.pkl"
    )

    joblib.dump(
        artifact,
        output_path
    )

    print("\n======================================")
    print("PROVIDER PEER PRICING COMPLETED")
    print("======================================")
    print("Total contexts:", len(artifact))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()