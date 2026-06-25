import re
import joblib
import pandas as pd
from pathlib import Path

from src.config_loader import ConfigLoader


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
    return re.sub(r"[^A-Z0-9]", "", clean_text(value))


def treatment_family(code):
    code = clean_code(code)

    if code == "":
        return "UNKNOWN"

    match = re.match(r"^[A-Z]+", code)

    if match:
        return match.group(0)

    return "NUMERIC"


def main():
    print("\n======================================")
    print("BUILDING TREATMENT INTELLIGENCE")
    print("======================================")

    config = ConfigLoader().get_config()
    df = joblib.load(config["paths"]["processed_data"])

    print("Processed shape:", df.shape)

    service_col = "SERVICE_TYPE_NORM" if "SERVICE_TYPE_NORM" in df.columns else "SERVICE_TYPE"
    category_col = "PA_CATG_NORM" if "PA_CATG_NORM" in df.columns else "PA_CATG"
    diag_col = config["columns"]["diagnosis"]

    for col in [service_col, category_col, diag_col]:
        if col not in df.columns:
            df[col] = ""

        df[col] = df[col].apply(clean_text)

    if "PROV_TREAT_CODE_CLEAN" not in df.columns:
        df["PROV_TREAT_CODE_CLEAN"] = df[config["columns"]["treatment_code"]].apply(clean_code)

    df["PROV_TREAT_CODE_CLEAN"] = df["PROV_TREAT_CODE_CLEAN"].apply(clean_code)
    df["TREATMENT_FAMILY_INTEL"] = df["PROV_TREAT_CODE_CLEAN"].apply(treatment_family)

    artifact = {}

    group_cols = [
        service_col,
        category_col,
        diag_col,
    ]

    grouped = df.groupby(group_cols, dropna=False)

    for key, g in grouped:
        total = len(g)

        if total == 0:
            continue

        code_counts = g["PROV_TREAT_CODE_CLEAN"].value_counts().to_dict()
        family_counts = g["TREATMENT_FAMILY_INTEL"].value_counts().to_dict()

        code_probs = {
            code: round(count / total, 6)
            for code, count in code_counts.items()
        }

        family_probs = {
            fam: round(count / total, 6)
            for fam, count in family_counts.items()
        }

        top_codes = [
            {
                "treatment_code": code,
                "count": int(count),
                "probability": round(count / total, 6),
            }
            for code, count in list(code_counts.items())[:10]
        ]

        top_families = [
            {
                "treatment_family": fam,
                "count": int(count),
                "probability": round(count / total, 6),
            }
            for fam, count in list(family_counts.items())[:10]
        ]

        artifact[key] = {
            "TOTAL_CONTEXT_CLAIMS": int(total),
            "CODE_COUNTS": code_counts,
            "CODE_PROBS": code_probs,
            "FAMILY_COUNTS": family_counts,
            "FAMILY_PROBS": family_probs,
            "TOP_TREATMENT_CODES": top_codes,
            "TOP_TREATMENT_FAMILIES": top_families,
            "SERVICE_COL_USED": service_col,
            "CATEGORY_COL_USED": category_col,
        }

    output_dir = Path(config["paths"]["models_dir"]).parent / "intelligence"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "treatment_intelligence_artifact.pkl"
    joblib.dump(artifact, output_path)

    print("Treatment intelligence keys:", len(artifact))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()