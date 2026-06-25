import pandas as pd
import joblib
from pathlib import Path

from src.config_loader import ConfigLoader
from src.preprocessing import ClaimsPreprocessor
from src.baseline_scoring import BaselineScorer
from src.risk_scoring import RiskScorer
from src.explanation_builder import ExplanationBuilder
from src.cost_predictor import CostPredictor
from src.isolation_model import IsolationModel


def cost_risk_check(actual_cost, final_expected, p75, p90):
    if final_expected is None or pd.isna(final_expected):
        return "UNKNOWN", 0, "No reliable expected cost available"

    if p75 is None or pd.isna(p75):
        return "UNKNOWN", 0, "No P75 baseline available"

    if p90 is None or pd.isna(p90):
        return "UNKNOWN", 0, "No P90 baseline available"

    actual_cost = float(actual_cost)
    final_expected = float(final_expected)
    p75 = float(p75)
    p90 = float(p90)

    ratio = actual_cost / final_expected if final_expected > 0 else 1

    if actual_cost <= p75:
        return "NORMAL", 0, "Within normal range (<= P75)"

    if actual_cost <= p90:
        if ratio < 1.5:
            return "NORMAL", 0, "Slightly above P75 but acceptable"

        return "WARNING", 0, "Above P75 baseline range"

    deviation_pct = ((actual_cost - p90) / p90) * 100 if p90 > 0 else 0

    if deviation_pct < 15:
        return "WARNING", 0, "Slightly above P90 baseline"

    if ratio < 3:
        return "MEDIUM", 1, "Moderately above historical baseline"

    return "HIGH", 1, "Extremely above historical baseline"


def safe_round(value, digits=2):
    if value is None or pd.isna(value):
        return None

    return round(float(value), digits)


def restore_raw_columns(df_source, df_target, columns):
    df_target = df_target.copy()

    for col in columns:
        if col in df_source.columns:
            df_target[col] = df_source[col].values

    return df_target


def main():
    print("\n======================================")
    print("SINGLE ROW PREDICTION")
    print("======================================")

    config = ConfigLoader().get_config()

    show_debug_evidence = (
        config.get("output_settings", {})
        .get("show_debug_evidence", True)
    )

    input_row = {
        "SERVICE_TYPE": "Out-Patient",
        "PA_CATG": "Pharmacy",
        "PA_PRIMARY_DIAG": "K02.62",
        "PROV_TREAT_CODE": "DNT03017",

        "PAT_DRUG_NAME": "AGIOLAX GRANULES - JR250",
        "DRUG_SCIENTIFIC_NAME": "",
        "DRUG_DURATION": "",
        "DOSAGE_DESC": "",

        "PA_QTY": 1,
        "PA_EST_AMT_LC": 30,
        "PA_APPR_AMT_LC": 0,
        "PA_REJ_AMT_LC": 0,

        "PROV_NAME": "Al Ahli Hospital",
        "DOC_NAME": "Dr. Rabab Jaber Maziad",
        "FACILITIES": "General",

        "DOCTOR_LICENSE": "D195",
    }

    df = pd.DataFrame([input_row])

    print("\nInput Row:")
    print(df.T)

    # =====================================================
    # PREPROCESS
    # =====================================================
    preprocessor = ClaimsPreprocessor(config)
    df_processed = preprocessor.preprocess(df)

    raw_cols = [
        "DOCTOR_LICENSE",
        "DOC_NAME",
        "PROV_NAME",
        "FACILITIES",
    ]

    df_processed = restore_raw_columns(
        df_source=df,
        df_target=df_processed,
        columns=raw_cols
    )

    row = df_processed.iloc[0]

    # =====================================================
    # COST PREDICTION
    # =====================================================
    cost_predictor = CostPredictor(config)
    cost_pred = cost_predictor.predict_cost(row, debug=True)

    actual_cost = row["PA_EST_AMT_LC"]

    cost_risk, cost_anomaly, cost_explanation = cost_risk_check(
        actual_cost=actual_cost,
        final_expected=cost_pred["final_expected_cost"],
        p75=cost_pred["pattern_p75"],
        p90=cost_pred["pattern_p90"],
    )

    # =====================================================
    # BASELINE / RULE SCORING
    # =====================================================
    baseline_scorer = BaselineScorer(config)
    df_scored = baseline_scorer.score_all(df_processed)

    df_scored = restore_raw_columns(
        df_source=df,
        df_target=df_scored,
        columns=raw_cols
    )

    print("\n================ LICENSE DEBUG ================")
    print("DOCTOR_LICENSE in df_processed:", "DOCTOR_LICENSE" in df_processed.columns)
    print("DOCTOR_LICENSE in df_scored:", "DOCTOR_LICENSE" in df_scored.columns)

    if "DOCTOR_LICENSE" in df_scored.columns:
        print("Input license:", df_scored["DOCTOR_LICENSE"].iloc[0])

    license_cols = [
        "DOCTOR_NOT_IN_LICENSE_BASELINE",
        "MULTIPLE_LICENSE_ANOMALY",
        "MISSING_LICENSE_ANOMALY",
        "UNSEEN_LICENSE_ANOMALY",
        "LICENSE_ANOMALY",
        "LICENSE_EXPLANATION",
    ]

    for c in license_cols:
        if c in df_scored.columns:
            print(c, "=", df_scored[c].iloc[0])
        else:
            print(c, "= COLUMN NOT FOUND")

    # =====================================================
    # ISOLATION FOREST SCORING
    # =====================================================
    isolation = IsolationModel(config)
    isolation.load()

    df_scored = isolation.predict(df_scored)

    df_scored = restore_raw_columns(
        df_source=df,
        df_target=df_scored,
        columns=raw_cols
    )

    print("\n================ ISOLATION INPUT DEBUG ================")
    print("DOC_NAME in df_scored:", "DOC_NAME" in df_scored.columns)
    print("PROV_NAME in df_scored:", "PROV_NAME" in df_scored.columns)
    print("FACILITIES in df_scored:", "FACILITIES" in df_scored.columns)
    print("DOCTOR_LICENSE in df_scored:", "DOCTOR_LICENSE" in df_scored.columns)

    try:
        df_scored = isolation.predict(df_scored)
    except Exception as e:
        print("\nIsolation skipped due to error:")
        print(str(e))

        df_scored["ISOLATION_ANOMALY"] = 0
        df_scored["ISOLATION_SCORE"] = 0
        df_scored["ISOLATION_PRED"] = 1

    # Single row isolation not reliable, so disable it
    #if len(df_scored) == 1:
       # df_scored["ISOLATION_ANOMALY"] = 0
       # df_scored["ISOLATION_SCORE"] = 0
       # df_scored["ISOLATION_PRED"] = 1

    # =====================================================
    # ADD COST PREDICTION DETAILS
    # =====================================================
    df_scored["BASELINE_EXPECTED_COST"] = cost_pred["baseline_expected_cost"]

    df_scored["ML_EXPECTED_COST_RAW"] = (
        safe_round(cost_pred.get("ml_expected_cost_raw"))
    )

    df_scored["ML_EXPECTED_COST"] = (
        safe_round(cost_pred.get("ml_expected_cost"))
    )

    df_scored["FINAL_EXPECTED_COST"] = (
        safe_round(cost_pred["final_expected_cost"])
    )

    df_scored["EXPECTED_SOURCE"] = cost_pred["source"]
    df_scored["COST_LEVEL_USED"] = cost_pred["level_used"]
    df_scored["COST_SUPPORT"] = cost_pred["support"]
    df_scored["COST_CONFIDENCE"] = cost_pred["confidence"]

    df_scored["PATTERN_MEAN"] = cost_pred["pattern_mean"]
    df_scored["PATTERN_MEDIAN"] = cost_pred["pattern_median"]
    df_scored["PATTERN_P75"] = cost_pred["pattern_p75"]
    df_scored["PATTERN_P90"] = cost_pred["pattern_p90"]
    df_scored["PATTERN_MAX"] = cost_pred["pattern_max"]

    df_scored["COST_RISK"] = cost_risk
    df_scored["COST_ANOMALY_FINAL"] = cost_anomaly
    df_scored["COST_EXPLANATION"] = cost_explanation

    df_scored["UNSEEN_DRUG_FOR_COST"] = cost_pred.get("unseen_drug", 0)

    if show_debug_evidence:
        df_scored["PATTERN_COST_LIST"] = str(
            cost_pred.get("pattern_cost_list", [])
        )
        df_scored["TOP_PRICE_SUMMARY"] = ", ".join(
            cost_pred.get("top_price_summary", [])
        )
    else:
        df_scored["PATTERN_COST_LIST"] = ""
        df_scored["TOP_PRICE_SUMMARY"] = ""

    # =====================================================
    # FINAL RISK SCORE
    # =====================================================
    risk_scorer = RiskScorer(config)
    df_scored = risk_scorer.score(df_scored)

    # =====================================================
    # EXPLANATION
    # =====================================================
    explanation_builder = ExplanationBuilder(config)
    df_scored = explanation_builder.build(df_scored)

    # =====================================================
    # PRINT COST RESULT
    # =====================================================
    print("\n======================================")
    print("COST PREDICTION RESULT")
    print("======================================")

    print("Actual Cost:", actual_cost)

    ml_raw = cost_pred.get("ml_expected_cost_raw")
    ml_final = cost_pred.get("ml_expected_cost")

    range_low = cost_pred.get("pattern_median")
    range_high = cost_pred.get("pattern_p90")

    print(
        "CatBoost Raw Expected:",
        safe_round(ml_raw)
    )

    print(
        "Allowed Range:",
        f"{safe_round(range_low)} - {safe_round(range_high)}"
        if safe_round(range_low) is not None
        and safe_round(range_high) is not None
        else None
    )

    print(
        "Final ML Expected:",
        safe_round(ml_final)
    )

    print(
        "Final Expected:",
        safe_round(cost_pred["final_expected_cost"])
    )

    print("Source:", cost_pred["source"])
    print("Level:", cost_pred["level_used"])
    print("Support:", cost_pred["support"])
    print("Confidence:", cost_pred["confidence"])

    print("\nPattern Range")
    print("Pattern Mean:", safe_round(cost_pred["pattern_mean"]))
    print("Pattern Median:", safe_round(cost_pred["pattern_median"]))
    print("Pattern P75:", safe_round(cost_pred["pattern_p75"]))
    print("Pattern P90:", safe_round(cost_pred["pattern_p90"]))
    print("Pattern Max:", safe_round(cost_pred["pattern_max"]))

    if show_debug_evidence:
        print("\nHistorical Evidence")

        if len(cost_pred.get("pattern_cost_list", [])) > 0:
            print("Matched Cost List:", cost_pred["pattern_cost_list"])

        if len(cost_pred.get("top_price_summary", [])) > 0:
            print(
                "Top Historical Prices:",
                ", ".join(cost_pred["top_price_summary"])
            )

    print("\nCost Risk:", cost_risk)
    print("Cost Anomaly:", cost_anomaly)
    print("Cost Explanation:", cost_explanation)

    # =====================================================
    # FINAL ANOMALY SUMMARY
    # =====================================================
    print("\n======================================")
    print("FINAL ANOMALY SUMMARY")
    print("======================================")

    row = df_scored.iloc[0]

    summary_outputs = [
        ("COST_RISK", "Cost Risk"),
        ("COST_ANOMALY_FINAL", "Cost Anomaly"),

        ("UNKNOWN_DIAGNOSIS_ANOMALY", "Unknown Diagnosis"),
        ("UNSEEN_TREATMENT_CODE_ANOMALY", "Unseen Treatment"),
        ("RARE_TREATMENT_ANOMALY", "Rare Treatment"),

        ("PHARMACY_MISSING_DRUG_ANOMALY", "Pharmacy Missing Drug"),
        ("NON_PHARMACY_WITH_DRUG_ANOMALY", "Drug Under Non Pharmacy"),

        ("DRUG_NOT_IN_MASTER_ANOMALY", "Drug Not In Master"),
        ("DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY", "Drug Not Used For Diagnosis"),
        ("UNSEEN_DRUG_ANOMALY", "Unseen Drug"),
        ("RARE_DRUG_ANOMALY", "Rare Drug"),

        ("DURATION_EXACT_SEEN", "Duration Exact Seen"),
        ("DURATION_UNSEEN_WITHIN_RANGE", "Duration Unseen Within Range"),
        ("DURATION_OUTSIDE_RANGE_ANOMALY", "Duration Outside Range"),
        ("EXTREME_DURATION_ANOMALY", "Extreme Duration"),

        ("DOSAGE_EXACT_SEEN", "Dosage Exact Seen"),
        ("DOSAGE_UNSEEN_WITHIN_RANGE", "Dosage Unseen Within Range"),
        ("DOSAGE_OUTSIDE_RANGE_ANOMALY", "Dosage Outside Range"),

        ("DOCTOR_NOT_IN_LICENSE_BASELINE", "Doctor Not In License Baseline"),
        ("MULTIPLE_LICENSE_ANOMALY", "Multiple License"),
        ("MISSING_LICENSE_ANOMALY", "Missing License"),
        ("UNSEEN_LICENSE_ANOMALY", "Unseen License"),
        ("LICENSE_ANOMALY", "License Anomaly"),

        ("DOCTOR_FACILITY_ANOMALY", "Doctor Facility Anomaly"),
        ("DOCTOR_TREATMENT_ANOMALY", "Doctor Treatment Anomaly"),
        ("DOCTOR_DRUG_ANOMALY", "Doctor Drug Anomaly"),

        ("PROVIDER_BEHAVIOR_ANOMALY", "Provider Behavior Anomaly"),
        ("PHARMACY_HIGH_COST_ANOMALY", "Pharmacy High Cost"),

        ("ISOLATION_ANOMALY", "Isolation Anomaly"),
        ("ISOLATION_SCORE", "Isolation Score"),
        ("ISOLATION_SCORE_RAW", "Isolation Raw Score"),
        ("ISOLATION_CONTEXT_SCORE", "Isolation Context Score"),
        ("ISOLATION_CONTEXT_ANOMALY", "Isolation Context Anomaly"),

        ("FINAL_RISK_SCORE", "Final Risk Score"),
        ("FINAL_LABEL", "Final Label"),

        ("SHORT_EXPLANATION", "Short Explanation"),
        ("DETAILED_EXPLANATION", "Detailed Explanation"),
    ]

    for col, label in summary_outputs:
        if col not in row.index:
            continue

        value = row[col]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value == "" or value.lower() == "nan":
            continue

        print(f"{label:<35}: {value}")

    # =====================================================
    # SAVE OUTPUT
    # =====================================================
    output_dir = Path("output/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "single_row_prediction.xlsx"
    df_scored.to_excel(output_file, index=False)

    print("\nSaved prediction:")
    print(output_file)


if __name__ == "__main__":
    main()