import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.prediction_pipeline import predict_claim


st.set_page_config(
    page_title="Healthcare Claim Intelligence",
    layout="wide"
)


def safe(value, default=""):
    if value is None:
        return default
    return value


def flag_text(value):
    try:
        return "YES" if int(value) == 1 else "NO"
    except Exception:
        return str(value)


def safe_round(value, digits=2, default="NA"):
    try:
        if value is None or pd.isna(value):
            return default
        return round(float(value), digits)
    except Exception:
        return default


def make_display_table(data, key_name="Parameter"):
    if data is None:
        data = {}

    if isinstance(data, str):
        data = {"Message": data}

    if isinstance(data, list):
        data = {f"Item {i+1}": v for i, v in enumerate(data)}

    if not isinstance(data, dict):
        data = {"Value": data}

    rows = []
    for key, value in data.items():
        if value is None:
            value = ""
        rows.append({key_name: str(key), "Value": str(value)})

    return pd.DataFrame(rows)


def card(title, value, note=""):
    st.markdown(
        f"""
        <div style="
            border:1px solid #e5e7eb;
            border-radius:14px;
            padding:16px;
            background-color:#ffffff;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);
            min-height:110px;
        ">
            <div style="font-size:14px;color:#6b7280;margin-bottom:8px;">{title}</div>
            <div style="font-size:24px;font-weight:700;color:#111827;">{value}</div>
            <div style="font-size:13px;color:#6b7280;margin-top:8px;">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def save_feedback(input_row, result, decision, comment):
    feedback_dir = Path("data/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = feedback_dir / "feedback.csv"

    row = {
        "FEEDBACK_DT": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "AUDITOR_DECISION": decision,
        "AUDITOR_COMMENT": comment,
        "SERVICE_DT": input_row.get("SERVICE_DT"),
        "PA_CATG": input_row.get("PA_CATG"),
        "PA_PRIMARY_DIAG": input_row.get("PA_PRIMARY_DIAG"),
        "PROV_TREAT_CODE": input_row.get("PROV_TREAT_CODE"),
        "INS_TREAT_CODE": input_row.get("INS_TREAT_CODE"),
        "CPT_CODE": input_row.get("CPT_CODE"),
        "PAT_DRUG_NAME": input_row.get("PAT_DRUG_NAME"),
        "PROV_NAME": input_row.get("PROV_NAME"),
        "DOC_NAME": input_row.get("DOC_NAME"),
        "PA_EST_AMT_LC": input_row.get("PA_EST_AMT_LC"),
        "FINAL_LABEL": result.get("FINAL_LABEL"),
        "FINAL_RISK_SCORE": result.get("FINAL_RISK_SCORE"),
        "FINAL_CONFIDENCE": result.get("FINAL_CONFIDENCE"),
        "SHORT_EXPLANATION": result.get("SHORT_EXPLANATION"),
        "DETAILED_EXPLANATION": result.get("DETAILED_EXPLANATION"),
        "COST_ANOMALY_FINAL": result.get("COST_ANOMALY_FINAL"),
        "SIMILAR_CLAIMS_ANOMALY": result.get("SIMILAR_CLAIMS_ANOMALY"),
        "UPCODING_ANOMALY": result.get("UPCODING_ANOMALY"),
        "ISOLATION_ANOMALY": result.get("ISOLATION_ANOMALY"),
    }

    feedback_df = pd.DataFrame([row])

    if feedback_path.exists():
        old_df = pd.read_csv(feedback_path)
        feedback_df = pd.concat([old_df, feedback_df], ignore_index=True)

    feedback_df.to_csv(feedback_path, index=False)
    return feedback_path


st.title("Healthcare Claim Intelligence")
st.caption(
    "Cost prediction, anomaly signals, similar claims, provider intelligence, and auditor feedback"
)

st.sidebar.header("Claim Input")

SERVICE_TYPE = st.sidebar.selectbox("SERVICE_TYPE", ["Out-Patient", "In-Patient"])
SERVICE_DT = st.sidebar.text_input("SERVICE_DT", "26-Jan-2026")

PA_CATG = st.sidebar.text_input("PA_CATG", "Pharmacy")
PA_PRIMARY_DIAG = st.sidebar.text_input("PA_PRIMARY_DIAG", "K02.62")
PROV_TREAT_CODE = st.sidebar.text_input("PROV_TREAT_CODE", "DNT03017")

INS_TREAT_CODE = st.sidebar.text_input("INS_TREAT_CODE", "5257")
CPT_CODE = st.sidebar.text_input("CPT_CODE", "D2330")

PAT_DRUG_NAME = st.sidebar.text_input("PAT_DRUG_NAME", "")
DRUG_SCIENTIFIC_NAME = st.sidebar.text_input("DRUG_SCIENTIFIC_NAME", "")
DRUG_DURATION = st.sidebar.text_input("DRUG_DURATION", "")
DOSAGE_DESC = st.sidebar.text_input("DOSAGE_DESC", "")

PA_QTY = st.sidebar.number_input("PA_QTY", value=1.0)
PA_EST_AMT_LC = st.sidebar.number_input("PA_EST_AMT_LC", value=30.0)
PA_APPR_AMT_LC = st.sidebar.number_input("PA_APPR_AMT_LC", value=0.0)
PA_REJ_AMT_LC = st.sidebar.number_input("PA_REJ_AMT_LC", value=0.0)

PROV_NAME = st.sidebar.text_input("PROV_NAME", "Al Ahli Hospital")
DOC_NAME = st.sidebar.text_input("DOC_NAME", "Dr. Rabab Jaber Maziad")
FACILITIES = st.sidebar.text_input("FACILITIES", "Dentistry")
DOCTOR_LICENSE = st.sidebar.text_input("DOCTOR_LICENSE", "D195")

input_row = {
    "SERVICE_TYPE": SERVICE_TYPE,
    "SERVICE_DT": SERVICE_DT,
    "PA_CATG": PA_CATG,
    "PA_PRIMARY_DIAG": PA_PRIMARY_DIAG,
    "PROV_TREAT_CODE": PROV_TREAT_CODE,
    "INS_TREAT_CODE": INS_TREAT_CODE,
    "CPT_CODE": CPT_CODE,
    "PAT_DRUG_NAME": PAT_DRUG_NAME,
    "DRUG_SCIENTIFIC_NAME": DRUG_SCIENTIFIC_NAME,
    "DRUG_DURATION": DRUG_DURATION,
    "DOSAGE_DESC": DOSAGE_DESC,
    "PA_QTY": PA_QTY,
    "PA_EST_AMT_LC": PA_EST_AMT_LC,
    "PA_APPR_AMT_LC": PA_APPR_AMT_LC,
    "PA_REJ_AMT_LC": PA_REJ_AMT_LC,
    "PROV_NAME": PROV_NAME,
    "DOC_NAME": DOC_NAME,
    "FACILITIES": FACILITIES,
    "DOCTOR_LICENSE": DOCTOR_LICENSE,
}


with st.expander("Input Claim", expanded=False):
    st.dataframe(make_display_table(input_row), width="stretch")


if st.button("Predict Claim Risk", type="primary"):

    with st.spinner("Running claim intelligence engine..."):
        result = predict_claim(input_row)

    st.subheader("API Output")
    st.json(result)
    st.stop()
    final_score = safe(result.get("FINAL_RISK_SCORE"), 0)
    final_confidence = safe(result.get("FINAL_CONFIDENCE"), "NA")
    cost_risk = safe(result.get("COST_RISK"), "UNKNOWN")

    section_title(
        "Final Decision",
        "Overall decision from historical evidence, quantile cost prediction, LightGBM classifier, similar claims, provider intelligence, and isolation signals."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        card("Final Label", final_label)

    with col2:
        card("Final Risk Score", final_score)

    with col3:
        card("LightGBM Confidence", f"{final_confidence}%")

    with col4:
        card("Cost Risk", cost_risk)

    st.divider()

    # =====================================================
    # COST INTELLIGENCE - CASE1A
    # =====================================================
    section_title(
        "Cost Intelligence",
        "Case1A: Global history + provider history + CatBoost quantile prediction + LightGBM + Isolation Forest."
    )

    actual_cost = safe_round(result.get("PA_EST_AMT_LC"))

    p50 = safe_round(result.get("P50_COST") or result.get("p50_cost"))
    p75 = safe_round(result.get("P75_COST") or result.get("p75_cost"))
    p90 = safe_round(result.get("P90_COST") or result.get("p90_cost"))

    final_expected = safe_round(
        result.get("FINAL_EXPECTED_COST_ENGINE")
        or result.get("FINAL_EXPECTED_COST")
    )

    trusted_evidence = safe(result.get("TRUSTED_EVIDENCE"), "UNKNOWN")
    final_reason = safe(result.get("FINAL_ENGINE_REASON"), "")
    ratio = safe_round(result.get("FINAL_COST_RATIO"))

    global_support = safe(result.get("GLOBAL_SUPPORT"), 0)
    global_median = safe_round(result.get("GLOBAL_MEDIAN"))
    global_mode = safe_round(result.get("GLOBAL_MODE_COST"))
    global_mode_count = safe(result.get("GLOBAL_MODE_COUNT"), 0)
    global_p75 = safe_round(result.get("GLOBAL_P75"))
    global_p90 = safe_round(result.get("GLOBAL_P90"))
    global_common = result.get("GLOBAL_COMMON_CHARGES", [])

    provider_support = safe(result.get("PROVIDER_SUPPORT"), 0)
    provider_median = safe_round(result.get("PROVIDER_MEDIAN"))
    provider_mode = safe_round(result.get("PROVIDER_MODE_COST"))
    provider_mode_count = safe(result.get("PROVIDER_MODE_COUNT"), 0)
    provider_p75 = safe_round(result.get("PROVIDER_P75"))
    provider_p90 = safe_round(result.get("PROVIDER_P90"))
    provider_common = result.get("PROVIDER_COMMON_CHARGES", [])

    ml_label = safe(result.get("FINAL_LABEL"))
    ml_conf = safe(result.get("FINAL_CONFIDENCE"))
    iso_flag = flag_text(result.get("ISOLATION_ANOMALY"))
    iso_score = safe_round(result.get("ISOLATION_SCORE"))

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card("Actual Cost", actual_cost)

    with c2:
        card("Final Expected Cost", final_expected)

    with c3:
        card("Trusted Evidence", trusted_evidence)

    with c4:
        card("Cost Ratio", ratio)

    st.markdown("#### Global Historical Intelligence")

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        card("Global Median", global_median)

    with g2:
        card(
            "Most Repeated Global Charge",
            global_mode,
            f"Charged {global_mode_count} times"
        )

    with g3:
        card(
            "Global Historical Range",
            f"{global_p75} - {global_p90}",
            "P75 to P90"
        )

    with g4:
        card("Global Support", global_support)

    if isinstance(global_common, list) and len(global_common) > 0:
        st.markdown("##### Common Global Charges")
        for item in global_common:
            if isinstance(item, dict):
                charge = item.get("charge")
                count = item.get("count")
                st.markdown(f"- **{safe_round(charge)}** charged **{count} times**")
            else:
                st.markdown(f"- {item}")

    st.markdown("#### Provider Historical Intelligence")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        card("Provider Median", provider_median)

    with p2:
        card(
            "Most Repeated Provider Charge",
            provider_mode,
            f"Charged {provider_mode_count} times"
        )

    with p3:
        card(
            "Provider Historical Range",
            f"{provider_p75} - {provider_p90}",
            "P75 to P90"
        )

    with p4:
        card("Provider Support", provider_support)

    if isinstance(provider_common, list) and len(provider_common) > 0:
        st.markdown("##### Common Provider Charges")
        for item in provider_common:
            if isinstance(item, dict):
                charge = item.get("charge")
                count = item.get("count")
                st.markdown(f"- **{safe_round(charge)}** charged **{count} times**")
            else:
                st.markdown(f"- {item}")

    st.markdown("#### CatBoost Quantile Prediction")

    q1, q2, q3 = st.columns(3)

    with q1:
        card("P50 Expected Cost", p50, "Median ML estimate")

    with q2:
        card("P75 Upper Normal", p75, "Upper-normal ML estimate")

    with q3:
        card("P90 High-Risk Limit", p90, "ML high-cost threshold")

    st.markdown("#### Risk Models")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        card("LightGBM Label", ml_label)

    with r2:
        card("LightGBM Confidence", f"{ml_conf}%")

    with r3:
        card("Isolation Anomaly", iso_flag)

    with r4:
        card("Isolation Score", iso_score)

    st.markdown("#### Final Decision Engine")

    st.info(final_reason)

    if str(trusted_evidence).startswith("PROVIDER"):
        st.success(
            "Strong provider-level repeated history exists. Provider evidence dominates CatBoost prediction."
        )
    elif str(trusted_evidence).startswith("GLOBAL"):
        st.success(
            "Strong global historical evidence exists. Global history dominates CatBoost prediction."
        )
    else:
        st.warning(
            "Historical support is weak. CatBoost prediction is used as primary evidence."
        )

    st.divider()

    # =====================================================
    # SIMILAR CLAIMS INTELLIGENCE
    # =====================================================
    section_title(
        "Similar Claims Intelligence",
        "Compares the current claim with historically similar claims from the recent window."
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        card("Similar Claims", safe(result.get("SIMILAR_CLAIMS_COUNT"), 0))

    with s2:
        card("Rejected Similar Claims", safe(result.get("SIMILAR_REJECTED_COUNT"), 0))

    with s3:
        card("Rejection Rate", f"{safe(result.get('SIMILAR_REJECTION_RATE'), 0)}%")

    with s4:
        card("Similar Claims Anomaly", flag_text(result.get("SIMILAR_CLAIMS_ANOMALY")))

    similar_summary = {
        "Status": result.get("SIMILAR_CLAIMS_STATUS"),
        "Best Match Context": result.get("SIMILAR_CLAIMS_LEVEL_USED"),
        "Current Cost Above Similar P90": result.get("SIMILAR_COST_ABOVE_P90"),
        "Top Rejection Code": result.get("SIMILAR_TOP_REJ_CODE") or "Not available",
        "Top Rejection Description": result.get("SIMILAR_TOP_REJ_DESC") or "Not available",
    }

    st.dataframe(make_display_table(similar_summary), width="stretch")
    st.info(result.get("SIMILAR_CLAIMS_EXPLANATION", ""))

    with st.expander("Top Similar Contexts", expanded=False):
        top_contexts = result.get("SIMILAR_TOP_CONTEXTS", [])

        if isinstance(top_contexts, list) and len(top_contexts) > 0:
            try:
                st.dataframe(pd.DataFrame(top_contexts), width="stretch")
            except Exception:
                st.write(top_contexts)
        else:
            st.info("No similar contexts found")

    st.divider()

    # =====================================================
    # PROVIDER TREATMENT PATTERN ANALYSIS
    # =====================================================
    section_title(
        "Provider Treatment Pattern Analysis",
        "Checks whether provider treatment usage differs from diagnosis-level history."
    )

    u1, u2, u3, u4 = st.columns(4)

    with u1:
        card("Possible Upcoding Behavior", flag_text(result.get("UPCODING_ANOMALY")))

    with u2:
        card("Confidence", safe(result.get("UPCODING_CONFIDENCE"), "UNKNOWN"))

    with u3:
        card("Usage Ratio", safe(result.get("USAGE_RATIO"), "Not available"))

    with u4:
        card("Support Status", safe(result.get("UPCODING_SUPPORT_STATUS"), "UNKNOWN"))

    upcoding_summary = {
        "Global Diagnosis Claims": result.get("GLOBAL_DIAG_TOTAL"),
        "Global Diagnosis + Treatment Claims": result.get("GLOBAL_DIAG_TREAT_COUNT"),
        "Global Treatment Usage %": result.get("GLOBAL_TREAT_USAGE_PCT"),
        "Provider Diagnosis Claims": result.get("PROVIDER_DIAG_TOTAL"),
        "Provider Diagnosis + Treatment Claims": result.get("PROVIDER_DIAG_TREAT_COUNT"),
        "Provider Treatment Usage %": result.get("PROVIDER_TREAT_USAGE_PCT"),
        "Diagnosis P75 Cost": result.get("DIAG_P75_COST"),
        "Diagnosis P90 Cost": result.get("DIAG_P90_COST"),
        "Treatment P75 Cost": result.get("TREATMENT_P75_COST"),
        "Treatment P90 Cost": result.get("TREATMENT_P90_COST"),
        "Current Cost Above Treatment P90": result.get("CURRENT_COST_ABOVE_TREATMENT_P90"),
        "Treatment High Cost Pattern": result.get("TREATMENT_HIGH_COST_PATTERN"),
    }

    st.dataframe(make_display_table(upcoding_summary), width="stretch")
    st.info(result.get("UPCODING_EXPLANATION", ""))

    st.divider()

    # =====================================================
    # KEY RISK SIGNALS
    # =====================================================
    section_title(
        "Key Risk Signals",
        "All important anomaly flags are shown in one place."
    )

    signal_data = {
        "Cost Anomaly": result.get("COST_ANOMALY_FINAL"),
        "Unknown Diagnosis": result.get("UNKNOWN_DIAGNOSIS_ANOMALY"),
        "Unseen Treatment": result.get("UNSEEN_TREATMENT_CODE_ANOMALY"),
        "Rare Treatment": result.get("RARE_TREATMENT_ANOMALY"),
        "Pharmacy Missing Drug": result.get("PHARMACY_MISSING_DRUG_ANOMALY"),
        "Drug Under Non-Pharmacy": result.get("NON_PHARMACY_WITH_DRUG_ANOMALY"),
        "Drug Not In Master": result.get("DRUG_NOT_IN_MASTER_ANOMALY"),
        "Drug Not Used For Diagnosis": result.get("DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY"),
        "Unseen Drug": result.get("UNSEEN_DRUG_ANOMALY"),
        "Rare Drug": result.get("RARE_DRUG_ANOMALY"),
        "Duration Outside Range": result.get("DURATION_OUTSIDE_RANGE_ANOMALY"),
        "Extreme Duration": result.get("EXTREME_DURATION_ANOMALY"),
        "Dosage Outside Range": result.get("DOSAGE_OUTSIDE_RANGE_ANOMALY"),
        "License Anomaly": result.get("LICENSE_ANOMALY"),
        "Doctor Facility Anomaly": result.get("DOCTOR_FACILITY_ANOMALY"),
        "Doctor Treatment Anomaly": result.get("DOCTOR_TREATMENT_ANOMALY"),
        "Doctor Drug Anomaly": result.get("DOCTOR_DRUG_ANOMALY"),
        "Provider Behavior Anomaly": result.get("PROVIDER_BEHAVIOR_ANOMALY"),
        "Pharmacy High Cost": result.get("PHARMACY_HIGH_COST_ANOMALY"),
        "Similar Claims Anomaly": result.get("SIMILAR_CLAIMS_ANOMALY"),
        "Provider Treatment Pattern Deviation": result.get("UPCODING_ANOMALY"),
        "Isolation Anomaly": result.get("ISOLATION_ANOMALY"),
        "Isolation Context Anomaly": result.get("ISOLATION_CONTEXT_ANOMALY"),
    }

    st.dataframe(make_display_table(signal_data, key_name="Signal"), width="stretch")

    st.divider()

    # =====================================================
    # PROVIDER-LEVEL INTELLIGENCE
    # =====================================================
    section_title(
        "Provider-Level Intelligence",
        "Evidence generated from the current claim context and recent provider history."
    )

    provider_findings = result.get("PROVIDER_FINDINGS", [])
    provider_evidence = result.get("PROVIDER_EVIDENCE", {})

    if isinstance(provider_findings, str):
        provider_findings = [provider_findings]

    if isinstance(provider_findings, list) and len(provider_findings) > 0:
        for finding in provider_findings:
            st.info(str(finding))
    else:
        st.info("No provider-level findings available.")

    with st.expander("Provider Evidence Snapshot", expanded=True):
        st.dataframe(make_display_table(provider_evidence), width="stretch")

    st.markdown("#### Provider-Doctor Pattern")

    provider_doctor_finding = result.get(
        "PROVIDER_DOCTOR_FINDING",
        "No provider-doctor review pattern observed for this claim context."
    )

    st.info(provider_doctor_finding)

    with st.expander("Provider-Doctor Evidence", expanded=False):
        st.dataframe(
            make_display_table(result.get("PROVIDER_DOCTOR_EVIDENCE", {})),
            width="stretch"
        )

    st.divider()

    # =====================================================
    # EXPLANATION
    # =====================================================
    section_title("Explanation", "Human-readable reasoning for the decision.")

    st.write("Short Explanation")
    st.info(result.get("SHORT_EXPLANATION", ""))

    st.write("Detailed Explanation")
    st.warning(result.get("DETAILED_EXPLANATION", ""))

    st.divider()

    # =====================================================
    # AUDITOR FEEDBACK
    # =====================================================
    section_title(
        "Auditor Feedback",
        "Save human review decision for future rule tuning and model improvement."
    )

    auditor_decision = st.selectbox(
        "Auditor Decision",
        ["NEEDS_REVIEW", "TRUE_FRAUD", "FALSE_POSITIVE", "VALID_CLAIM"]
    )

    auditor_comment = st.text_area(
        "Auditor Comment",
        placeholder="Enter review notes..."
    )

    if st.button("Save Feedback"):
        feedback_path = save_feedback(
            input_row=input_row,
            result=result,
            decision=auditor_decision,
            comment=auditor_comment,
        )

        st.success(f"Feedback saved successfully: {feedback_path}")

    with st.expander("Raw Result JSON", expanded=False):
        st.json(result)