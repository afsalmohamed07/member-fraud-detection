import joblib
import pandas as pd
import numpy as np
from src.api_output_builder import APIOutputBuilder
from src.config_loader import ConfigLoader
from src.preprocessing import ClaimsPreprocessor
from src.baseline_scoring import BaselineScorer
from src.cost_predictor import CostPredictor
from src.provider_level_intelligence import ProviderLevelIntelligence
from src.upcoding_detector import UpcodingDetector
from src.similar_claims import SimilarClaimsIntelligence
from src.isolation_model import IsolationModel
from src.risk_classifier import RiskClassifier
from src.explanation_builder import ExplanationBuilder
from src.treatment_expectation import TreatmentExpectationIntelligence
from src.provider_fingerprint import ProviderFingerprintAnalyzer
from src.claim_pattern_autoencoder import ClaimPatternAutoencoder
from src.provider_peer_pricing import ProviderPeerPricing


config = ConfigLoader().get_config()
history_df = joblib.load(config["paths"]["processed_data"])
treatment_expectation = TreatmentExpectationIntelligence(config)
preprocessor = ClaimsPreprocessor(config)
baseline_scorer = BaselineScorer(config)
cost_predictor = CostPredictor(config)
api_output_builder = APIOutputBuilder(review_threshold=60)
provider_intelligence = ProviderLevelIntelligence(config)
upcoding_detector = UpcodingDetector(config)
similar_claims_engine = SimilarClaimsIntelligence(config)
isolation_model = IsolationModel(config)
risk_classifier = RiskClassifier(config)
explanation_builder = ExplanationBuilder(config)
provider_fingerprint = ProviderFingerprintAnalyzer(config)
claim_pattern_model = ClaimPatternAutoencoder(config)
provider_peer_pricing = ProviderPeerPricing(config)



def _safe_apply_result(df: pd.DataFrame, result):
    if not isinstance(result, dict):
        return df

    df = df.copy()

    for k, v in result.items():
        if isinstance(v, (list, dict)):
            df[k] = pd.Series([v], index=df.index, dtype="object")
        else:
            df[k] = v

    return df


def _get_row_value(row, *names, default=np.nan):
    for name in names:
        if name in row.index:
            value = row.get(name)

            if value is not None:
                return value

    return default


def _to_float(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def predict_claim(input_row: dict):
    df = pd.DataFrame([input_row])

    # =====================================================
    # PREPROCESS
    # =====================================================
    df = preprocessor.preprocess(df)

    # =====================================================
    # EXISTING BASELINE SCORING
    # =====================================================
    df = baseline_scorer.score_all(df)
    treatment_result = treatment_expectation.analyze(df.iloc[0])
    df = _safe_apply_result(df, treatment_result)
        # =====================================================
    # CASE1A COST PREDICTOR
    # returns:
    # GLOBAL history + PROVIDER history + CatBoost P50/P75/P90
    # + FINAL_EXPECTED_COST_ENGINE
    # =====================================================
    # =====================================================
    # PROVIDER PEER PRICING
    # Must run before CatBoost cost predictor
    # because CatBoost uses peer pricing features
    # =====================================================
    peer_pricing_result = provider_peer_pricing.analyze(
        df.iloc[0]
    )

    df = _safe_apply_result(
        df,
        peer_pricing_result
    )

    # =====================================================
    # COST PREDICTOR
    # Uses CatBoost + peer pricing context features
    # =====================================================
    cost_result = cost_predictor.predict_cost(
        df.iloc[0],
        debug=False
    )

    df = _safe_apply_result(
        df,
        cost_result
    )

    df = _safe_apply_result(df, cost_result)

    row = df.iloc[0]

    # =====================================================
    # FINAL EXPECTED COST
    # Priority:
    # 1. FINAL_EXPECTED_COST_ENGINE from cost_predictor
    # 2. final_expected_cost fallback
    # 3. ML expected fallback
    # 4. provider/global median fallback
    # =====================================================
    final_expected = _get_row_value(
        row,
        "FINAL_EXPECTED_COST_ENGINE",
        "final_expected_cost",
        "ML_EXPECTED_COST",
        "ml_expected_cost",
        "PROVIDER_MEDIAN",
        "GLOBAL_MEDIAN",
        "PATTERN_MEDIAN",
        "pattern_median",
        default=0
    )

    final_expected = _to_float(final_expected, default=0)

    df["FINAL_EXPECTED_COST"] = final_expected

    df["PA_EST_AMT_LC"] = pd.to_numeric(
        df["PA_EST_AMT_LC"],
        errors="coerce"
    ).fillna(0)

    df["FINAL_COST_RATIO"] = np.where(
        df["FINAL_EXPECTED_COST"] > 0,
        df["PA_EST_AMT_LC"] / df["FINAL_EXPECTED_COST"],
        1
    )

    # =====================================================
    # PROVIDER LEVEL INTELLIGENCE
    # =====================================================
    provider_result = provider_intelligence.analyze(df.iloc[0])
    df = _safe_apply_result(df, provider_result)
    # =====================================================
    # PROVIDER FINGERPRINT
    # =====================================================
    fingerprint_result = provider_fingerprint.analyze(df.iloc[0])
    df = _safe_apply_result(df, fingerprint_result)
    # =====================================================
    # UPCODING DETECTOR
    # =====================================================
    upcoding_result = upcoding_detector.detect(df.iloc[0])
    df = _safe_apply_result(df, upcoding_result)

    # =====================================================
    # SIMILAR CLAIMS
    # =====================================================
    similar_result = similar_claims_engine.analyze(df.iloc[0])
    df = _safe_apply_result(df, similar_result)

    # =====================================================
    # ISOLATION FOREST
    # =====================================================
    df = isolation_model.predict(df)
    # =====================================================
        # =====================================================
    # CLAIM PATTERN AUTOENCODER
    # =====================================================
    pattern_result = claim_pattern_model.predict(
        df.iloc[0]
    )

    df = _safe_apply_result(
        df,
        pattern_result
    )
    # =====================================================
    # LIGHTGBM FINAL CLASSIFIER
    # =====================================================
    # =====================================================
    # FINAL LABEL WILL BE GENERATED BY RULE SCORE / API BUILDER
    # LightGBM disabled because current labels are rule-generated,
    # not confirmed fraud labels.
    # =====================================================
    df["FINAL_LABEL"] = "RULE_BASED"
    df["FINAL_CONFIDENCE"] = 0

    # =====================================================
    # FINAL RISK SCORE DISPLAY
    # =====================================================
    df["FINAL_RISK_SCORE"] = 0

    # =====================================================
    # COST RISK DISPLAY
    # =====================================================
    df["COST_RISK"] = np.select(
        [
            df["FINAL_COST_RATIO"] >= 5,
            df["FINAL_COST_RATIO"] >= 2,
            df["FINAL_COST_RATIO"] >= 1.2,
            df["FINAL_COST_RATIO"] < 0.5,
        ],
        [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW_COST",
        ],
        default="NORMAL"
    )

    # =====================================================
    # HUMAN EXPLANATION
    # =====================================================
    df = explanation_builder.build(df)

    result = (
        df.iloc[0]
        .replace({np.nan: None})
        .to_dict()
    )

    return api_output_builder.build(result)