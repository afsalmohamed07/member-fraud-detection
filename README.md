# Insurance FWA Anomaly Detection & Investigation Intelligence Platform

This project is a production-style healthcare insurance FWA (Fraud, Waste, Abuse) anomaly detection platform designed for claim-line level intelligence scoring and investigation support.

The system combines:

- Historical baseline intelligence
- CatBoost expected cost prediction
- LightGBM risk classification
- Isolation Forest anomaly detection
- Provider / doctor behavioral intelligence
- Clinical validation rules
- Investigation evidence generation

---

# Core Objectives

The platform focuses on detecting:

- Abnormal claim cost behavior
- Provider pricing anomalies
- Doctor prescribing anomalies
- Drug-diagnosis mismatches
- Duration and dosage anomalies
- Treatment usage anomalies
- Upcoding behavior
- Similar historical rejected claim patterns
- Clinical inconsistency signals
- New / unseen treatment intelligence

---

# Main Models Used

## 1. CatBoost Quantile Cost Models

Three CatBoost models are trained:

- P50 expected cost model
- P75 upper-normal cost model
- P90 high-risk cost threshold model

Purpose:

- Predict expected claim cost
- Compare actual vs expected cost
- Generate financial anomaly intelligence

Used Features:

- SERVICE_TYPE
- PA_CATG
- PA_PRIMARY_DIAG
- PROV_TREAT_CODE
- Provider information
- Treatment family intelligence

---

## 2. Isolation Forest

Used for:

- Unsupervised anomaly detection
- Behavioral anomaly scoring
- Rare claim behavior identification

Focus Areas:

- Cost ratios
- Provider behavior
- Drug rarity
- Treatment rarity
- Duration anomalies
- Dosage anomalies
- Clinical mismatch patterns

---

## 3. LightGBM Risk Classifier

Final classification engine.

Predicts:

- NORMAL
- LOW_COST
- MEDIUM
- HIGH
- CRITICAL

Uses combined intelligence from:

- CatBoost expected cost
- Historical baselines
- Isolation score
- Upcoding signals
- Provider intelligence
- Similar claims intelligence
- Clinical anomaly rules

---

# Main Intelligence Areas

## Cost Intelligence

Detects:

- High-cost claims
- Extremely high-cost claims
- Provider-specific pricing deviations
- Cost ratio abnormalities

---

## Treatment Intelligence

Detects:

- Unseen treatment codes
- Rare treatment usage
- Diagnosis-treatment mismatch
- Unexpected treatment probability

---

## Drug Intelligence

Detects:

- Drug not in master
- Drug not used historically for diagnosis
- Rare drug usage
- Unseen drugs
- Drug-category mismatch

---

## Duration & Dosage Intelligence

Detects:

- Duration outside expected range
- Extreme duration values
- Dosage outside expected range
- Clinically uncommon dosage patterns

---

## Provider & Doctor Intelligence

Detects:

- Provider abnormal pricing patterns
- Doctor treatment anomalies
- Doctor drug anomalies
- Facility behavior mismatch
- Multiple license anomalies
- Historical rejection behavior

---

## Upcoding Intelligence

Detects:

- Provider treatment usage significantly above diagnosis-level historical distribution
- Unusual provider treatment concentration
- High provider usage ratio compared to global diagnosis usage

---

## Similar Claims Intelligence

Finds:

- Historically similar claims
- Rejected similar claims
- Partial approval patterns
- Similar claim rejection rate
- Top historical rejection codes
- Similar historical cost distribution

---

# Full Working Flow

```text
Raw Claim Files
        ↓
Dataset Merge
        ↓
Data Validation
        ↓
Data Cleaning & Normalization
        ↓
Feature Engineering
        ↓
Baseline Artifact Creation
        ↓
CatBoost Model Training
        ↓
Isolation Forest Training
        ↓
LightGBM Risk Classifier Training
        ↓
Provider / Similar Claims Intelligence Build
        ↓
New Claim Input
        ↓
Baseline Matching
        ↓
CatBoost Expected Cost Prediction
        ↓
Isolation Anomaly Scoring
        ↓
Behavioral Intelligence Scoring
        ↓
Final Risk Classification
        ↓
Human-readable Investigation Explanation
```

# Output Generated

The system generates:

- Risk Score
- Final Risk Label
- Cost anomaly explanation
- Provider anomaly explanation
- Upcoding explanation
- Similar claims explanation
- Isolation anomaly score
- CatBoost expected cost prediction
- Historical support evidence
- Top rejection reason evidence

---

# Main Technologies Used

- Python
- Pandas
- NumPy
- CatBoost
- LightGBM
- Scikit-learn
- Isolation Forest
- Joblib
- Streamlit
- FastAPI

---

# Full Rebuild Pipeline

```bash
python -m scripts.99_run_full_rebuild
```

This pipeline automatically performs:

- Dataset merge
- Validation
- Preprocessing
- Baseline creation
- CatBoost training
- Isolation Forest training
- LightGBM training
- Investigation artifact generation
- Treatment intelligence generation

---

# Individual Run Order

```bash
python -m scripts.00_merge_datasets
python -m scripts.01_validate_data
python -m scripts.02_preprocess_data
python -m scripts.03_build_baseline
python -m scripts.10_build_cost_prediction_baselines
python -m scripts.06_train_catboost
python -m scripts.07_train_isolation
python -m scripts.11_train_pharmacy_cost_model
python -m scripts.14_train_risk_classifier
python -m scripts.16_build_investigation_artifacts
python -m scripts.17_build_treatment_intelligence
```

# Run Application

## Streamlit

```bash
streamlit run app.py
```

## FastAPI

```bash
uvicorn app:app --reload
```