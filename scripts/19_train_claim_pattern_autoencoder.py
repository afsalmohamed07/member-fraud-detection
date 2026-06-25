import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from src.config_loader import ConfigLoader


FEATURE_COLS = [

    "DIAGNOSIS_TREATMENT_PROB",

    "DIAGNOSIS_DRUG_PROB",

    "DOCTOR_TREATMENT_PROB",

    "DOCTOR_DRUG_PROB",

    "PROVIDER_TREATMENT_PROB",

 

    "DURATION_SCORE",

    "DOSAGE_SCORE",
]


# =====================================================
# SAFE STRING
# =====================================================
def safe_str(x):

    if pd.isna(x):
        return "UNKNOWN"

    return str(x).strip().upper()


# =====================================================
# BUILD PROBABILITY MAPS
# =====================================================
def build_probability_maps(df):

    print("\nBuilding probability maps...")

    # =====================================================
    # DIAGNOSIS -> TREATMENT
    # =====================================================
    diag_treat = (

        df.groupby(
            [
                "PA_PRIMARY_DIAG",
                "PROV_TREAT_CODE",
            ]
        )

        .size()

        .reset_index(name="COUNT")
    )

    diag_total = (

        df.groupby("PA_PRIMARY_DIAG")

        .size()

        .reset_index(name="TOTAL")
    )

    diag_treat = diag_treat.merge(
        diag_total,
        on="PA_PRIMARY_DIAG",
        how="left"
    )

    diag_treat["PROB"] = (
        diag_treat["COUNT"]
        /
        diag_treat["TOTAL"]
    )

    diag_treat_map = {

        (
            safe_str(r["PA_PRIMARY_DIAG"]),
            safe_str(r["PROV_TREAT_CODE"])
        ): r["PROB"]

        for _, r in diag_treat.iterrows()
    }

    # =====================================================
    # DIAGNOSIS -> DRUG
    # =====================================================
    diag_drug = (

        df.groupby(
            [
                "PA_PRIMARY_DIAG",
                "PAT_DRUG_NAME",
            ]
        )

        .size()

        .reset_index(name="COUNT")
    )

    diag_drug_total = (

        df.groupby("PA_PRIMARY_DIAG")

        .size()

        .reset_index(name="TOTAL")
    )

    diag_drug = diag_drug.merge(
        diag_drug_total,
        on="PA_PRIMARY_DIAG",
        how="left"
    )

    diag_drug["PROB"] = (
        diag_drug["COUNT"]
        /
        diag_drug["TOTAL"]
    )

    diag_drug_map = {

        (
            safe_str(r["PA_PRIMARY_DIAG"]),
            safe_str(r["PAT_DRUG_NAME"])
        ): r["PROB"]

        for _, r in diag_drug.iterrows()
    }

    # =====================================================
    # DOCTOR -> TREATMENT
    # =====================================================
    doc_treat = (

        df.groupby(
            [
                "DOC_NAME",
                "PROV_TREAT_CODE",
            ]
        )

        .size()

        .reset_index(name="COUNT")
    )

    doc_total = (

        df.groupby("DOC_NAME")

        .size()

        .reset_index(name="TOTAL")
    )

    doc_treat = doc_treat.merge(
        doc_total,
        on="DOC_NAME",
        how="left"
    )

    doc_treat["PROB"] = (
        doc_treat["COUNT"]
        /
        doc_treat["TOTAL"]
    )

    doc_treat_map = {

        (
            safe_str(r["DOC_NAME"]),
            safe_str(r["PROV_TREAT_CODE"])
        ): r["PROB"]

        for _, r in doc_treat.iterrows()
    }

    # =====================================================
    # DOCTOR -> DRUG
    # =====================================================
    doc_drug = (

        df.groupby(
            [
                "DOC_NAME",
                "PAT_DRUG_NAME",
            ]
        )

        .size()

        .reset_index(name="COUNT")
    )

    doc_drug_total = (

        df.groupby("DOC_NAME")

        .size()

        .reset_index(name="TOTAL")
    )

    doc_drug = doc_drug.merge(
        doc_drug_total,
        on="DOC_NAME",
        how="left"
    )

    doc_drug["PROB"] = (
        doc_drug["COUNT"]
        /
        doc_drug["TOTAL"]
    )

    doc_drug_map = {

        (
            safe_str(r["DOC_NAME"]),
            safe_str(r["PAT_DRUG_NAME"])
        ): r["PROB"]

        for _, r in doc_drug.iterrows()
    }

    # =====================================================
    # PROVIDER -> TREATMENT
    # =====================================================
    prov_treat = (

        df.groupby(
            [
                "PROV_NAME",
                "PROV_TREAT_CODE",
            ]
        )

        .size()

        .reset_index(name="COUNT")
    )

    prov_total = (

        df.groupby("PROV_NAME")

        .size()

        .reset_index(name="TOTAL")
    )

    prov_treat = prov_treat.merge(
        prov_total,
        on="PROV_NAME",
        how="left"
    )

    prov_treat["PROB"] = (
        prov_treat["COUNT"]
        /
        prov_treat["TOTAL"]
    )

    prov_treat_map = {

        (
            safe_str(r["PROV_NAME"]),
            safe_str(r["PROV_TREAT_CODE"])
        ): r["PROB"]

        for _, r in prov_treat.iterrows()
    }

    return {

        "diag_treat_map": diag_treat_map,

        "diag_drug_map": diag_drug_map,

        "doc_treat_map": doc_treat_map,

        "doc_drug_map": doc_drug_map,

        "prov_treat_map": prov_treat_map,
    }


# =====================================================
# BUILD FEATURE FRAME
# =====================================================
def build_feature_frame(df, prob_maps):

    rows = []

    for _, row in df.iterrows():

        diag = safe_str(
            row.get("PA_PRIMARY_DIAG")
        )

        treat = safe_str(
            row.get("PROV_TREAT_CODE")
        )

        drug = safe_str(
            row.get("PAT_DRUG_NAME")
        )

        doctor = safe_str(
            row.get("DOC_NAME")
        )

        provider = safe_str(
            row.get("PROV_NAME")
        )

        est_amt = pd.to_numeric(
            row.get("PA_EST_AMT_LC", 0),
            errors="coerce"
        )

        if pd.isna(est_amt):
            est_amt = 0

        expected_cost = pd.to_numeric(
            row.get("FINAL_EXPECTED_COST", 0),
            errors="coerce"
        )

        if pd.isna(expected_cost) or expected_cost <= 0:
            cost_ratio = 1
        else:
            cost_ratio = est_amt / expected_cost

        duration = pd.to_numeric(
            row.get("DURATION_DAYS", 0),
            errors="coerce"
        )

        if pd.isna(duration):
            duration = 0

        dosage = pd.to_numeric(
            row.get("DOSAGE_PER_DAY", 0),
            errors="coerce"
        )

        if pd.isna(dosage):
            dosage = 0

        rows.append({

            "DIAGNOSIS_TREATMENT_PROB":
                prob_maps["diag_treat_map"].get(
                    (diag, treat),
                    0
                ),

            "DIAGNOSIS_DRUG_PROB":
                prob_maps["diag_drug_map"].get(
                    (diag, drug),
                    0
                ),

            "DOCTOR_TREATMENT_PROB":
                prob_maps["doc_treat_map"].get(
                    (doctor, treat),
                    0
                ),

            "DOCTOR_DRUG_PROB":
                prob_maps["doc_drug_map"].get(
                    (doctor, drug),
                    0
                ),

            "PROVIDER_TREATMENT_PROB":
                prob_maps["prov_treat_map"].get(
                    (provider, treat),
                    0
                ),

      

            "DURATION_SCORE":
                duration,

            "DOSAGE_SCORE":
                dosage,
        })

    out = pd.DataFrame(rows)

    out = out.replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)

    return out[FEATURE_COLS]


# =====================================================
# BUILD AUTOENCODER
# =====================================================
def build_autoencoder(input_dim):

    input_layer = Input(shape=(input_dim,))

    encoded = Dense(
        8,
        activation="relu"
    )(input_layer)

    encoded = Dense(
        4,
        activation="relu"
    )(encoded)

    decoded = Dense(
        8,
        activation="relu"
    )(encoded)

    decoded = Dense(
        input_dim,
        activation="linear"
    )(decoded)

    model = Model(
        input_layer,
        decoded
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# =====================================================
# MAIN TRAIN
# =====================================================
def main():

    print("\n======================================")
    print("TRAIN CLAIM PATTERN AUTOENCODER")
    print("======================================")

    config = ConfigLoader().get_config()

    processed_path = config["paths"]["processed_data"]

    models_dir = Path(
        config["paths"]["models_dir"]
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading processed data...")

    df = joblib.load(processed_path)

    print("Processed shape:", df.shape)

    for col in [

        "PA_PRIMARY_DIAG",

        "PROV_TREAT_CODE",

        "PAT_DRUG_NAME",

        "DOC_NAME",

        "PROV_NAME",
    ]:

        if col not in df.columns:
            df[col] = "UNKNOWN"

        df[col] = df[col].apply(safe_str)

    if "PA_EST_AMT_LC" not in df.columns:
        df["PA_EST_AMT_LC"] = 0

    if "FINAL_EXPECTED_COST" not in df.columns:
        df["FINAL_EXPECTED_COST"] = df["PA_EST_AMT_LC"]

    if "DURATION_DAYS" not in df.columns:
        df["DURATION_DAYS"] = 0

    if "DOSAGE_PER_DAY" not in df.columns:
        df["DOSAGE_PER_DAY"] = 0

    print("\nSelecting normal-looking rows...")

    normal_mask = (

        (
            pd.to_numeric(
                df.get("PA_REJ_AMT_LC", 0),
                errors="coerce"
            ).fillna(0)
            <= 0
        )

        &

        (
            pd.to_numeric(
                df.get("PA_APPR_AMT_LC", 0),
                errors="coerce"
            ).fillna(0)
            >= 0
        )
    )

    normal_df = df[normal_mask].copy()

    print("Normal rows:", len(normal_df))

    if len(normal_df) < 1000:

        raise ValueError(
            "Not enough normal rows to train autoencoder"
        )

    prob_maps = build_probability_maps(
        normal_df
    )

    print("\nBuilding feature frame...")

    X = build_feature_frame(
        normal_df,
        prob_maps
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("\nTraining autoencoder...")

    model = build_autoencoder(
        input_dim=X_scaled.shape[1]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(

        X_scaled,

        X_scaled,

        epochs=50,

        batch_size=512,

        validation_split=0.2,

        callbacks=[early_stop],

        verbose=1
    )

    print("\nCalculating threshold...")

    reconstructed = model.predict(
        X_scaled,
        verbose=0
    )

    errors = np.mean(
        np.square(X_scaled - reconstructed),
        axis=1
    )

    threshold = float(
        np.quantile(errors, 0.95)
    )

    print("Threshold:", threshold)

    model.save(
        models_dir / "claim_pattern_autoencoder.h5"
    )

    joblib.dump(
        scaler,
        models_dir / "claim_pattern_scaler.pkl"
    )

    joblib.dump(
        threshold,
        models_dir / "claim_pattern_threshold.pkl"
    )

    joblib.dump(
        FEATURE_COLS,
        models_dir / "claim_pattern_features.pkl"
    )

    joblib.dump(
        prob_maps,
        models_dir / "claim_pattern_probability_maps.pkl"
    )

    print("\nSaved autoencoder artifacts.")


if __name__ == "__main__":
    main()