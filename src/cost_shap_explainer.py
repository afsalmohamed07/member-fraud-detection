import joblib
import pandas as pd
from pathlib import Path
from catboost import Pool


class CostShapExplainer:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]

    def _clean_text(self, value):

        if value is None or pd.isna(value):
            return "UNKNOWN"

        value = str(value).upper().strip()
        value = " ".join(value.split())

        if value in ["", "NAN", "NONE", "NULL"]:
            return "UNKNOWN"

        if value.endswith(".0"):
            value = value[:-2]

        return value

    def _clean_code(self, value):

        value = self._clean_text(value)

        return value.replace(" ", "")

    def _get_model_path(self):

        paths = self.config.get("paths", {})

        possible = [
            paths.get("cost_model_path"),
            Path(paths.get("models_dir", "models")) / "catboost_cost_model.pkl",
            Path(paths.get("models_dir", "models")) / "cost_model.pkl",
        ]

        for path in possible:

            if path is None:
                continue

            path = Path(path)

            if path.exists():
                return path

        return None

    def explain_row(self, row, top_n=8):

        model_path = self._get_model_path()

        if model_path is None:
            return []

        try:
            model = joblib.load(model_path)

        except Exception as e:
            print("SHAP MODEL LOAD ERROR:", e)
            return []

        # =====================================================
        # LOAD EXACT TRAINED FEATURE ORDER
        # =====================================================

        models_dir = Path(self.config["paths"]["models_dir"])

        try:

            cost_features = joblib.load(
                models_dir / "catboost_features.pkl"
            )

        except Exception as e:

            print("FEATURE LOAD ERROR:", e)
            return []

        cat_feature_path = models_dir / "catboost_cat_features.pkl"

        if cat_feature_path.exists():

            categorical_features = joblib.load(cat_feature_path)

        else:

            categorical_features = []

        # =====================================================
        # BUILD INPUT ROW
        # =====================================================

        data = {}

        code_like_cols = [
            "PROV_TREAT_CODE_CLEAN",
            "CPT_CODE",
            "INS_TREAT_CODE",
        ]

        for col in cost_features:

            value = row.get(col, None)

            # categorical columns
            if col in categorical_features:

                if col in code_like_cols:
                    value = self._clean_code(value)
                else:
                    value = self._clean_text(value)

            # numeric/stat features
            else:

                value = pd.to_numeric(
                    value,
                    errors="coerce"
                )

                if pd.isna(value):
                    value = 0

            data[col] = [value]

        X = pd.DataFrame(data)

        cat_features_existing = [
            col for col in categorical_features
            if col in X.columns
        ]

        try:

            pool = Pool(
                X,
                cat_features=cat_features_existing
            )

            shap_values = model.get_feature_importance(
                pool,
                type="ShapValues"
            )

            row_shap = shap_values[0][:-1]

            contributions = []

            for feature, value, shap_value in zip(
                cost_features,
                X.iloc[0].tolist(),
                row_shap
            ):

                contributions.append(
                    {
                        "feature": feature,
                        "value": str(value),
                        "impact": round(float(shap_value), 4),
                    }
                )

            # hide internal statistical features from user-facing SHAP
            hidden_features = {
                "ML_BASELINE_MEDIAN",
                "ML_BASELINE_MEAN",
                "ML_BASELINE_MAX",
                "ML_BASELINE_P75",
                "ML_BASELINE_P90",
                "ML_SUPPORT_COUNT",
                "ML_MEAN_MEDIAN_RATIO",
                "ML_MAX_MEDIAN_RATIO",
                "ML_LOG_SUPPORT",
            }

            contributions = [
                item for item in contributions
                if item["feature"] not in hidden_features
            ]

            contributions = sorted(
                contributions,
                key=lambda x: abs(x["impact"]),
                reverse=True
            )
            

            return contributions[:top_n]

        except Exception as e:

            print("SHAP CALCULATION ERROR:", e)

            return []

    def top_drivers_text(self, contributions):

        if len(contributions) == 0:
            return "CatBoost SHAP explanation not available"

        return " | ".join(
            [
                f"{item['feature']}={item['value']} ({item['impact']:+})"
                for item in contributions
            ]
        )