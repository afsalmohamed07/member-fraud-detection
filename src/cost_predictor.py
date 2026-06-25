import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class CostPredictor:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]

        baseline_path = Path(config["paths"]["baselines_dir"]) / "cost_prediction_baselines.pkl"
        self.baselines = joblib.load(baseline_path)

    def _get_family(self, code):
        code = str(code).upper().strip()
        code = re.sub(r"[^A-Z0-9]", "", code)

        if code.startswith("PHARMACY"):
            return "PHARMACY"

        match = re.match(r"^[A-Z]+", code)
        if match:
            return match.group(0)

        return "UNKNOWN"

    def _prepare_row(self, row):
        row = row.copy()

        if "PROV_TREAT_FAMILY" not in row.index:
            row["PROV_TREAT_FAMILY"] = self._get_family(
                row.get("PROV_TREAT_CODE_CLEAN", "")
            )

        return row

    def _top_price_summary(self, cost_list, top_n=5):
        if cost_list is None:
            return []

        if not isinstance(cost_list, list):
            return []

        if len(cost_list) == 0:
            return []

        s = pd.Series(cost_list)
        freq = s.value_counts().head(top_n)

        return [
            {
                "charge": float(price),
                "count": int(count),
            }
            for price, count in freq.items()
        ]

    def _mode_info(self, cost_list):
        summary = self._top_price_summary(cost_list, top_n=1)

        if not summary:
            return {
                "mode_cost": None,
                "mode_count": 0,
                "mode_percent": 0,
            }

        total = len(cost_list)
        mode_count = summary[0]["count"]

        return {
            "mode_cost": summary[0]["charge"],
            "mode_count": mode_count,
            "mode_percent": round((mode_count / total) * 100, 2) if total else 0,
        }

    def _empty_history(self, prefix, level_name):
        return {
            f"{prefix}_LEVEL": level_name,
            f"{prefix}_SUPPORT": 0,
            f"{prefix}_CONFIDENCE": "NO_MATCH",
            f"{prefix}_MEAN": None,
            f"{prefix}_MEDIAN": None,
            f"{prefix}_P75": None,
            f"{prefix}_P90": None,
            f"{prefix}_MAX": None,
            f"{prefix}_MODE_COST": None,
            f"{prefix}_MODE_COUNT": 0,
            f"{prefix}_MODE_PERCENT": 0,
            f"{prefix}_COMMON_CHARGES": [],
        }

    def _extract_history(self, result, prefix, level_name):
        cost_list = result.get("BASELINE_COST_LIST", [])
        mode = self._mode_info(cost_list)

        return {
            f"{prefix}_LEVEL": level_name,
            f"{prefix}_SUPPORT": result.get("SUPPORT_COUNT"),
            f"{prefix}_CONFIDENCE": result.get("CONFIDENCE_LEVEL"),
            f"{prefix}_MEAN": result.get("BASELINE_COST_MEAN"),
            f"{prefix}_MEDIAN": result.get("BASELINE_COST_MEDIAN"),
            f"{prefix}_P75": result.get("BASELINE_COST_P75"),
            f"{prefix}_P90": result.get("BASELINE_COST_P90"),
            f"{prefix}_MAX": result.get("BASELINE_COST_MAX"),
            f"{prefix}_MODE_COST": mode["mode_cost"],
            f"{prefix}_MODE_COUNT": mode["mode_count"],
            f"{prefix}_MODE_PERCENT": mode["mode_percent"],
            f"{prefix}_COMMON_CHARGES": self._top_price_summary(cost_list),
        }

    def _match_table(self, table, row, group_cols):
        condition = pd.Series([True] * len(table), index=table.index)

        for col in group_cols:
            if col not in row.index or col not in table.columns:
                return pd.DataFrame()

            condition = condition & (table[col] == row[col])

        return table[condition]

    def _find_history_by_cols(self, row, group_cols, prefix, level_name):
        for _, obj in self.baselines.items():
            if not isinstance(obj, dict):
                continue

            if "group_cols" not in obj or "table" not in obj:
                continue

            if obj["group_cols"] != group_cols:
                continue

            matched = self._match_table(obj["table"], row, group_cols)

            if len(matched) == 0:
                return self._empty_history(prefix, f"NO_{level_name}")

            return self._extract_history(
                matched.iloc[0].to_dict(),
                prefix,
                level_name
            )

        return self._empty_history(prefix, f"NO_{level_name}_TABLE")

    def _find_provider_history(self, row):
        diag_col = self.columns["diagnosis"]
        provider_col = self.columns["provider"]

        group_cols = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            diag_col,
            "PROV_TREAT_CODE_CLEAN",
            provider_col,
        ]

        return self._find_history_by_cols(
            row=row,
            group_cols=group_cols,
            prefix="PROVIDER",
            level_name="PROVIDER_DIAG_TREAT"
        )

    def _find_global_history(self, row):
        """
        Global = same service/category/diagnosis/treatment across all providers.
        Since current cost_prediction_baselines.pkl may be provider-level,
        we aggregate from the provider-level table by ignoring provider.
        """
        diag_col = self.columns["diagnosis"]

        if "level_1" not in self.baselines:
            return self._empty_history("GLOBAL", "NO_LEVEL_1_TABLE")

        obj = self.baselines["level_1"]
        table = obj["table"]

        match_cols = [
            "SERVICE_TYPE_NORM",
            "PA_CATG_NORM",
            diag_col,
            "PROV_TREAT_CODE_CLEAN",
        ]

        condition = pd.Series([True] * len(table), index=table.index)

        for col in match_cols:
            if col not in row.index or col not in table.columns:
                return self._empty_history("GLOBAL", "NO_GLOBAL_COLUMNS")

            condition = condition & (table[col] == row[col])

        matched = table[condition]

        if len(matched) == 0:
            return self._empty_history("GLOBAL", "NO_GLOBAL_DIAG_TREAT")

        all_costs = []

        for costs in matched.get("BASELINE_COST_LIST", []):
            if isinstance(costs, list):
                all_costs.extend(costs)

        if len(all_costs) == 0:
            support = matched["SUPPORT_COUNT"].sum()
            median = matched["BASELINE_COST_MEDIAN"].median()
            p75 = matched["BASELINE_COST_P75"].median()
            p90 = matched["BASELINE_COST_P90"].median()
            max_v = matched["BASELINE_COST_MAX"].max()
            mean_v = matched["BASELINE_COST_MEAN"].mean()
            all_costs = [median] if pd.notna(median) else []
        else:
            support = len(all_costs)
            s = pd.Series(all_costs)
            median = s.median()
            p75 = s.quantile(0.75)
            p90 = s.quantile(0.90)
            max_v = s.max()
            mean_v = s.mean()

        mode = self._mode_info(all_costs)

        return {
            "GLOBAL_LEVEL": "GLOBAL_DIAG_TREAT",
            "GLOBAL_SUPPORT": int(support),
            "GLOBAL_CONFIDENCE": (
                "HIGH" if support >= 30 else
                "MEDIUM" if support >= 10 else
                "LOW" if support >= 3 else
                "VERY_LOW"
            ),
            "GLOBAL_MEAN": float(mean_v) if pd.notna(mean_v) else None,
            "GLOBAL_MEDIAN": float(median) if pd.notna(median) else None,
            "GLOBAL_P75": float(p75) if pd.notna(p75) else None,
            "GLOBAL_P90": float(p90) if pd.notna(p90) else None,
            "GLOBAL_MAX": float(max_v) if pd.notna(max_v) else None,
            "GLOBAL_MODE_COST": mode["mode_cost"],
            "GLOBAL_MODE_COUNT": mode["mode_count"],
            "GLOBAL_MODE_PERCENT": mode["mode_percent"],
            "GLOBAL_COMMON_CHARGES": self._top_price_summary(all_costs),
        }

    def _predict_ml_cost(self, row):
        from src.ml_scoring import MLPredictor

        predictor = MLPredictor(self.config)
        return predictor.predict_cost_row(row)

    def _predict_pharmacy_ml_cost(self, row):
        try:
            from src.pharmacy_ml_model import PharmacyCostModel

            predictor = PharmacyCostModel(self.config)
            return predictor.predict_row(row)
        except Exception:
            return np.nan

    def _is_pharmacy_case(self, row):
        pa_catg_raw = str(row.get("PA_CATG", "")).strip().upper()
        pa_catg_norm = str(row.get("PA_CATG_NORM", "")).strip().upper()
        treat_raw = str(row.get("PROV_TREAT_CODE", "")).strip().upper()
        treat_clean = str(row.get("PROV_TREAT_CODE_CLEAN", "")).strip().upper()

        drug_col = self.columns["drug_name"]
        drug_value = str(row.get(drug_col, "")).strip().upper()

        drug_present = drug_value not in [
            "",
            "NAN",
            "NONE",
            "NULL",
            "NA",
            "N/A",
            "UNKNOWN",
        ]

        return (
            pa_catg_raw == "PHARMACY" or
            pa_catg_norm == "PHARMACY" or
            treat_raw == "PHARMACY" or
            treat_clean == "PHARMACY" or
            drug_present
        )

    def _final_engine(self, provider_history, global_history, ml_result):
        p50 = ml_result.get("P50_COST")
        p75 = ml_result.get("P75_COST")
        p90 = ml_result.get("P90_COST")

        return {
            "FINAL_EXPECTED_COST_ENGINE": p50,
            "TRUSTED_EVIDENCE": "CATBOOST_COST_MODEL",
            "FINAL_ENGINE_REASON": (
                f"CatBoost cost model used for final cost prediction. "
                f"Expected cost={p50}, normal upper limit={p75}, high-risk limit={p90}."
            ),

            "CATBOOST_EXPECTED_COST": p50,
            "CATBOOST_NORMAL_UPPER": p75,
            "CATBOOST_HIGH_RISK_LIMIT": p90,

            # history only support, not prediction
            "PROVIDER_HISTORY_SUPPORT": provider_history.get("PROVIDER_SUPPORT") or 0,
            "GLOBAL_HISTORY_SUPPORT": global_history.get("GLOBAL_SUPPORT") or 0,
        }

    def predict_cost(self, row, debug=True):
        row = self._prepare_row(row)

        is_pharmacy = self._is_pharmacy_case(row)

        if is_pharmacy:
            ml_expected_raw = self._predict_pharmacy_ml_cost(row)

            try:
                pharmacy_expected = float(ml_expected_raw)
            except Exception:
                pharmacy_expected = np.nan

            if pd.notna(pharmacy_expected) and pharmacy_expected > 0:
                pharmacy_normal_upper = round(pharmacy_expected * 1.5, 2)
                pharmacy_high_risk = round(pharmacy_expected * 2.0, 2)
            else:
                pharmacy_normal_upper = np.nan
                pharmacy_high_risk = np.nan

            return {
                "IS_PHARMACY_CASE": 1,

                "P50_COST": pharmacy_expected,
                "P75_COST": pharmacy_normal_upper,
                "P90_COST": pharmacy_high_risk,

                "CATBOOST_EXPECTED_COST": pharmacy_expected,
                "CATBOOST_NORMAL_UPPER": pharmacy_normal_upper,
                "CATBOOST_HIGH_RISK_LIMIT": pharmacy_high_risk,

                "ML_EXPECTED_COST": pharmacy_expected,
                "FINAL_EXPECTED_COST_ENGINE": pharmacy_expected,

                "ML_NORMAL_LOW": pharmacy_expected,
                "ML_NORMAL_HIGH": pharmacy_normal_upper,
                "ML_HIGH_RISK_LIMIT": pharmacy_high_risk,

                "TRUSTED_EVIDENCE": "PHARMACY_DRUG_PROVIDER_MODEL",
                "FINAL_ENGINE_REASON": (
                    "Pharmacy claim uses pharmacy drug/provider model."
                ),

                "final_expected_cost": pharmacy_expected,
                "ml_expected_cost": pharmacy_expected,
                "p50_cost": pharmacy_expected,
                "p75_cost": pharmacy_normal_upper,
                "p90_cost": pharmacy_high_risk,
            }
        provider_history = self._find_provider_history(row)
        global_history = self._find_global_history(row)

        try:
            ml_result = self._predict_ml_cost(row)
        except Exception as e:
            print("CATBOOST ML PREDICTION ERROR:", e)
            ml_result = {
                "P50_COST": np.nan,
                "P75_COST": np.nan,
                "P90_COST": np.nan,
                "ML_EXPECTED_COST": np.nan,
                "ML_NORMAL_LOW": np.nan,
                "ML_NORMAL_HIGH": np.nan,
                "ML_HIGH_RISK_LIMIT": np.nan,
            }

        final_engine = self._final_engine(
            provider_history=provider_history,
            global_history=global_history,
            ml_result=ml_result,
        )

        return {
            "IS_PHARMACY_CASE": 0,

            **global_history,
            **provider_history,
            **ml_result,
            **final_engine,

            "CATBOOST_EXPECTED_COST": ml_result.get("P50_COST"),
            "CATBOOST_NORMAL_UPPER": ml_result.get("P75_COST"),
            "CATBOOST_HIGH_RISK_LIMIT": ml_result.get("P90_COST"),

            "final_expected_cost": final_engine["FINAL_EXPECTED_COST_ENGINE"],
            "ml_expected_cost": ml_result.get("ML_EXPECTED_COST"),
            "p50_cost": ml_result.get("P50_COST"),
            "p75_cost": ml_result.get("P75_COST"),
            "p90_cost": ml_result.get("P90_COST"),
            "source": final_engine["TRUSTED_EVIDENCE"],
            "support": provider_history.get("PROVIDER_SUPPORT"),
            "confidence": provider_history.get("PROVIDER_CONFIDENCE"),
            "pattern_median": provider_history.get("PROVIDER_MEDIAN"),
            "pattern_p75": provider_history.get("PROVIDER_P75"),
            "pattern_p90": provider_history.get("PROVIDER_P90"),
            "pattern_max": provider_history.get("PROVIDER_MAX"),
            "top_price_summary": provider_history.get("PROVIDER_COMMON_CHARGES", []),
        }