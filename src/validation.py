import pandas as pd


class DataValidator:
    def __init__(self, config):
        self.config = config
        self.columns = config["columns"]

    def validate_required_columns(self, df: pd.DataFrame):
        required_cols = [
            self.columns["diagnosis"],
            self.columns["treatment_code"],
            self.columns["drug_name"],
            self.columns["scientific_drug_name"],
            self.columns["drug_duration"],
            self.columns["dosage_desc"],
            self.columns["quantity"],
            self.columns["provider"],
            self.columns["doctor"],
            self.columns["est_amount"],
            self.columns["appr_amount"],
            self.columns["rej_amount"],
            self.columns["service_type"],
            self.columns["category"],
        ]

        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return True

    def validate_empty_data(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Dataset is empty")

        return True

    def validate_amount_columns(self, df: pd.DataFrame):
        amount_cols = [
            self.columns["est_amount"],
            self.columns["appr_amount"],
            self.columns["rej_amount"],
        ]

        issues = {}

        for col in amount_cols:
            numeric_col = pd.to_numeric(df[col], errors="coerce")

            issues[col] = {
                "null_count": int(numeric_col.isna().sum()),
                "negative_count": int((numeric_col < 0).sum()),
                "zero_count": int((numeric_col == 0).sum()),
                "max_value": float(numeric_col.max()) if numeric_col.notna().any() else None,
                "min_value": float(numeric_col.min()) if numeric_col.notna().any() else None,
            }

        return issues

    def validate_duplicate_rows(self, df: pd.DataFrame):
        duplicate_count = int(df.duplicated().sum())

        return {
            "duplicate_rows": duplicate_count
        }

    def run_all_checks(self, df: pd.DataFrame):
        self.validate_empty_data(df)
        self.validate_required_columns(df)

        result = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "amount_column_issues": self.validate_amount_columns(df),
            "duplicate_check": self.validate_duplicate_rows(df),
            "status": "PASSED"
        }

        return result