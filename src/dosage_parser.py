import re
import pandas as pd
import numpy as np


class DosageParser:

    def __init__(self):
        pass

    def clean_dosage_text(self, value):
        if pd.isna(value):
            return "UNKNOWN"

        value = str(value).strip().upper()

        value = re.sub(r"\s+", " ", value)

        if value in ["", "NAN", "NONE", "NULL"]:
            return "UNKNOWN"

        return value

    def dosage_to_per_day(self, value):
        value = self.clean_dosage_text(value)

        if value == "UNKNOWN":
            return np.nan

        mapping = {
            "ONCE A DAY": 1,
            "EVERY DAY": 1,
            "EVERY MORNING": 1,
            "AT BED TIME": 1,

            "TWICE A DAY": 2,
            "EVERY 12 HOURS": 2,

            "3 TIMES A DAY": 3,
            "THRICE A DAY": 3,
            "EVERY 8 HOURS": 3,

            "4 TIMES A DAY": 4,
            "EVERY 6 HOURS": 4,

            "5 TIMES A DAY": 5,

            "6 TIMES A DAY": 6,

            "EVERY 5 HOURS": 4.8,
            "EVERY 4 HOURS": 6,
            "EVERY 3 HOURS": 8,
            "EVERY 2 HOURS": 12,
            "EVERY HOUR": 24,

            "STAT": 1,
            "IMMEDIATELY": 1,

            "PRN(AS NEEDED)": np.nan,
            "PRN": np.nan,

            "EVERY OTHER DAY": 0.5,

            "ONCE WEEKLY": round(1 / 7, 4),
            "TWICE WEEKLY": round(2 / 7, 4),
            "3 TIMES A WEEK": round(3 / 7, 4),

            "EVERY 2 WEEKS": round(1 / 14, 4),
            "EVERY 3 WEEKS": round(1 / 21, 4),
            "EVERY 4 WEEKS": round(1 / 28, 4),

            "ONCE MONTHLY": round(1 / 30, 4),
            "EVERY MONTH": round(1 / 30, 4),
            "EVERY 2 MONTHS": round(1 / 60, 4),
            "EVERY 4 MONTHS": round(1 / 120, 4),
        }

        if value in mapping:
            return mapping[value]

        # pattern: "3 TIMES A DAY"
        match = re.search(r"(\d+)\s*TIMES?\s*A\s*DAY", value)
        if match:
            return float(match.group(1))

        # pattern: "EVERY 8 HOURS"
        match = re.search(r"EVERY\s*(\d+)\s*HOURS?", value)
        if match:
            hours = float(match.group(1))
            if hours > 0:
                return round(24 / hours, 4)

        # pattern: "3 TIMES A WEEK"
        match = re.search(r"(\d+)\s*TIMES?\s*A\s*WEEK", value)
        if match:
            return round(float(match.group(1)) / 7, 4)

        return np.nan

    def normalize_dosage_label(self, value):
        per_day = self.dosage_to_per_day(value)
        cleaned = self.clean_dosage_text(value)

        if pd.isna(per_day):
            if cleaned in ["PRN", "PRN(AS NEEDED)"]:
                return "PRN_AS_NEEDED"
            return "UNKNOWN"

        if per_day == 1:
            return "1_PER_DAY"

        if per_day == 2:
            return "2_PER_DAY"

        if per_day == 3:
            return "3_PER_DAY"

        if per_day == 4:
            return "4_PER_DAY"

        if per_day == 5:
            return "5_PER_DAY"

        if per_day == 6:
            return "6_PER_DAY"

        if per_day == 24:
            return "24_PER_DAY"

        if per_day < 1:
            return "LESS_THAN_1_PER_DAY"

        return f"{per_day}_PER_DAY"

    def detect_dosage_type(self, value):
        cleaned = self.clean_dosage_text(value)

        if cleaned == "UNKNOWN":
            return "UNKNOWN"

        if "WEEK" in cleaned:
            return "WEEKLY"

        if "MONTH" in cleaned:
            return "MONTHLY"

        if "HOUR" in cleaned:
            return "HOURLY"

        if "DAY" in cleaned or "MORNING" in cleaned or "BED" in cleaned:
            return "DAILY"

        if cleaned in ["STAT", "IMMEDIATELY"]:
            return "STAT"

        if "PRN" in cleaned:
            return "PRN"

        return "OTHER"

    def process_dosage_column(self, df, dosage_col):
        df = df.copy()

        df["DOSAGE_RAW_CLEANED"] = df[dosage_col].apply(
            self.clean_dosage_text
        )

        df["DOSAGE_PER_DAY"] = df[dosage_col].apply(
            self.dosage_to_per_day
        )

        df["DOSAGE_NORMALIZED"] = df[dosage_col].apply(
            self.normalize_dosage_label
        )

        df["DOSAGE_TYPE"] = df[dosage_col].apply(
            self.detect_dosage_type
        )

        return df


if __name__ == "__main__":

    sample = pd.DataFrame({
        "DOSAGE_DESC": [
            "3 times a Day",
            "Twice a Day",
            "Once a day",
            "Once Weekly",
            "4 times a Day",
            "Every 12 hours",
            "At bed time",
            "Every Other Day",
            "PRN(As Needed)",
            "Once Monthly",
            "Stat",
            "Every 6 Hours",
            "Every 8 hours",
            "Twice Weekly",
            "6 times a Day",
            "Every 2 weeks",
            "5 times a Day",
            "Every morning",
            "Every 5 Hours",
            "Every day",
            "3 times a week",
            "Every 2 Hours",
            "Every month",
            "Every hour",
            "Every 3 hours",
            "Immediately",
            "Every 3 Weeks",
            "Every 4 Weeks",
            "Every 4 Months",
            "Every 2 Months"
        ]
    })

    parser = DosageParser()

    result = parser.process_dosage_column(
        sample,
        "DOSAGE_DESC"
    )

    print(result)