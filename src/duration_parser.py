import re
import pandas as pd
import numpy as np


class DurationParser:

    def __init__(self):

        self.day_pattern = re.compile(r"(\d+\.?\d*)D")
        self.week_pattern = re.compile(r"(\d+\.?\d*)W")
        self.month_pattern = re.compile(r"(\d+\.?\d*)M")

    def clean_duration_text(self, value):

        if pd.isna(value):
            return np.nan

        value = str(value).upper().strip()

        # remove spaces
        value = value.replace(" ", "")

        # normalize repeated units
        value = re.sub(r"D+", "D", value)
        value = re.sub(r"W+", "W", value)
        value = re.sub(r"M+", "M", value)

        # remove leading zeros
        value = re.sub(r"^0+", "", value)

        # fix empty after zero removal
        if value == "":
            return np.nan

        return value

    def duration_to_days(self, value):

        value = self.clean_duration_text(value)

        if pd.isna(value):
            return np.nan

        try:

            # DAYS
            day_match = self.day_pattern.fullmatch(value)
            if day_match:
                return round(float(day_match.group(1)), 2)

            # WEEKS
            week_match = self.week_pattern.fullmatch(value)
            if week_match:
                return round(float(week_match.group(1)) * 7, 2)

            # MONTHS
            month_match = self.month_pattern.fullmatch(value)
            if month_match:
                return round(float(month_match.group(1)) * 30, 2)

            # plain numeric
            if value.isdigit():
                return float(value)

            return np.nan

        except:
            return np.nan

    def detect_duration_unit(self, value):

        value = self.clean_duration_text(value)

        if pd.isna(value):
            return "UNKNOWN"

        if value.endswith("D"):
            return "DAY"

        if value.endswith("W"):
            return "WEEK"

        if value.endswith("M"):
            return "MONTH"

        return "UNKNOWN"

    def process_duration_column(self, df, duration_col):

        df = df.copy()

        df["DURATION_RAW_CLEANED"] = (
            df[duration_col]
            .apply(self.clean_duration_text)
        )

        df["DURATION_DAYS"] = (
            df[duration_col]
            .apply(self.duration_to_days)
        )

        df["DURATION_UNIT"] = (
            df[duration_col]
            .apply(self.detect_duration_unit)
        )

        return df


if __name__ == "__main__":

    sample = pd.DataFrame({
        "DRUG_DURATION": [
            "7D",
            "30DD",
            "2W",
            "1M",
            "090DD",
            "3MMM",
            "12WWW",
            "0.3W",
            "365D",
            "5DDD",
            "0"
        ]
    })

    parser = DurationParser()

    result = parser.process_duration_column(
        sample,
        "DRUG_DURATION"
    )

    print(result)