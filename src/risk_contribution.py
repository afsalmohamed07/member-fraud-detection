import pandas as pd


class RiskContributionEngine:
    def __init__(self):
        self.weights = {
            "UNKNOWN_DIAGNOSIS_ANOMALY": 30,
            "UNSEEN_TREATMENT_CODE_ANOMALY": 25,
            "RARE_TREATMENT_ANOMALY": 10,

            "PHARMACY_MISSING_DRUG_ANOMALY": 25,
            "NON_PHARMACY_WITH_DRUG_ANOMALY": 20,

            "DRUG_NOT_IN_MASTER_ANOMALY": 30,
            "DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY": 25,
            "UNSEEN_DRUG_ANOMALY": 25,
            "RARE_DRUG_ANOMALY": 10,

            "DURATION_OUTSIDE_RANGE_ANOMALY": 10,
            "EXTREME_DURATION_ANOMALY": 25,
            "DOSAGE_OUTSIDE_RANGE_ANOMALY": 15,

            "MULTIPLE_LICENSE_ANOMALY": 30,
            "MISSING_LICENSE_ANOMALY": 25,
            "UNSEEN_LICENSE_ANOMALY": 35,
            "LICENSE_ANOMALY": 20,

            "DOCTOR_FACILITY_ANOMALY": 20,
            "DOCTOR_TREATMENT_ANOMALY": 25,
            "DOCTOR_DRUG_ANOMALY": 20,

            "PROVIDER_BEHAVIOR_ANOMALY": 20,
            "PHARMACY_HIGH_COST_ANOMALY": 20,

            "ISOLATION_ANOMALY": 10,
            "ISOLATION_CONTEXT_ANOMALY": 10,
            "COST_ANOMALY_FINAL": 20,
        }

        self.names = {
            "UNKNOWN_DIAGNOSIS_ANOMALY": "Unknown diagnosis",
            "UNSEEN_TREATMENT_CODE_ANOMALY": "Unseen treatment",
            "RARE_TREATMENT_ANOMALY": "Rare treatment",
            "PHARMACY_MISSING_DRUG_ANOMALY": "Pharmacy missing drug",
            "NON_PHARMACY_WITH_DRUG_ANOMALY": "Drug under non-pharmacy",
            "DRUG_NOT_IN_MASTER_ANOMALY": "Drug not in master",
            "DRUG_NOT_USED_FOR_DIAGNOSIS_ANOMALY": "Drug not used for diagnosis",
            "UNSEEN_DRUG_ANOMALY": "Unseen drug",
            "RARE_DRUG_ANOMALY": "Rare drug",
            "DURATION_OUTSIDE_RANGE_ANOMALY": "Duration outside range",
            "EXTREME_DURATION_ANOMALY": "Extreme duration",
            "DOSAGE_OUTSIDE_RANGE_ANOMALY": "Dosage outside range",
            "MULTIPLE_LICENSE_ANOMALY": "Multiple license",
            "MISSING_LICENSE_ANOMALY": "Missing license",
            "UNSEEN_LICENSE_ANOMALY": "Unseen license",
            "LICENSE_ANOMALY": "License anomaly",
            "DOCTOR_FACILITY_ANOMALY": "Doctor facility anomaly",
            "DOCTOR_TREATMENT_ANOMALY": "Doctor treatment anomaly",
            "DOCTOR_DRUG_ANOMALY": "Doctor drug anomaly",
            "PROVIDER_BEHAVIOR_ANOMALY": "Provider behavior anomaly",
            "PHARMACY_HIGH_COST_ANOMALY": "Pharmacy high cost",
            "ISOLATION_ANOMALY": "Isolation anomaly",
            "ISOLATION_CONTEXT_ANOMALY": "Isolation context anomaly",
            "COST_ANOMALY_FINAL": "Cost anomaly",
        }

    def explain_row(self, row):
        contributions = []

        for col, weight in self.weights.items():
            if col not in row.index:
                continue

            value = row[col]

            if pd.isna(value):
                continue

            try:
                triggered = int(value) == 1
            except Exception:
                triggered = False

            if triggered:
                contributions.append(
                    {
                        "signal": col,
                        "name": self.names.get(col, col),
                        "impact": weight,
                    }
                )

        contributions = sorted(
            contributions,
            key=lambda x: x["impact"],
            reverse=True
        )

        return contributions

    def top_drivers_text(self, contributions, top_n=5):
        if len(contributions) == 0:
            return "No major risk drivers triggered"

        top = contributions[:top_n]

        return " | ".join(
            [
                f"{item['name']} (+{item['impact']})"
                for item in top
            ]
        )