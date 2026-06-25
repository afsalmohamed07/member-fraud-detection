import pandas as pd
from pathlib import Path
from datetime import datetime


class AuditorFeedbackManager:

    def __init__(self):

        self.feedback_path = Path(
            "data/auditor_feedback/auditor_feedback.csv"
        )

        self.feedback_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.feedback_path.exists():

            df = pd.DataFrame(columns=[

                "CLAIM_ID",

                "RISK_SCORE",

                "TRIGGERED_SIGNALS",

                "AUDITOR_DECISION",

                "AUDITOR_REASON",

                "REVIEW_TIMESTAMP",

            ])

            df.to_csv(
                self.feedback_path,
                index=False
            )

    def save_feedback(
        self,
        claim_id,
        risk_score,
        triggered_signals,
        auditor_decision,
        auditor_reason,
    ):

        df = pd.read_csv(
            self.feedback_path
        )

        row = {

            "CLAIM_ID": claim_id,

            "RISK_SCORE": risk_score,

            "TRIGGERED_SIGNALS": (
                "|".join(triggered_signals)
                if isinstance(triggered_signals, list)
                else str(triggered_signals)
            ),

            "AUDITOR_DECISION": auditor_decision,

            "AUDITOR_REASON": auditor_reason,

            "REVIEW_TIMESTAMP": str(
                datetime.now()
            ),
        }

        df = pd.concat(
            [
                df,
                pd.DataFrame([row])
            ],
            ignore_index=True
        )

        df.to_csv(
            self.feedback_path,
            index=False
        )

        return row