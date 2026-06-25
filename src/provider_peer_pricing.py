import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class ProviderPeerPricing:

    def __init__(self, config):
        artifact_path = Path(
            "artifacts/intelligence/provider_peer_pricing.pkl"
        )

        self.artifact = joblib.load(artifact_path)

    def _safe(self, x):
        if pd.isna(x):
            return "UNKNOWN"

        value = str(x).strip().upper()

        if value in ["", "NAN", "NONE", "NULL", "NA", "N/A"]:
            return "UNKNOWN"

        return value

    def _mode_price(self, prices):
        if not prices:
            return None

        s = pd.Series(prices)

        return round(
            float(s.mode().iloc[0]),
            2
        )

    def _price_range(self, prices):
        if not prices:
            return None, None

        return (
            round(float(np.min(prices)), 2),
            round(float(np.max(prices)), 2),
        )

    def analyze(self, row):
        service = self._safe(
            row.get("SERVICE_TYPE_NORM")
            or row.get("SERVICE_TYPE")
        )

        category = self._safe(
            row.get("PA_CATG_NORM")
            or row.get("PA_CATG")
        )

        provider = self._safe(
            row.get("PROV_NAME")
        )

        actual = pd.to_numeric(
            row.get("PA_EST_AMT_LC", 0),
            errors="coerce"
        )

        if pd.isna(actual):
            actual = 0

        if category == "PHARMACY":
            drug = self._safe(
                row.get("PAT_DRUG_NAME")
            )

            context_key = (
                "PHARMACY",
                service,
                category,
                drug,
            )

            context_text = (
                f"pharmacy drug context {drug}"
            )

        else:
            diagnosis = self._safe(
                row.get("PA_PRIMARY_DIAG_CLEAN")
                or row.get("PA_PRIMARY_DIAG")
            )

            treatment = self._safe(
                row.get("PROV_TREAT_CODE_CLEAN")
                or row.get("PROV_TREAT_CODE")
            )

            context_key = (
                "NON_PHARMACY",
                service,
                category,
                diagnosis,
                treatment,
            )

            context_text = (
                f"service/category/diagnosis/treatment context "
                f"{service}/{category}/{diagnosis}/{treatment}"
            )

        context_data = self.artifact.get(context_key)

        if not context_data:
            return {
                "PROVIDER_PEER_PRICING":
                    f"No peer provider pricing history available for {context_text}.",
                "PROVIDER_PEER_PRICING_AVAILABLE": 0,
            }

        provider_prices = context_data.get("provider_prices", {})
        all_prices = context_data.get("all_prices", [])

        market_min, market_max = self._price_range(all_prices)
        market_common = self._mode_price(all_prices)

        provider_common = None
        provider_position = "UNKNOWN"
        provider_min = None
        provider_max = None

        if provider in provider_prices:
            provider_specific_prices = provider_prices[provider]

            provider_common = self._mode_price(
                provider_specific_prices
            )

            provider_min, provider_max = self._price_range(
                provider_specific_prices
            )

        if (
            provider_common is not None
            and market_common is not None
        ):
            if provider_common < market_common:
                provider_position = "LOWER_THAN_PEERS"

            elif provider_common > market_common:
                provider_position = "HIGHER_THAN_PEERS"

            else:
                provider_position = "SIMILAR_TO_PEERS"

        msg = (
            f"For this {context_text}, peer providers historically charged "
            f"between {market_min} and {market_max}. "
            f"Most common peer charge is {market_common}. "
        )

        if provider_common is not None:

            if provider_min is not None and provider_max is not None:
                provider_range_text = (
                    f"{provider.title()} historically charged between "
                    f"{provider_min} and {provider_max} for this context, "
                    f"with most common charge {provider_common}. "
                )
            else:
                provider_range_text = (
                    f"{provider.title()} most commonly charged "
                    f"{provider_common} for this context. "
                )

            if provider_position == "LOWER_THAN_PEERS":
                msg += (
                    provider_range_text
                    + "This provider is lower than peer providers. "
                )

            elif provider_position == "HIGHER_THAN_PEERS":
                msg += (
                    provider_range_text
                    + "This provider is higher than peer providers. "
                )

            else:
                msg += (
                    provider_range_text
                    + "This provider is similar to peer providers. "
                )

        else:
            msg += (
                f"No provider-specific price history available for "
                f"{provider.title()} in this context. "
            )

        if (
            market_min is not None
            and market_max is not None
        ):
            if actual < market_min:
                msg += (
                    f"Current submitted amount {actual} is below "
                    f"peer provider pricing range."
                )

            elif actual > market_max:
                msg += (
                    f"Current submitted amount {actual} is above "
                    f"peer provider pricing range."
                )

            else:
                msg += (
                    f"Current submitted amount {actual} is within "
                    f"peer provider pricing range."
                )

        return {
            "PROVIDER_PEER_PRICING": msg,
            "PROVIDER_PEER_PRICING_AVAILABLE": 1,
            "PROVIDER_MARKET_MIN": market_min,
            "PROVIDER_MARKET_MAX": market_max,
            "PROVIDER_MARKET_COMMON": market_common,
            "PROVIDER_USUAL_PRICE": provider_common,
            "PROVIDER_PROVIDER_MIN": provider_min,
            "PROVIDER_PROVIDER_MAX": provider_max,
            "PROVIDER_PRICE_POSITION": provider_position,
        }