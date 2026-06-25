import joblib
from pathlib import Path

from src.config_loader import ConfigLoader
from src.provider_fingerprint_builder import (
    ProviderFingerprintBuilder
)


def main():

    print("\n======================================")
    print("BUILDING PROVIDER FINGERPRINT")
    print("======================================")

    config = ConfigLoader().get_config()

    df = joblib.load(
        config["paths"]["processed_data"]
    )

    builder = ProviderFingerprintBuilder()

    artifact = builder.build(df)

    output_dir = (
        Path(config["paths"]["models_dir"]).parent
        / "intelligence"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "provider_fingerprint_artifact.pkl"
    )

    joblib.dump(
        artifact,
        output_path
    )

    print(
        f"\nProviders profiled: {len(artifact)}"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()