import yaml
from pathlib import Path


class ConfigLoader:

    def __init__(self, config_path="config/config.yaml"):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}"
            )

        self.config = self._load_config()

    def _load_config(self):

        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        return config

    def get_config(self):
        return self.config

    def get_paths(self):
        return self.config.get("paths", {})

    def get_columns(self):
        return self.config.get("columns", {})

    def get_preprocessing(self):
        return self.config.get("preprocessing", {})

    def get_baseline(self):
        return self.config.get("baseline", {})

    def get_risk_scoring(self):
        return self.config.get("risk_scoring", {})

    def get_catboost(self):
        return self.config.get("catboost", {})

    def get_isolation_forest(self):
        return self.config.get("isolation_forest", {})


if __name__ == "__main__":

    loader = ConfigLoader()

    print("\n========== FULL CONFIG ==========")
    print(loader.get_config())

    print("\n========== PATHS ==========")
    print(loader.get_paths())

    print("\n========== COLUMNS ==========")
    print(loader.get_columns())