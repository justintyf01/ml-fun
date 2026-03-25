"""Load all model artifacts at application startup."""
import json
import os

import pandas as pd
import xgboost as xgb

N_FOLDS = 5


class ModelArtifacts:
    """Container for all loaded artifacts needed for inference."""

    def __init__(self, artifacts_dir: str = "model_artifacts"):
        self.artifacts_dir = artifacts_dir
        self.models: list[xgb.XGBRegressor] = []
        self.target_encodings: list[dict] = []
        self.ohe_columns: list[str] = []
        self.aggregate_lookups: dict = {}
        self.reference_data: dict = {}
        self.onemap_cache: dict = {}
        self.school_data: list[dict] = []
        self.hawker_points: list[tuple] = []
        self.mall_points: list[tuple] = []
        self.park_points: list[tuple] = []
        self.mrt_coords: list[tuple] = []

    def load(self):
        """Load all artifacts from disk."""
        print("Loading model artifacts...")

        # Load XGBoost models (5 folds)
        for i in range(N_FOLDS):
            fold_dir = os.path.join(self.artifacts_dir, f"fold_{i}")
            model = xgb.XGBRegressor()
            model.load_model(os.path.join(fold_dir, "xgboost.json"))
            self.models.append(model)

            with open(os.path.join(fold_dir, "target_encodings.json")) as f:
                self.target_encodings.append(json.load(f))

        print(f"  Loaded {len(self.models)} XGBoost models")

        # OHE columns
        with open(os.path.join(self.artifacts_dir, "ohe_columns.json")) as f:
            self.ohe_columns = json.load(f)

        # Aggregate lookups
        with open(os.path.join(self.artifacts_dir, "aggregate_lookups.json")) as f:
            self.aggregate_lookups = json.load(f)

        # Reference data
        with open(os.path.join(self.artifacts_dir, "reference_data.json")) as f:
            self.reference_data = json.load(f)

        # Caches
        caches_dir = os.path.join(self.artifacts_dir, "caches")

        with open(os.path.join(caches_dir, "onemap_cache.json")) as f:
            self.onemap_cache = json.load(f)
        print(f"  Loaded {len(self.onemap_cache):,} cached addresses")

        with open(os.path.join(caches_dir, "school_cache.json")) as f:
            self.school_data = json.load(f)

        self.hawker_points = self._load_point_cache(os.path.join(caches_dir, "hawker_cache.json"))
        self.mall_points = self._load_point_cache(os.path.join(caches_dir, "mall_cache.json"))
        self.park_points = self._load_point_cache(os.path.join(caches_dir, "park_cache.json"))

        mrt_path = os.path.join(caches_dir, "MRT Stations.csv")
        mrt_df = pd.read_csv(mrt_path)
        self.mrt_coords = list(zip(mrt_df["Latitude"], mrt_df["Longitude"]))
        print(f"  Loaded {len(self.mrt_coords)} MRT/LRT stations")

        print("All artifacts loaded successfully.")

    @staticmethod
    def _load_point_cache(path: str) -> list[tuple]:
        with open(path) as f:
            data = json.load(f)
        return [(d["lat"], d["lon"]) for d in data if d.get("lat") is not None]
