"""Load all model artifacts at application startup."""
import json
import os

import pandas as pd
import xgboost as xgb

BEST_FOLD = 3  # fold_3 had the highest validation R² (0.9799)


class ModelArtifacts:
    """Container for all loaded artifacts needed for inference."""

    def __init__(self, artifacts_dir: str = "model_artifacts"):
        self.artifacts_dir = artifacts_dir
        self.models: list[xgb.XGBRegressor] = []
        self.target_encodings: list[dict] = []
        self.ohe_columns: list[str] = []
        self.aggregate_lookups: dict = {}
        self.reference_data: dict = {}
        self.block_lookup: dict = {}
        self.street_to_town: dict = {}
        self.onemap_cache: dict = {}
        self.school_data: list[dict] = []
        self.hawker_points: list[tuple] = []
        self.mall_points: list[tuple] = []
        self.park_points: list[tuple] = []
        self.mrt_stations: list[tuple] = []  # (lat, lon, display_name) e.g. "EW8/CC9 Paya Lebar"
        self.address_town_points: list[tuple] = []  # (lat, lon, town) for nearest-town lookup

    def load(self):
        """Load all artifacts from disk."""
        print("Loading model artifacts...")

        # Load single best fold
        fold_dir = os.path.join(self.artifacts_dir, f"fold_{BEST_FOLD}")
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(fold_dir, "xgboost.json"))
        self.models.append(model)

        with open(os.path.join(fold_dir, "target_encodings.json")) as f:
            self.target_encodings.append(json.load(f))

        print(f"  Loaded fold_{BEST_FOLD} (best validation R²)")

        # OHE columns
        with open(os.path.join(self.artifacts_dir, "ohe_columns.json")) as f:
            self.ohe_columns = json.load(f)

        # Aggregate lookups
        with open(os.path.join(self.artifacts_dir, "aggregate_lookups.json")) as f:
            self.aggregate_lookups = json.load(f)

        # Reference data
        with open(os.path.join(self.artifacts_dir, "reference_data.json")) as f:
            self.reference_data = json.load(f)

        # Block lookup (for frontend auto-fill)
        block_lookup_path = os.path.join(self.artifacts_dir, "block_lookup.json")
        if os.path.exists(block_lookup_path):
            with open(block_lookup_path) as f:
                self.block_lookup = json.load(f)
            print(f"  Loaded {len(self.block_lookup):,} blocks in lookup")

        street_to_town_path = os.path.join(self.artifacts_dir, "street_to_town.json")
        if os.path.exists(street_to_town_path):
            with open(street_to_town_path) as f:
                self.street_to_town = json.load(f)

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
        for _, row in mrt_df.iterrows():
            clean_name = (str(row["STN_NAME"])
                          .replace(" MRT STATION", "")
                          .replace(" LRT STATION", "")
                          .title())
            display = f"{row['STN_NO']} {clean_name}"
            self.mrt_stations.append((float(row["Latitude"]), float(row["Longitude"]), display))
        print(f"  Loaded {len(self.mrt_stations)} MRT/LRT stations")

        # Precompute (lat, lon, town) for nearest-town lookups
        for key, coords in self.onemap_cache.items():
            if key in self.block_lookup:
                self.address_town_points.append((
                    float(coords[0]), float(coords[1]),
                    self.block_lookup[key]["town"],
                ))
        print(f"  Precomputed {len(self.address_town_points):,} address-town points")

        print("All artifacts loaded successfully.")

    @staticmethod
    def _load_point_cache(path: str) -> list[tuple]:
        with open(path) as f:
            data = json.load(f)
        return [(d["lat"], d["lon"]) for d in data if d.get("lat") is not None]
