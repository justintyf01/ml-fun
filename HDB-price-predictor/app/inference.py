"""
Single-row feature engineering and prediction pipeline.
Replicates the training pipeline from features_v2.py / features_v3.py
for a single prediction request.
"""
import math
import re

import numpy as np
import pandas as pd

from app.model_loader import ModelArtifacts
from app.schemas import PredictionRequest, PredictionResponse, PredictionRange, FeaturesSummary


# ── Constants (mirrored from features_v2.py) ────────────────

CBD_COORD = (1.283933, 103.851463)

REGIONAL_HUBS = [
    (1.369933, 103.84958),
    (1.436058, 103.786053),
    (1.333152, 103.742286),
    (1.353266, 103.945143),
]

MATURE_ESTATES = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE",
    "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH",
}

GEP_SCHOOLS = {
    "ANGLO-CHINESE SCHOOL (PRIMARY)", "CATHOLIC HIGH SCHOOL",
    "HENRY PARK PRIMARY SCHOOL", "NAN HUA PRIMARY SCHOOL",
    "NANYANG PRIMARY SCHOOL", "RAFFLES GIRLS' PRIMARY SCHOOL",
    "ROSYTH SCHOOL", "ST. HILDA'S PRIMARY SCHOOL", "TAO NAN SCHOOL",
}

CATEGORICAL_FEATURES = ["flat_type", "flat_model"]
TARGET_ENCODE_FEATURES = ["town", "street_name"]


# ── Haversine (mirrored from utils.py) ──────────────────────

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_distance(lat, lon, points):
    if not points:
        return 99999.0
    return min(haversine_distance(lat, lon, p[0], p[1]) for p in points)


def _count_within(lat, lon, points, radius_m):
    return sum(1 for p in points if haversine_distance(lat, lon, p[0], p[1]) <= radius_m)


# ── Feature engineering ─────────────────────────────────────

def build_features_single(req: PredictionRequest, artifacts: ModelArtifacts) -> dict:
    """
    Build all raw features for a single prediction request.
    Returns a dict of feature values (before encoding).
    """
    # ── Base features ───────────────────────────────────────
    years_match = re.search(r"(\d+)\s*year", req.remaining_lease)
    months_match = re.search(r"(\d+)\s*month", req.remaining_lease)
    years = float(years_match.group(1)) if years_match else 0.0
    months = float(months_match.group(1)) if months_match else 0.0
    remaining_lease_float = years + months / 12

    storey_parts = req.storey_range.split(" TO ")
    storey_midpoint = (float(storey_parts[0]) + float(storey_parts[1])) / 2

    year_match = re.match(r"(\d{4})", req.month)
    month_match = re.search(r"-(\d{2})", req.month)
    transaction_year = int(year_match.group(1)) if year_match else 2024
    transaction_month = int(month_match.group(1)) if month_match else 1

    lease_below_60 = max(0, 60 - remaining_lease_float)
    lease_below_40 = max(0, 40 - remaining_lease_float)
    building_age = transaction_year - req.lease_commence_date

    # ── Location features ───────────────────────────────────
    # Prefer coordinates supplied by the client (from OneMap geocoding).
    # Fall back to pre-built cache; last resort: Singapore centre.
    if req.lat is not None and req.lon is not None:
        lat, lon = float(req.lat), float(req.lon)
    else:
        full_address = f"{req.block} {req.street_name}"
        coords = artifacts.onemap_cache.get(full_address, [1.35, 103.8])
        lat, lon = float(coords[0]), float(coords[1])

    dist_to_cbd = haversine_distance(lat, lon, CBD_COORD[0], CBD_COORD[1])
    dist_to_nearest_hub = min(
        haversine_distance(lat, lon, h[0], h[1]) for h in REGIONAL_HUBS
    )
    _nearest_mrt = min(artifacts.mrt_stations, key=lambda m: haversine_distance(lat, lon, m[0], m[1]))
    dist_to_actual_nearest_mrt = haversine_distance(lat, lon, _nearest_mrt[0], _nearest_mrt[1])
    nearest_mrt_name = _nearest_mrt[2]

    # ── Amenity features ────────────────────────────────────
    is_mature_estate = 1 if req.town in MATURE_ESTATES else 0

    valid_schools = [s for s in artifacts.school_data if s["lat"] is not None]
    school_points = [(s["lat"], s["lon"]) for s in valid_schools]
    elite_points = [
        (s["lat"], s["lon"]) for s in valid_schools if s["name"] in GEP_SCHOOLS
    ]
    num_schools_within_1km = _count_within(lat, lon, school_points, 1000)
    dist_to_nearest_elite_school = _nearest_distance(lat, lon, elite_points)
    elite_school_within_1km = 1 if dist_to_nearest_elite_school <= 1000 else 0

    primary_schools   = [s for s in valid_schools if "PRIMARY" in s["name"]]
    secondary_schools = [s for s in valid_schools if "SECONDARY" in s["name"]]

    def _nearest_school_label(schools):
        if not schools:
            return "N/A"
        nearest = min(schools, key=lambda s: haversine_distance(lat, lon, s["lat"], s["lon"]))
        dist_m  = haversine_distance(lat, lon, nearest["lat"], nearest["lon"])
        # Drop redundant level/type words; the card label already says Pri/Sec
        import re
        short = re.sub(r"\s*\(\s*(PRIMARY|SECONDARY)\s*\)", "", nearest["name"], flags=re.IGNORECASE)
        short = re.sub(r"\b(PRIMARY|SECONDARY)\s+SCHOOL\b", "SCHOOL", short, flags=re.IGNORECASE)
        short = re.sub(r"\bSCHOOL\b", "", short, flags=re.IGNORECASE)
        short = re.sub(r"\b(PRIMARY|SECONDARY)\b", "", short, flags=re.IGNORECASE)
        short = re.sub(r"\s{2,}", " ", short).strip().title()
        return f"{short} · {dist_m/1000:.1f} km"

    nearest_primary_school   = _nearest_school_label(primary_schools)
    nearest_secondary_school = _nearest_school_label(secondary_schools)

    dist_to_nearest_hawker = _nearest_distance(lat, lon, artifacts.hawker_points)
    num_hawkers_within_1km = _count_within(lat, lon, artifacts.hawker_points, 1000)

    dist_to_nearest_mall = _nearest_distance(lat, lon, artifacts.mall_points)
    num_malls_within_2km = _count_within(lat, lon, artifacts.mall_points, 2000)

    dist_to_nearest_park = _nearest_distance(lat, lon, artifacts.park_points)

    # ── Phase 4 features ────────────────────────────────────
    area_x_storey = req.floor_area_sqm * storey_midpoint
    lease_x_area = remaining_lease_float * req.floor_area_sqm
    lease_x_storey = remaining_lease_float * storey_midpoint
    is_high_floor = 1 if storey_midpoint >= 20 else 0
    is_ground_floor = 1 if storey_midpoint <= 3 else 0

    # Aggregate lookups
    vol_key = f"{req.town}|{req.month}"
    town_month_volume = artifacts.aggregate_lookups["town_month_volume"].get(
        vol_key, artifacts.aggregate_lookups["fallback_volume"]
    )

    med_key = f"{req.town}|{transaction_year}"
    town_year_median_price = artifacts.aggregate_lookups["town_year_median_price"].get(
        med_key, artifacts.aggregate_lookups["fallback_median"]
    )

    return {
        "town": req.town,
        "flat_type": req.flat_type,
        "storey_midpoint": storey_midpoint,
        "floor_area_sqm": req.floor_area_sqm,
        "flat_model": req.flat_model,
        "remaining_lease_float": remaining_lease_float,
        "transaction_year": transaction_year,
        "transaction_month": transaction_month,
        "dist_to_cbd": dist_to_cbd,
        "dist_to_nearest_hub": dist_to_nearest_hub,
        "dist_to_actual_nearest_mrt": dist_to_actual_nearest_mrt,
        "is_mature_estate": is_mature_estate,
        "num_schools_within_1km": num_schools_within_1km,
        "dist_to_nearest_elite_school": dist_to_nearest_elite_school,
        "elite_school_within_1km": elite_school_within_1km,
        "nearest_primary_school": nearest_primary_school,
        "nearest_secondary_school": nearest_secondary_school,
        "dist_to_nearest_hawker": dist_to_nearest_hawker,
        "num_hawkers_within_1km": num_hawkers_within_1km,
        "dist_to_nearest_mall": dist_to_nearest_mall,
        "num_malls_within_2km": num_malls_within_2km,
        "dist_to_nearest_park": dist_to_nearest_park,
        "nearest_mrt_name": nearest_mrt_name,
        "lease_below_60": lease_below_60,
        "lease_below_40": lease_below_40,
        "building_age": building_age,
        "lat": lat,
        "lon": lon,
        "area_x_storey": area_x_storey,
        "lease_x_area": lease_x_area,
        "lease_x_storey": lease_x_storey,
        "is_high_floor": is_high_floor,
        "is_ground_floor": is_ground_floor,
        "town_month_volume": town_month_volume,
        "town_year_median_price": town_year_median_price,
        "street_name": req.street_name,
    }


def predict(req: PredictionRequest, artifacts: ModelArtifacts) -> PredictionResponse:
    """Run the full inference pipeline: features → encode → predict across 5 folds."""
    raw_features = build_features_single(req, artifacts)

    fold_predictions = []

    for fold_i in range(len(artifacts.models)):
        row = dict(raw_features)
        te_maps = artifacts.target_encodings[fold_i]

        # Apply target encoding
        for col in TARGET_ENCODE_FEATURES:
            col_map = te_maps[col]["map"]
            global_mean = te_maps[col]["global_mean"]
            row[f"{col}_encoded"] = col_map.get(str(row[col]), global_mean)

        # Remove original string columns
        for col in TARGET_ENCODE_FEATURES:
            del row[col]

        # Build single-row DataFrame
        df = pd.DataFrame([row])

        # One-hot encode
        df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

        # Reindex to match training columns
        df = df.reindex(columns=artifacts.ohe_columns, fill_value=0)

        pred = float(artifacts.models[fold_i].predict(df)[0])
        fold_predictions.append(pred)

    mean_pred = int(round(np.mean(fold_predictions)))
    low_pred = int(round(mean_pred * 0.95))
    high_pred = int(round(mean_pred * 1.05))

    return PredictionResponse(
        predicted_price=mean_pred,
        prediction_range=PredictionRange(low=low_pred, high=high_pred),
        features_summary=FeaturesSummary(
            dist_to_cbd_km=round(raw_features["dist_to_cbd"] / 1000, 2),
            dist_to_nearest_mrt_m=round(raw_features["dist_to_actual_nearest_mrt"], 0),
            nearest_mrt=raw_features["nearest_mrt_name"],
            nearest_primary_school=raw_features["nearest_primary_school"],
            nearest_secondary_school=raw_features["nearest_secondary_school"],
            remaining_lease_years=round(raw_features["remaining_lease_float"], 2),
            is_mature_estate=bool(raw_features["is_mature_estate"]),
            building_age=raw_features["building_age"],
        ),
    )
