import pandas as pd
import numpy as np
import os
import json
import math

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

print("Loading data...")
df = pd.read_csv(os.path.join(_ROOT, 'data', 'HDB_Resale_Prices.csv'))
df["years"] = df["remaining_lease"].str.extract(r"(\d+) years").astype(float).fillna(0)
df["months"] = df["remaining_lease"].str.extract(r"(\d+) month").astype(float).fillna(0)
df["remaining_lease_float"] = df["years"] + (df["months"] / 12)
df["storey_midpoint"] = df["storey_range"].str.split(" TO ", expand=True).astype(float).mean(axis=1)
df["transaction_year"] = df["month"].str.extract(r"(\d{4})").astype(int)
df["transaction_month"] = df["month"].str.extract(r"-(\d{2})").astype(int)
df['full_address'] = df['block'] + " " + df['street_name']

with open(os.path.join(_ROOT, "data", "caches", "onemap_cache.json"), "r") as f:
    address_to_coords = json.load(f)

df['lat'] = df['full_address'].map(lambda x: address_to_coords.get(x, (1.35, 103.8))[0])
df['lon'] = df['full_address'].map(lambda x: address_to_coords.get(x, (1.35, 103.8))[1])

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

CBD_COORD = (1.283933, 103.851463)
df['dist_to_cbd'] = df.apply(lambda row: haversine_distance(row['lat'], row['lon'], CBD_COORD[0], CBD_COORD[1]), axis=1)

mrt_stations = [(1.369933, 103.84958), (1.436058, 103.786053), (1.333152, 103.742286), (1.353266, 103.945143)]
df['dist_to_nearest_mrt'] = df.apply(lambda row: min([haversine_distance(row['lat'], row['lon'], m[0], m[1]) for m in mrt_stations]), axis=1)

mature_estates = ["ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH"]
df['is_mature_estate'] = df['town'].isin(mature_estates).astype(int)

with open(os.path.join(_ROOT, "data", "caches", "school_cache.json"), "r") as f:
    school_coords_data = json.load(f)
    school_points = [(s["lat"], s["lon"]) for s in school_coords_data if s["lat"] is not None]

    gep_schools = ["ANGLO-CHINESE SCHOOL (PRIMARY)", "CATHOLIC HIGH SCHOOL", "HENRY PARK PRIMARY SCHOOL", "NAN HUA PRIMARY SCHOOL", "NANYANG PRIMARY SCHOOL", "RAFFLES GIRLS' PRIMARY SCHOOL", "ROSYTH SCHOOL", "ST. HILDA'S PRIMARY SCHOOL", "TAO NAN SCHOOL"]
    elite_school_points = [(s["lat"], s["lon"]) for s in school_coords_data if s["lat"] is not None and s["name"] in gep_schools]

print(f"Total elite schools found matching: {len(elite_school_points)}")

def count_schools_within_1km(lat, lon):
    return sum(1 for s_lat, s_lon in school_points if haversine_distance(lat, lon, s_lat, s_lon) <= 1000)

df['num_schools_within_1km'] = df.apply(lambda row: count_schools_within_1km(row['lat'], row['lon']), axis=1)

if len(elite_school_points) > 0:
    df['dist_to_nearest_elite_school'] = df.apply(lambda row: min([haversine_distance(row['lat'], row['lon'], m[0], m[1]) for m in elite_school_points]), axis=1)
    df['elite_school_within_1km'] = (df['dist_to_nearest_elite_school'] <= 1000).astype(int)
    
    feature_columns = [
        "town", "flat_type", "storey_midpoint", "floor_area_sqm", "flat_model", 
        "remaining_lease_float", "transaction_year", "transaction_month",
        "dist_to_cbd", "dist_to_nearest_mrt", "is_mature_estate", "num_schools_within_1km",
        "dist_to_nearest_elite_school", "elite_school_within_1km"
    ]

    X = df[feature_columns]
    X_encoded = pd.get_dummies(X, columns=["flat_type", "flat_model"], drop_first=True)
    y = df['resale_price']

    X_temp, X_test, y_temp, y_test = train_test_split(X_encoded, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

    town_mean_prices = y_train.groupby(X_train['town']).mean()
    global_mean_price = y_train.mean()

    X_train_tenc = X_train.copy()
    X_val_tenc = X_val.copy()
    X_train_tenc['town_encoded'] = X_train['town'].map(town_mean_prices).fillna(global_mean_price)
    X_val_tenc['town_encoded'] = X_val['town'].map(town_mean_prices).fillna(global_mean_price)
    X_train_tenc.drop(columns=['town'], inplace=True)
    X_val_tenc.drop(columns=['town'], inplace=True)

    print("Training XGBoost with elite schools...")
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb.fit(X_train_tenc, y_train, eval_set=[(X_val_tenc, y_val)], verbose=False)
    xgb_pred = xgb.predict(X_val_tenc)

    xgb_rmse = np.sqrt(mean_squared_error(df.loc[X_val.index, 'resale_price'], xgb_pred))
    print(f"XGB RMSE WITH ELITE SCHOOL FEATURES: ${xgb_rmse:,.2f}")
else:
    print("NO ELITE SCHOOLS FOUND")
