import pandas as pd
import numpy as np
import os
import json
import math

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
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

def count_schools_within_1km(lat, lon):
    return sum(1 for s_lat, s_lon in school_points if haversine_distance(lat, lon, s_lat, s_lon) <= 1000)

df['num_schools_within_1km'] = df.apply(lambda row: count_schools_within_1km(row['lat'], row['lon']), axis=1)

feature_columns = [
    "town", "flat_type", "storey_midpoint", "floor_area_sqm", "flat_model", 
    "remaining_lease_float", "transaction_year", "transaction_month",
    "dist_to_cbd", "dist_to_nearest_mrt", "is_mature_estate", "num_schools_within_1km"
]

X = df[feature_columns]
X_encoded = pd.get_dummies(X, columns=["flat_type", "flat_model"], drop_first=True)

y = df['resale_price'] / df['floor_area_sqm'] # target is now price per sqm

X_temp, X_test, y_temp, y_test = train_test_split(X_encoded, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

town_mean_prices = y_train.groupby(X_train['town']).mean()
global_mean_price = y_train.mean()

X_train_tenc = X_train.copy()
X_val_tenc = X_val.copy()
X_test_tenc = X_test.copy()
X_train_tenc['town_encoded'] = X_train['town'].map(town_mean_prices).fillna(global_mean_price)
X_val_tenc['town_encoded'] = X_val['town'].map(town_mean_prices).fillna(global_mean_price)
X_test_tenc['town_encoded'] = X_test['town'].map(town_mean_prices).fillna(global_mean_price)
X_train_tenc.drop(columns=['town'], inplace=True)
X_val_tenc.drop(columns=['town'], inplace=True)
X_test_tenc.drop(columns=['town'], inplace=True)

# Train XGB
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, early_stopping_rounds=50)
xgb.fit(X_train_tenc, y_train, eval_set=[(X_val_tenc, y_val)], verbose=False)
xgb_pred = xgb.predict(X_val_tenc)

# Train LGBM
lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1)
lgbm.fit(X_train_tenc, y_train, eval_set=[(X_val_tenc, y_val)])
lgbm_pred = lgbm.predict(X_val_tenc)

cat = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=8, random_seed=42, verbose=False, early_stopping_rounds=50)
cat.fit(X_train_tenc, y_train, eval_set=(X_val_tenc, y_val))
cat_pred = cat.predict(X_val_tenc)

xgb_rmse = np.sqrt(mean_squared_error(df.loc[X_val.index, 'resale_price'], xgb_pred * X_val['floor_area_sqm']))
lgbm_rmse = np.sqrt(mean_squared_error(df.loc[X_val.index, 'resale_price'], lgbm_pred * X_val['floor_area_sqm']))
cat_rmse = np.sqrt(mean_squared_error(df.loc[X_val.index, 'resale_price'], cat_pred * X_val['floor_area_sqm']))

print(f"XGB RMSE: ${xgb_rmse:,.2f}")
print(f"LGBM RMSE: ${lgbm_rmse:,.2f}")
print(f"CAT RMSE: ${cat_rmse:,.2f}")

ensemble_pred = (xgb_pred + lgbm_pred + cat_pred) / 3
ensemble_rmse = np.sqrt(mean_squared_error(df.loc[X_val.index, 'resale_price'], ensemble_pred * X_val['floor_area_sqm']))
print(f"ENSEMBLE RMSE: ${ensemble_rmse:,.2f}")

weighted_pred = (0.5 * xgb_pred) + (0.3 * lgbm_pred) + (0.2 * cat_pred)
weighted_rmse = np.sqrt(mean_squared_error(df.loc[X_val.index, 'resale_price'], weighted_pred * X_val['floor_area_sqm']))
print(f"WEIGHTED ENSEMBLE RMSE: ${weighted_rmse:,.2f}")

