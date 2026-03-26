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

def count_schools_within_1km(lat, lon):
    return sum(1 for s_lat, s_lon in school_points if haversine_distance(lat, lon, s_lat, s_lon) <= 1000)

df['num_schools_within_1km'] = df.apply(lambda row: count_schools_within_1km(row['lat'], row['lon']), axis=1)

# Engineered features
df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']

feature_columns = [
    "town", "flat_type", "storey_midpoint", "floor_area_sqm", "flat_model", 
    "remaining_lease_float", "transaction_year", "transaction_month",
    "dist_to_cbd", "dist_to_nearest_mrt", "is_mature_estate", "num_schools_within_1km"
]

X = df[feature_columns]
X_encoded = pd.get_dummies(X, columns=["flat_type", "flat_model"], drop_first=True)

# Test 1: Baseline
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

xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1, early_stopping_rounds=50)
xgb.fit(X_train_tenc, y_train, eval_set=[(X_val_tenc, y_val)], verbose=False)
baseline_pred = xgb.predict(X_val_tenc)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
print(f"Baseline XGBoost RMSE: ${baseline_rmse:,.2f}")

# Test 2: Predict Price Per Sqm instead
y_psm = df['price_per_sqm']
X_train, X_val, y_train_psm, y_val_psm = train_test_split(X_temp, y_psm.loc[X_temp.index], test_size=0.176, random_state=42)

town_mean_psm = y_train_psm.groupby(X_train['town']).mean()
global_mean_psm = y_train_psm.mean()

X_train_tenc_psm = X_train.copy()
X_val_tenc_psm = X_val.copy()
X_train_tenc_psm['town_encoded'] = X_train['town'].map(town_mean_psm).fillna(global_mean_psm)
X_val_tenc_psm['town_encoded'] = X_val['town'].map(town_mean_psm).fillna(global_mean_psm)
X_train_tenc_psm.drop(columns=['town'], inplace=True)
X_val_tenc_psm.drop(columns=['town'], inplace=True)

xgb_psm = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1, early_stopping_rounds=50)
xgb_psm.fit(X_train_tenc_psm, y_train_psm, eval_set=[(X_val_tenc_psm, y_val_psm)], verbose=False)
psm_pred = xgb_psm.predict(X_val_tenc_psm)

# convert back to total price
final_pred_from_psm = psm_pred * X_val['floor_area_sqm']
psm_rmse = np.sqrt(mean_squared_error(y_val, final_pred_from_psm))
print(f"Predicting Price per Sqm RMSE: ${psm_rmse:,.2f}")

# Test 3: Log Transform
y_log = np.log(df['resale_price'])
X_train, X_val, y_train_log, y_val_log = train_test_split(X_temp, y_log.loc[X_temp.index], test_size=0.176, random_state=42)
town_mean_log = y_train_log.groupby(X_train['town']).mean()
global_mean_log = y_train_log.mean()

X_train_tenc_log = X_train.copy()
X_val_tenc_log = X_val.copy()
X_train_tenc_log['town_encoded'] = X_train['town'].map(town_mean_log).fillna(global_mean_log)
X_val_tenc_log['town_encoded'] = X_val['town'].map(town_mean_log).fillna(global_mean_log)
X_train_tenc_log.drop(columns=['town'], inplace=True)
X_val_tenc_log.drop(columns=['town'], inplace=True)

xgb_log = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1, early_stopping_rounds=50)
xgb_log.fit(X_train_tenc_log, y_train_log, eval_set=[(X_val_tenc_log, y_val_log)], verbose=False)
log_pred = xgb_log.predict(X_val_tenc_log)

# convert back
final_pred_from_log = np.exp(log_pred)
log_rmse = np.sqrt(mean_squared_error(y_val, final_pred_from_log))
print(f"Predicting Log(Price) RMSE: ${log_rmse:,.2f}")
