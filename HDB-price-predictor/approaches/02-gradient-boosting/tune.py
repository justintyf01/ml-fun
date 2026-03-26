import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import math
import requests
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading data and applying v1.5 features...")

# 1. Load Data & Basic Cleaning
df = pd.read_csv(os.path.join(_ROOT, 'data', 'HDB_Resale_Prices.csv'))
df["years"] = df["remaining_lease"].str.extract(r"(\d+) years").astype(float).fillna(0)
df["months"] = df["remaining_lease"].str.extract(r"(\d+) month").astype(float).fillna(0)
df["remaining_lease_float"] = df["years"] + (df["months"] / 12)
df["storey_midpoint"] = df["storey_range"].str.split(" TO ", expand=True).astype(float).mean(axis=1)
df["transaction_year"] = df["month"].str.extract(r"(\d{4})").astype(int)
df["transaction_month"] = df["month"].str.extract(r"-(\d{2})").astype(int)

# 2. Add Location Features (Using cached OneMap coords to save time)
df['full_address'] = df['block'] + " " + df['street_name']

cache_file = os.path.join(_ROOT, "data", "caches", "onemap_cache.json")
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        address_to_coords = json.load(f)
else:
    raise Exception("onemap_cache.json not found! Run models.py first to build the cache.")

df['lat'] = df['full_address'].map(lambda x: address_to_coords.get(x, (1.35, 103.8))[0])
df['lon'] = df['full_address'].map(lambda x: address_to_coords.get(x, (1.35, 103.8))[1])

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# CBD
CBD_COORD = (1.283933, 103.851463)
df['dist_to_cbd'] = df.apply(lambda row: haversine_distance(row['lat'], row['lon'], CBD_COORD[0], CBD_COORD[1]), axis=1)

# MRTs
mrt_stations = [(1.369933, 103.84958), (1.436058, 103.786053), (1.333152, 103.742286), (1.353266, 103.945143)]
df['dist_to_nearest_mrt'] = df.apply(lambda row: min([haversine_distance(row['lat'], row['lon'], m[0], m[1]) for m in mrt_stations]), axis=1)

# Estate & School
mature_estates = ["ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH"]
df['is_mature_estate'] = df['town'].isin(mature_estates).astype(int)

# Load actual school coordinates from cache
if os.path.exists(os.path.join(_ROOT, "data", "caches", "school_cache.json")):
    with open(os.path.join(_ROOT, "data", "caches", "school_cache.json"), "r") as f:
        school_coords_data = json.load(f)
        school_points = [(s["lat"], s["lon"]) for s in school_coords_data if s["lat"] is not None]
else:
    raise FileNotFoundError("school_cache.json not found! Run the standalone script to build the school cache first.")

def count_schools_within_1km(lat, lon):
    return sum(1 for s_lat, s_lon in school_points if haversine_distance(lat, lon, s_lat, s_lon) <= 1000)

df['num_schools_within_1km'] = df.apply(lambda row: count_schools_within_1km(row['lat'], row['lon']), axis=1)

# 3. Features & Encoding
feature_columns = [
    "town", "flat_type", "storey_midpoint", "floor_area_sqm", "flat_model", 
    "remaining_lease_float", "transaction_year", "transaction_month",
    "dist_to_cbd", "dist_to_nearest_mrt", "is_mature_estate", "num_schools_within_1km"
]
categorical_features = ["flat_type", "flat_model"] # Town is target encoded
y = df["resale_price"]
X = df[feature_columns]
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# 4. Splits & Target Encoding Town
X_temp, X_test, y_temp, y_test = train_test_split(X_encoded, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

town_mean_prices = y_train.groupby(X_train['town']).mean()
global_mean_price = y_train.mean()

for split in [X_train, X_val, X_test]:
    split['town_encoded'] = split['town'].map(town_mean_prices).fillna(global_mean_price)
    split.drop(columns=['town'], inplace=True)

# We'll combine train/val for Cross Validation
X_cv = pd.concat([X_train, X_val])
y_cv = pd.concat([y_train, y_val])

print(f"Dataset ready. Starting GridSearchCV on XGBoost... (CV samples: {len(X_cv)})")

# 5. XGBoost Hyperparameter Tuning
# Trying a focused grid around the defaults / known good Tree parameters
param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [6, 8, 10],            # Deeper trees can capture complex location interactions
    'min_child_weight': [1, 3],         # Controls overfitting in deep trees
    'subsample': [0.8, 1.0],            # 1.0 is default, 0.8 adds randomness
    'colsample_bytree': [0.8, 1.0], 
    'n_estimators': [500]               # Fixed to 500 to keep search time reasonable
}

xgb = XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')

# 3-fold cross validation
grid_search = GridSearchCV(
    estimator=xgb, 
    param_grid=param_grid, 
    scoring='neg_root_mean_squared_error', 
    cv=3,
    verbose=2,
    n_jobs=1 # run sequentially to save ram, inner parallel happens via n_jobs=-1 in XGB
)

grid_search.fit(X_cv, y_cv)

print("\n" + "="*50)
print("  TUNING COMPLETE")
print("="*50)

best_rmse = -grid_search.best_score_
print(f"\nBest CV RMSE: ${best_rmse:,.2f}")
print("Best parameters found:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")

# Test on the completely unseen holdout TEST set to verify final generalization
print("\nEvaluating on Test Set...")
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Final TEST Set RMSE: ${test_rmse:,.2f}")
print(f"Final test set R^2: {best_model.score(X_test, y_test):.4f}")
