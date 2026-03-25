"""
Update onemap_cache.json with coordinates for any new addresses
found in the expanded dataset (hdb_prices_2017.csv).

Reads the existing cache, identifies missing addresses, and fetches
coordinates from the OneMap API. Supports resume on interruption.
"""
import pandas as pd
import json
import os
import time
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

API_TOKEN = ""
CACHE_FILE = os.path.join(_ROOT, "data", "caches", "onemap_cache.json")
DATASET_PATH = os.path.join(_ROOT, "data", "hdb_prices_2017.csv")
FALLBACK_PATH = os.path.join(_ROOT, "data", "HDB_Resale_Prices.csv")


def get_onemap_coordinates(address, token=""):
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=N&pageNum=1"
    headers = {"Authorization": token} if token else {}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        if data.get("found", 0) > 0:
            lat = float(data["results"][0]["LATITUDE"])
            lon = float(data["results"][0]["LONGITUDE"])
            return lat, lon, None
    except Exception as e:
        return None, None, str(e)
    return None, None, "Not Found"


def main():
    print("=" * 50)
    print("  OneMap Cache Builder v2")
    print("=" * 50)

    # Load dataset
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        print(f"Using expanded dataset: {len(df):,} rows")
    elif os.path.exists(FALLBACK_PATH):
        df = pd.read_csv(FALLBACK_PATH)
        print(f"Using fallback dataset: {len(df):,} rows")
    else:
        print("No dataset found!")
        return

    df["full_address"] = df["block"] + " " + df["street_name"]
    unique_addresses = df["full_address"].unique()
    print(f"Unique addresses: {len(unique_addresses):,}")

    # Load existing cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        print(f"Existing cache entries: {len(cache):,}")

    missing = [a for a in unique_addresses if a not in cache]
    print(f"Missing addresses to fetch: {len(missing):,}\n")

    if not missing:
        print("Cache is complete!")
        return

    failed_count = 0
    try:
        for i, address in enumerate(missing):
            lat, lon, err = get_onemap_coordinates(address, API_TOKEN)

            if lat is not None:
                cache[address] = (lat, lon)
                failed_count = 0
            else:
                print(f"  Failed: {address} ({err})")
                failed_count += 1
                if failed_count > 20:
                    print("\nToo many consecutive failures. Run again later to resume.")
                    break

            time.sleep(0.25)

            if (i + 1) % 100 == 0:
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f)
                print(f"  Processed {i + 1} / {len(missing)} ({len(cache):,} total cached)")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    print(f"\nSaved {len(cache):,} entries to {CACHE_FILE}")

    remaining = len([a for a in unique_addresses if a not in cache])
    if remaining:
        print(f"Still missing: {remaining:,}. Run again to continue.")
    else:
        print("Cache is 100% complete!")


if __name__ == "__main__":
    main()
