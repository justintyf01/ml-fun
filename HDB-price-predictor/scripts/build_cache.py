import pandas as pd
import requests
import json
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# token currently still not required
API_TOKEN = ""

print("==================================================")
print("  OneMap Cache Builder")
print("==================================================")

df = pd.read_csv(os.path.join(_ROOT, "data", "HDB_Resale_Prices.csv"))
df['full_address'] = df['block'] + " " + df['street_name']
unique_addresses = df['full_address'].unique()

print(f"Total unique addresses in dataset: {len(unique_addresses)}")

cache_file = os.path.join(_ROOT, "data", "caches", "onemap_cache.json")
address_to_coords = {}

if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        address_to_coords = json.load(f)
    print(f"Loaded existing cache with {len(address_to_coords)} addresses.")

missing_addresses = [a for a in unique_addresses if a not in address_to_coords]
print(f"Addresses left to fetch: {len(missing_addresses)}\n")

if not missing_addresses:
    print("Cache is already complete! No need to run.")
    exit()

def get_onemap_coordinates(address, token):
    """Hits the OneMap API to get lat/lon for a specific HDB block + street."""
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=N&pageNum=1"
    headers = {"Authorization": token} if token != "PUT_YOUR_TOKEN_HERE" else {}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        if 'error' in data:
            return None, None, data['error']
        if data.get('found', 0) > 0:
            lat = float(data['results'][0]['LATITUDE'])
            lon = float(data['results'][0]['LONGITUDE'])
            return lat, lon, None
    except Exception as e:
        return None, None, str(e)
    return None, None, "Not Found"


failed_count = 0
try:
    for i, address in enumerate(missing_addresses):
        lat, lon, err = get_onemap_coordinates(address, API_TOKEN)
        
        if lat is not None:
            address_to_coords[address] = (lat, lon)
        else:
            print(f"Failed to fetch {address}: {err}")
            failed_count += 1
            if failed_count > 10:
                print("\n[!] Too many consecutive failures. The API might be hard blocking you.")
                print("If you are getting 'Authentication token missing', you MUST provide the API_TOKEN at the top of the file.")
                break
                
        # API Rate Limit: 250 calls per minute = ~4.1 calls per second.
        # Sleeping for 0.25 seconds ensures we stay comfortably under the limit (max 240/min).
        time.sleep(0.25)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} / {len(missing_addresses)} blocks...")
            # Periodically save progress so we don't lose data if script crashes
            with open(cache_file, "w") as f:
                json.dump(address_to_coords, f)

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving current progress...")

# Final save
with open(cache_file, "w") as f:
    json.dump(address_to_coords, f)

print(f"\nSaved {len(address_to_coords)} coordinates to {cache_file}.")
if len(address_to_coords) < len(unique_addresses):
    print("Run the script again later to resume fetching the missing addresses.")
else:
    print("SUCCESS: Cache is 100% complete!")
    print("You can now remove the mock limit inside `models.py` and run your models.")

