import pandas as pd
import requests
import json
import time
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
This script builds a cache of school coordinates by querying the OneMap API.
It uses the postal code to find the latitude and longitude of each school.
https://www.onemap.gov.sg/apidocs/search
"""

df = pd.read_csv(os.path.join(_ROOT, "data", "school_data.csv"))

school_coords = []
print(f"Total schools to process: {len(df)}")

failed = 0
for idx, row in df.iterrows():
    name = row['school_name']
    postal = str(row['postal_code']).zfill(6) # Ensure 6 digit postal code
    
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={postal}&returnGeom=Y&getAddrDetails=N&pageNum=1"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('found', 0) > 0:
            lat = float(data['results'][0]['LATITUDE'])
            lon = float(data['results'][0]['LONGITUDE'])
            school_coords.append({"name": name, "lat": lat, "lon": lon, "postal_code": postal})
            print(f"[{idx+1}/{len(df)}] SUCCESS: {name} ({postal}) -> {lat}, {lon}")
        else:
            print(f"[{idx+1}/{len(df)}] NOT FOUND: {name} ({postal})")
            failed += 1
            school_coords.append({"name": name, "lat": None, "lon": None, "postal_code": postal})
            
    except Exception as e:
        print(f"[{idx+1}/{len(df)}] ERROR: {name} - {str(e)}")
        failed += 1
        school_coords.append({"name": name, "lat": None, "lon": None, "postal_code": postal})
        
    time.sleep(0.3)

with open(os.path.join(_ROOT, "data", "caches", "school_cache.json"), "w") as f:
    json.dump(school_coords, f, indent=4)

print(f"\nDone! Saved to school_cache.json. Missing/Failed: {failed}")
