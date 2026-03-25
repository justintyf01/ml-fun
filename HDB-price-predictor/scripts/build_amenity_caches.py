"""
Build amenity caches: hawker centres, shopping malls, parks.
Hawker centres from GeoJSON. Malls and parks geocoded via OneMap.
Outputs: data/caches/hawker_cache.json, mall_cache.json, park_cache.json
"""
import json
import os
import time
import urllib.request
import urllib.parse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHES = os.path.join(_ROOT, "data", "caches")


def get_onemap_coordinates(search_val):
    """Geocode via OneMap API."""
    encoded = urllib.parse.quote(search_val)
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={encoded}&returnGeom=Y&getAddrDetails=N&pageNum=1"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data.get("found", 0) > 0:
            return float(data["results"][0]["LATITUDE"]), float(data["results"][0]["LONGITUDE"])
    except Exception as e:
        print(f"  Error: {search_val} — {e}")
    return None, None


def build_hawker_cache():
    """Parse hawker centres from GeoJSON (already has coordinates)."""
    geojson_path = os.path.join(_ROOT, "data", "Hawker Centres", "Hawker Centres (GEOJSON).geojson")
    if not os.path.exists(geojson_path):
        print("Hawker GeoJSON not found!")
        return

    with open(geojson_path) as f:
        data = json.load(f)

    hawkers = []
    for feat in data["features"]:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]  # [lon, lat]
        hawkers.append({
            "name": props.get("NAME", ""),
            "lat": coords[1],
            "lon": coords[0],
            "postal_code": props.get("ADDRESSPOSTALCODE", ""),
        })

    out = os.path.join(_CACHES, "hawker_cache.json")
    with open(out, "w") as f:
        json.dump(hawkers, f, indent=2)
    print(f"Saved {len(hawkers)} hawker centres to {out}")


# Major shopping malls — (name, lat, lon) with known coordinates
MALLS = [
    ("ION Orchard", 1.3040, 103.8318),
    ("VivoCity", 1.2644, 103.8223),
    ("Jurong Point", 1.3397, 103.7065),
    ("Tampines Mall", 1.3528, 103.9452),
    ("NEX", 1.3508, 103.8718),
    ("Causeway Point", 1.4361, 103.7860),
    ("Northpoint City", 1.4293, 103.8365),
    ("Lot One", 1.3443, 103.7543),
    ("AMK Hub", 1.3691, 103.8487),
    ("Hougang Mall", 1.3726, 103.8935),
    ("Compass One", 1.3924, 103.8950),
    ("Waterway Point", 1.4068, 103.9023),
    ("White Sands", 1.3726, 103.9495),
    ("Bedok Mall", 1.3245, 103.9302),
    ("Century Square", 1.3528, 103.9433),
    ("West Mall", 1.3498, 103.7490),
    ("Clementi Mall", 1.3150, 103.7648),
    ("JCube", 1.3334, 103.7404),
    ("Plaza Singapura", 1.3006, 103.8453),
    ("Bugis Junction", 1.2991, 103.8555),
    ("Suntec City", 1.2946, 103.8573),
    ("Marina Bay Sands", 1.2834, 103.8607),
    ("Raffles City", 1.2934, 103.8528),
    ("Parkway Parade", 1.3013, 103.9054),
    ("Junction 8", 1.3499, 103.8487),
    ("Bukit Panjang Plaza", 1.3785, 103.7627),
    ("Tiong Bahru Plaza", 1.2867, 103.8270),
    ("Toa Payoh Hub", 1.3325, 103.8474),
    ("Sembawang Shopping Centre", 1.4493, 103.8201),
    ("Sun Plaza", 1.4393, 103.8381),
    ("Greenwich V", 1.3878, 103.9049),
    ("Eastpoint Mall", 1.3428, 103.9536),
    ("Westgate", 1.3344, 103.7424),
    ("IMM", 1.3348, 103.7466),
    ("Hillion Mall", 1.3831, 103.7633),
    ("Our Tampines Hub", 1.3528, 103.9401),
    ("Woodlands Civic Centre", 1.4371, 103.7863),
    ("Yew Tee Point", 1.3970, 103.7474),
    ("The Seletar Mall", 1.3939, 103.8774),
    ("Rivervale Mall", 1.3921, 103.9019),
    ("Punggol Plaza", 1.4037, 103.9027),
    ("Heartland Mall", 1.4007, 103.7441),
    ("Dawson Place", 1.2924, 103.8058),
]

# Major parks — (name, lat, lon) with known coordinates
PARKS = [
    ("East Coast Park", 1.3008, 103.9122),
    ("Bishan-Ang Mo Kio Park", 1.3619, 103.8460),
    ("Jurong Lake Gardens", 1.3383, 103.7292),
    ("Pasir Ris Park", 1.3813, 103.9512),
    ("Bedok Reservoir Park", 1.3391, 103.9329),
    ("West Coast Park", 1.2812, 103.7660),
    ("Punggol Waterway Park", 1.4084, 103.9058),
    ("Woodlands Waterfront", 1.4576, 103.7689),
    ("Sembawang Park", 1.4618, 103.8302),
    ("Admiralty Park", 1.4470, 103.7802),
    ("Labrador Nature Reserve", 1.2655, 103.8026),
    ("Kent Ridge Park", 1.2815, 103.7907),
    ("MacRitchie Reservoir Park", 1.3434, 103.8338),
    ("Bukit Timah Nature Reserve", 1.3500, 103.7760),
    ("Lower Peirce Reservoir Park", 1.3695, 103.8193),
    ("Tampines Eco Green", 1.3580, 103.9650),
    ("Yishun Park", 1.4262, 103.8416),
    ("Clementi Woods Park", 1.3278, 103.7780),
    ("Ang Mo Kio Town Garden West", 1.3710, 103.8373),
    ("Choa Chu Kang Park", 1.3862, 103.7441),
    ("Bukit Batok Nature Park", 1.3498, 103.7632),
    ("Toa Payoh Town Park", 1.3325, 103.8492),
    ("Kallang Riverside Park", 1.3110, 103.8651),
    ("Coney Island Park", 1.4099, 103.9186),
    ("Sengkang Riverside Park", 1.3964, 103.8879),
]


def build_static_cache(items, output_path, label):
    """Save a list of (name, lat, lon) tuples to a JSON cache."""
    results = [{"name": name, "lat": lat, "lon": lon} for name, lat, lon in items]
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} {label} to {output_path}")


def main():
    print("=" * 50)
    print("  Amenity Cache Builder")
    print("=" * 50)

    print("\n[1/3] Hawker centres (from GeoJSON)...")
    build_hawker_cache()

    print("\n[2/3] Shopping malls (hardcoded coordinates)...")
    build_static_cache(MALLS, os.path.join(_CACHES, "mall_cache.json"), "malls")

    print("\n[3/3] Parks (hardcoded coordinates)...")
    build_static_cache(PARKS, os.path.join(_CACHES, "park_cache.json"), "parks")

    print("\nDone!")


if __name__ == "__main__":
    main()
