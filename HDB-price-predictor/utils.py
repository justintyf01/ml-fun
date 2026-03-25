import math
import requests

# ---------------------------------------------------------
# Haversine distance helper functions
# ---------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates straight-line distance (meters) between two GPS coordinates."""
    R = 6371000 # Earth radius
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_onemap_coordinates(address):
    """Hits the OneMap API to get lat/lon for a specific HDB block + street."""
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=N&pageNum=1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data['found'] > 0:
            return float(data['results'][0]['LATITUDE']), float(data['results'][0]['LONGITUDE'])
    except Exception as e:
        pass
    return None, None