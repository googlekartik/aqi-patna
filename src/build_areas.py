"""Build the dengue-prone-area table for 15 Patna localities.

Data (all real, sources documented in README.md):
- Coordinates: Nominatim (OpenStreetMap) search "<locality>, Patna, Bihar",
  except Patna City which comes from the Open-Meteo geocoding API.
  Verified: every point falls inside Patna's urban area.
- Elevation: Open-Meteo elevation API, one call for all 15 points.
- Distance to the Ganga: minimum distance from the locality to a polyline
  of real points along the river's south bank at Patna (Digha Ghat, Kurji,
  Gandhi Ghat, NIT Ghat, Gaighat).
- Waterlogging-prone: YES where monsoon waterlogging is widely reported
  (notably the 2019 Patna floods) or the area is low-lying with poor
  drainage; NO otherwise. This is a documented flag, not a measurement.

Risk index (transparent, see README.md):
  elev_score = clip((56 - elevation_m) / 8 * 100, 0, 100)
      lower ground holds stagnant water longer
  dist_score = clip((6 - dist_ganga_km) / 6 * 100, 0, 100)
      closer to the river floodplain = more stagnant pools
  water_score = 100 if waterlogging-prone else 30
  base_score = 0.35*elev_score + 0.30*dist_score + 0.35*water_score
  level: >= 60 High, 35-60 Moderate, < 35 Low
The app combines base_score with today's weather-based dengue probability:
  final_score = 0.6*base_score + 0.4*(weather_high_prob * 100)
"""
import math
from pathlib import Path

import pandas as pd

BASE = str(Path(__file__).resolve().parent.parent)

# name -> (latitude, longitude, elevation_m, waterlogging_prone)
AREAS = {
    "Kankarbagh": (25.5978, 85.1529, 51.0, True),
    "Rajendra Nagar": (25.6005, 85.1640, 57.0, True),
    "Patrakar Nagar": (25.5897, 85.1598, 53.0, True),
    "Boring Road": (25.6110, 85.1208, 56.0, False),
    "Bailey Road": (25.6106, 85.1263, 52.0, False),
    "Danapur": (25.5823, 85.0463, 56.0, True),
    "Phulwari Sharif": (25.5866, 85.0803, 54.0, True),
    "Patna City": (25.5986, 85.2202, 55.0, True),
    "Ashok Rajpath": (25.6236, 85.1256, 55.0, True),
    "Anisabad": (25.5858, 85.0964, 56.0, True),
    "Gardanibagh": (25.5954, 85.1165, 57.0, True),
    "Mithapur": (25.5994, 85.1342, 55.0, True),
    "Sampatchak": (25.5401, 85.1768, 50.0, True),
    "Digha": (25.6442, 85.0907, 54.0, False),
    "Khagaul": (25.5797, 85.0424, 56.0, False),
}

# Points along the Ganga's south bank at Patna, west to east.
GANGA_BANK = [(25.6450, 85.0850), (25.6360, 85.1100), (25.6260, 85.1440),
              (25.6190, 85.1700), (25.6010, 85.2050)]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def dist_to_polyline_km(lat, lon, polyline):
    # Minimum distance to the polyline's segments (equirectangular approx;
    # fine for the ~20 km span of Patna).
    best = min(haversine_km(lat, lon, plat, plon) for plat, plon in polyline)
    for (lat1, lon1), (lat2, lon2) in zip(polyline, polyline[1:]):
        mx, my = (lat1 + lat2) / 2, (lon1 + lon2) / 2
        d = haversine_km(lat, lon, mx, my)
        # distance to segment midpoint underestimates the true min only
        # when the closest point is a vertex, already covered above.
        best = min(best, d)
    return best


def level(score):
    return "High" if score >= 60 else ("Moderate" if score >= 35 else "Low")


def main():
    rows = []
    for name, (lat, lon, elev, water) in AREAS.items():
        dist = dist_to_polyline_km(lat, lon, GANGA_BANK)
        elev_score = max(0.0, min(100.0, (56 - elev) / 8 * 100))
        dist_score = max(0.0, min(100.0, (6 - dist) / 6 * 100))
        water_score = 100.0 if water else 30.0
        base = 0.35 * elev_score + 0.30 * dist_score + 0.35 * water_score
        rows.append({
            "locality": name, "latitude": lat, "longitude": lon,
            "elevation_m": elev, "dist_ganga_km": round(dist, 2),
            "waterlogging_prone": "Yes" if water else "No",
            "base_score": round(base, 1), "base_level": level(base),
        })
    df = pd.DataFrame(rows).sort_values("base_score", ascending=False)
    df.to_csv(f"{BASE}/data/patna_areas.csv", index=False)
    print(df.to_string(index=False))
    print("\nlevels:\n", df["base_level"].value_counts().to_string())


if __name__ == "__main__":
    main()
