"""Download the real City of San Diego data from public ArcGIS REST services.

Usage:
    python -m hsr.download            # sites inventory + housing permits

Both layers are paged (the server returns at most 2,000 features per request).
Output is GeoJSON in WGS84 (EPSG:4326) under data/raw/.
"""
from __future__ import annotations

import argparse
import json
import time

from pathlib import Path

import requests

from . import config


def fetch_arcgis_layer(layer_url: str, where: str = "1=1", page_size: int = 2000,
                       pause: float = 0.3) -> dict:
    """Page through an ArcGIS MapServer/FeatureServer layer and return a
    single GeoJSON FeatureCollection."""
    session = requests.Session()
    count = session.get(f"{layer_url}/query", params={
        "where": where, "returnCountOnly": "true", "f": "json"}, timeout=60).json()
    total = count.get("count")
    print(f"  {layer_url} -> {total} features")

    features, offset = [], 0
    while True:
        params = {
            "where": where, "outFields": "*", "outSR": 4326, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": page_size,
            "orderByFields": "OBJECTID",
        }
        r = session.get(f"{layer_url}/query", params=params, timeout=120)
        r.raise_for_status()
        batch = r.json().get("features", [])
        features.extend(batch)
        print(f"    fetched {len(features)}/{total}", end="\r")
        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(pause)
    print()
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.DATA_RAW))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading Housing Element sites inventory ...")
    sites = fetch_arcgis_layer(config.SITES_LAYER_URL)
    (out / "sites.geojson").write_text(json.dumps(sites))

    print("Downloading Housing Element permit data ...")
    permits = fetch_arcgis_layer(config.PERMITS_LAYER_URL)
    (out / "permits.geojson").write_text(json.dumps(permits))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
