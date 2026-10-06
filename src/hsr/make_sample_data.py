"""Generate a realistic SYNTHETIC sample that mirrors the real schemas.

The sample lets anyone run the full pipeline offline (and lets CI test it).
It uses the same field names as the City's ArcGIS layers, community plan
areas placed at their approximate real locations, and a known data-generating
process, so we can check the model recovers the true drivers.

    python -m hsr.make_sample_data --n-sites 6000

NOTE: results produced from this sample are illustrative only. Run
`python -m hsr.download` and `python run_pipeline.py --data real` for findings.
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from . import config

# Approximate centroids of City of San Diego community plan areas (lat, lon)
# and a 0-1 "market strength" prior used only by the synthetic generator.
CPAS = {
    "Downtown": (32.7150, -117.1600, 0.95),
    "Uptown": (32.7480, -117.1640, 0.85),
    "North Park": (32.7470, -117.1300, 0.80),
    "Golden Hill": (32.7180, -117.1400, 0.60),
    "Mid-City:City Heights": (32.7480, -117.0990, 0.45),
    "Mid-City:Normal Heights": (32.7630, -117.1150, 0.65),
    "College Area": (32.7720, -117.0700, 0.60),
    "Mission Valley": (32.7700, -117.1500, 0.90),
    "Linda Vista": (32.7820, -117.1720, 0.55),
    "Clairemont Mesa": (32.8150, -117.1950, 0.50),
    "Kearny Mesa": (32.8300, -117.1400, 0.70),
    "Serra Mesa": (32.8050, -117.1350, 0.45),
    "University": (32.8700, -117.2150, 0.85),
    "Pacific Beach": (32.7980, -117.2400, 0.75),
    "La Jolla": (32.8400, -117.2700, 0.70),
    "Ocean Beach": (32.7480, -117.2470, 0.55),
    "Peninsula": (32.7300, -117.2350, 0.50),
    "Midway-Pacific Highway": (32.7550, -117.2050, 0.75),
    "Old Town San Diego": (32.7550, -117.1950, 0.55),
    "Mira Mesa": (32.9150, -117.1400, 0.55),
    "Rancho Bernardo": (33.0200, -117.0750, 0.40),
    "Rancho Penasquitos": (32.9600, -117.1050, 0.35),
    "Carmel Valley": (32.9450, -117.2200, 0.60),
    "Otay Mesa": (32.5700, -116.9800, 0.40),
    "Otay Mesa-Nestor": (32.5800, -117.0800, 0.30),
    "San Ysidro": (32.5550, -117.0450, 0.30),
    "Barrio Logan": (32.6980, -117.1450, 0.50),
    "Southeastern San Diego": (32.7000, -117.1000, 0.35),
    "Encanto Neighborhoods": (32.7100, -117.0600, 0.30),
    "Skyline-Paradise Hills": (32.6850, -117.0450, 0.25),
    "Navajo": (32.8050, -117.0500, 0.35),
    "Tierrasanta": (32.8250, -117.1000, 0.30),
}

LANDUSE = ["Vacant", "Single Family", "Multi-Family", "Commercial", "Office",
           "Industrial", "Parking", "Institutional"]
LANDUSE_P = [0.10, 0.30, 0.14, 0.24, 0.08, 0.05, 0.05, 0.04]
CTCAC = ["Highest Resource", "High Resource", "Moderate Resource",
         "Low Resource", "High Segregation & Poverty"]

LANDUSE_EFFECT = {"Vacant": 1.1, "Single Family": -0.9, "Multi-Family": -0.5,
                  "Commercial": 0.45, "Office": 0.1, "Industrial": -0.2,
                  "Parking": 0.9, "Institutional": -0.8}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_sites(n: int, rng: np.random.Generator) -> gpd.GeoDataFrame:
    names = list(CPAS)
    # bigger, denser CPAs get more sites
    weights = np.array([0.5 + CPAS[c][2] for c in names])
    cpa = rng.choice(names, size=n, p=weights / weights.sum())
    lat0 = np.array([CPAS[c][0] for c in cpa])
    lon0 = np.array([CPAS[c][1] for c in cpa])
    market_cpa = np.array([CPAS[c][2] for c in cpa])
    lat = lat0 + rng.normal(0, 0.010, n)
    lon = lon0 + rng.normal(0, 0.012, n)

    landuse = rng.choice(LANDUSE, size=n, p=LANDUSE_P)
    acres = np.clip(rng.lognormal(np.log(0.25), 0.9, n), 0.05, 25)
    # zoned max density (du/ac); denser in strong markets
    highden = np.select(
        [market_cpa > 0.8, market_cpa > 0.55],
        [rng.choice([73, 109, 145, 218, 290], n), rng.choice([29, 44, 73, 109], n)],
        rng.choice([15, 29, 44, 73], n))
    lowden = (highden * 0.5).astype(int)
    extg_units = np.where(landuse == "Single Family", 1,
                  np.where(landuse == "Multi-Family", rng.integers(2, 24, n), 0))
    poten_units = np.maximum(np.round(acres * highden * 0.85), extg_units + 1).astype(int)
    net_units = poten_units - extg_units

    tpa = (rng.random(n) < (0.35 + 0.5 * market_cpa)).astype(int)
    coastal = ((lon < -117.2) & (rng.random(n) < 0.8)).astype(int)
    chloz = (coastal & (rng.random(n) < 0.6)).astype(int)  # 30-ft coastal height limit
    floodway = (rng.random(n) < 0.04).astype(int)
    mhpa = np.where(rng.random(n) < 0.05, rng.uniform(0.05, 0.6, n), 0.0)
    air_noise = np.where(rng.random(n) < 0.10, "60-65 CNEL", "")
    # opportunity category correlated with market & latitude (north = higher)
    opp_score = market_cpa * 0.6 + (lat - 32.55) * 1.2 + rng.normal(0, 0.15, n)
    ctcac = pd.cut(opp_score, bins=[-9, 0.35, 0.5, 0.65, 0.8, 9],
                   labels=CTCAC[::-1]).astype(str)
    poverty = np.where(opp_score < 0.4, "Yes", "No")
    minority = np.where(opp_score < 0.5, "Yes", "No")
    recap = np.where(opp_score < 0.3, "Yes", "No")
    afford = (rng.random(n) < 0.15).astype(int)
    demo = (extg_units > 0).astype(int)

    # ---- true data-generating process (log-odds of approval by ~2025) ----
    spatial_trend = 0.9 * np.sin(lat * 60) * np.cos(lon * 45)  # unobserved local shocks
    eta = (-3.1
           + np.vectorize(LANDUSE_EFFECT.get)(landuse)
           + 0.45 * np.log1p(net_units)
           + 0.55 * tpa
           + 1.8 * (market_cpa - 0.55)
           - 0.9 * chloz - 1.0 * floodway - 2.0 * mhpa
           - 0.35 * (air_noise != "")
           - 0.6 * np.clip(extg_units / poten_units, 0, 1)
           + 0.3 * afford
           # non-linear feasibility effects real projects face:
           - 1.1 * (acres < 0.11)                        # lots too small to pencil out
           + 0.7 * (tpa & (highden >= 73))               # transit-area density bonus pays off
           - 0.8 * ((extg_units > 0) & (highden < 30))   # demolition not worth low upzoning
           + 0.35 * spatial_trend)
    p_true = _sigmoid(eta)
    realized = rng.random(n) < p_true

    apn8 = rng.choice(np.arange(10_000_000, 99_999_999), size=n, replace=False).astype(str)
    side_m = np.sqrt(acres * 4046.86)
    dlat = side_m / 111_000 / 2
    dlon = side_m / (111_000 * np.cos(np.radians(lat))) / 2
    geoms = [box(x - dx, y - dy, x + dx, y + dy) for x, y, dx, dy in zip(lon, lat, dlon, dlat)]

    sites = gpd.GeoDataFrame({
        "OBJECTID": np.arange(1, n + 1), "APN_8": apn8,
        "SITE_ID": [f"S{i:06d}" for i in range(n)], "CPNAME": cpa,
        "ACRES": acres.round(3), "ExtgUnits": extg_units, "PotenUnits": poten_units,
        "NetUnits": net_units, "LOWDEN": lowden, "HIGHDEN": highden,
        "Extg_Landuse": landuse, "TPAHIIP": tpa, "Ov_CZ": np.where(coastal, "Yes", "No"),
        "Ov_CHLOZ": np.where(chloz, "Yes", "No"),
        "Ov_Fldway100": np.where(floodway, "Yes", "No"), "Ov_MHPA": mhpa.round(3),
        "Ov_Air_Noise": air_noise, "CTCAC": ctcac, "Poverty": poverty,
        "Minority": minority, "RECAP": recap, "AFFORD": afford, "DEMO": demo,
        # leakage-prone fields present in the real layer; filled to prove they get dropped
        "STATUS_DESC": np.where(realized, "Permitted", "Available"),
        "approval_du_net_change_Sum": np.where(realized, net_units, 0),
        "_p_true": p_true,  # ground truth, synthetic only
    }, geometry=geoms, crs="EPSG:4326")
    return sites, realized, lat, lon


def make_permits(sites, realized, lat, lon, rng) -> gpd.GeoDataFrame:
    rows = []
    start = pd.Timestamp(config.OUTCOME_START)
    def add(i, date, du, adu, approval_type, apn=True):
        rows.append({
            "APPROVAL_ID": f"PMT-{len(rows):07d}",
            "APPROVAL_TYPE": approval_type,
            "APPROVAL_ISSUE_DATE": date,
            "JOB_APN": (sites.APN_8.iat[i] + "00") if apn else None,
            "LAT_JOB": lat[i] + rng.normal(0, 0.00005),
            "LNG_JOB": lon[i] + rng.normal(0, 0.00005),
            "TOTAL_DU": du, "TOTAL_ADU": adu, "TOTAL_JADU": 0,
            "GIS_COMM_PLAN_NAME": sites.CPNAME.iat[i],
        })

    net = sites.NetUnits.to_numpy()
    for i in np.where(realized)[0]:
        units = max(1, int(net[i] * rng.uniform(0.3, 1.1)))
        date = start + pd.Timedelta(days=int(rng.integers(30, 1550)))
        add(i, date, units, 0, "Building Permit - New Construction", apn=rng.random() > 0.08)
    # Noise that the label logic must ignore:
    sf = np.where(sites.Extg_Landuse.eq("Single Family").to_numpy() & ~realized)[0]
    for i in rng.choice(sf, size=len(sf) // 6, replace=False):      # ADU-only approvals
        add(i, start + pd.Timedelta(days=int(rng.integers(30, 1500))), 0, 1,
            "Building Permit - ADU")
    for i in rng.choice(np.where(~realized)[0], size=150, replace=False):  # before window
        add(i, start - pd.Timedelta(days=int(rng.integers(30, 900))), 5, 0,
            "Building Permit - New Construction")
    for i in rng.choice(len(sites), size=800, replace=False):        # non-housing work
        add(i, start + pd.Timedelta(days=int(rng.integers(30, 1500))), 0, 0,
            "Electrical Permit")

    df = pd.DataFrame(rows)
    # ArcGIS returns dates as epoch milliseconds; mimic that.
    df["APPROVAL_ISSUE_DATE"] = pd.to_datetime(df.APPROVAL_ISSUE_DATE).astype("datetime64[ms]").astype("int64")
    return gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in zip(df.LNG_JOB, df.LAT_JOB)],
                            crs="EPSG:4326")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sites", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    sites, realized, lat, lon = make_sites(args.n_sites, rng)
    permits = make_permits(sites, realized, lat, lon, rng)
    config.DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    sites.to_file(config.DATA_SAMPLE / "sites.geojson", driver="GeoJSON")
    permits.to_file(config.DATA_SAMPLE / "permits.geojson", driver="GeoJSON")
    print(f"Synthetic sample: {len(sites)} sites ({realized.mean():.1%} realized), "
          f"{len(permits)} permits -> {config.DATA_SAMPLE}")


if __name__ == "__main__":
    main()
