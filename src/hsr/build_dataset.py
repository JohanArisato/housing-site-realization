"""Build the modeling table: one row per inventoried site, leak-free features,
and a binary outcome (was new housing approved on the site after adoption?).

Outcome construction
--------------------
1. Keep housing approvals issued inside the outcome window.
2. Count net new dwelling units (TOTAL_DU). ADU/JADU-only approvals are
   excluded by default (config.COUNT_ADUS) because the Housing Element credits
   ADUs separately from its sites inventory.
3. Match approvals to sites by APN (first 8 digits of JOB_APN == APN_8).
   Approvals with no APN fall back to a point-in-polygon spatial join.
4. realized = 1 if matched new units >= config.MIN_NEW_UNITS.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config

TRUTHY = {"y", "yes", "true", "1", "t"}

LANDUSE_GROUPS = [  # (group, keywords) - first match wins
    ("vacant", ["vacant", "undeveloped"]),
    ("parking", ["parking"]),
    ("single_family", ["single"]),
    ("multi_family", ["multi", "apartment", "condo", "duplex"]),
    ("commercial", ["commercial", "retail", "shopping", "mixed"]),
    ("office", ["office"]),
    ("industrial", ["industrial", "warehouse"]),
]

CTCAC_ORDER = [  # higher = more opportunity; order matters ("High Segregation"
    (["segregation", "poverty", "hsp"], 0),   # must be caught before "high")
    (["highest"], 4), (["high"], 3), (["moderate"], 2), (["low"], 1),
]


def _flag(s: pd.Series) -> pd.Series:
    """Robustly turn Yes/No, Y/N, 1/0, or free text into 0/1."""
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    return s.fillna("").astype(str).str.strip().str.lower().isin(TRUTHY).astype(int)


def _nonempty(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return (~t.isin(["", "no", "n", "none", "0", "nan"])).astype(int)


def _col(df: pd.DataFrame, name: str, default) -> pd.Series:
    """Column if present, else a constant Series (layers differ between years)."""
    return df[name] if name in df else pd.Series(default, index=df.index)


def _landuse_group(v) -> str:
    t = str(v).lower()
    for group, keys in LANDUSE_GROUPS:
        if any(k in t for k in keys):
            return group
    return "other"


def _ctcac_ordinal(v) -> float:
    t = str(v).lower()
    for keys, val in CTCAC_ORDER:
        if any(k in t for k in keys):
            return val
    return np.nan


def _to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):           # ArcGIS epoch (ms, sometimes s)
        unit = "ms" if s.abs().max() > 1e11 else "s"
        return pd.to_datetime(s.astype("float64"), unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


# ---------------------------------------------------------------------------
def build_labels(sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame) -> pd.DataFrame:
    p = permits.copy()
    p["issue_date"] = _to_datetime(p["APPROVAL_ISSUE_DATE"])
    in_window = p.issue_date >= pd.Timestamp(config.OUTCOME_START)
    if config.OUTCOME_END:
        in_window &= p.issue_date <= pd.Timestamp(config.OUTCOME_END)
    units = p["TOTAL_DU"].fillna(0)
    if config.COUNT_ADUS:
        units = units + p.get("TOTAL_ADU", 0).fillna(0) + p.get("TOTAL_JADU", 0).fillna(0)
    p["new_units"] = units
    p = p[in_window & (p.new_units > 0)]

    # 1) APN match
    p["apn8"] = p["JOB_APN"].astype("string").str.replace(r"\D", "", regex=True).str[:8]
    has_apn = p.apn8.str.len().eq(8).fillna(False)
    by_apn = (p[has_apn].merge(sites[["APN_8"]].drop_duplicates(), left_on="apn8", right_on="APN_8")
              .groupby("APN_8").agg(units=("new_units", "sum"),
                                    first_issue=("issue_date", "min"),
                                    n_approvals=("APPROVAL_ID", "nunique")))

    # 2) spatial fallback for approvals without a usable APN
    no_apn = p[~has_apn].copy()
    if len(no_apn):
        pts = gpd.GeoDataFrame(no_apn, geometry=gpd.points_from_xy(no_apn.LNG_JOB, no_apn.LAT_JOB),
                               crs="EPSG:4326")
        joined = gpd.sjoin(pts, sites[["APN_8", "geometry"]], predicate="within")
        joined = joined.drop_duplicates(subset=["APPROVAL_ID", "APN_8"])
        by_space = joined.groupby("APN_8").agg(units=("new_units", "sum"),
                                               first_issue=("issue_date", "min"),
                                               n_approvals=("APPROVAL_ID", "nunique"))
        by_apn = pd.concat([by_apn, by_space]).groupby(level=0).agg(
            units=("units", "sum"), first_issue=("first_issue", "min"),
            n_approvals=("n_approvals", "sum"))

    lab = sites[["APN_8"]].merge(by_apn, left_on="APN_8", right_index=True, how="left")
    lab["approved_units"] = lab.units.fillna(0)
    lab["realized"] = (lab.approved_units >= config.MIN_NEW_UNITS).astype(int)
    return lab[["APN_8", "approved_units", "first_issue", "realized"]]


def build_features(sites: gpd.GeoDataFrame) -> pd.DataFrame:
    s = sites.drop(columns=[c for c in config.LEAKAGE_COLUMNS if c in sites], errors="ignore")
    f = pd.DataFrame(index=s.index)
    acres = s["ACRES"].clip(lower=0.01)
    poten = s["PotenUnits"].clip(lower=1)
    extg = s["ExtgUnits"].fillna(0).clip(lower=0)
    net = s["NetUnits"].fillna(poten - extg).clip(lower=0)

    f["log_acres"] = np.log(acres)
    f["log_net_units"] = np.log1p(net)
    f["existing_units"] = extg
    f["utilization"] = (extg / poten).clip(0, 1)          # how built-out today
    f["max_density"] = _col(s, "HIGHDEN", np.nan)
    f["transit_priority"] = _flag(_col(s, "TPAHIIP", 0))
    f["coastal_zone"] = _flag(_col(s, "Ov_CZ", ""))
    f["coastal_height_limit"] = _flag(_col(s, "Ov_CHLOZ", ""))
    f["floodway"] = _flag(_col(s, "Ov_Fldway100", ""))
    f["mhpa_share"] = pd.to_numeric(_col(s, "Ov_MHPA", 0), errors="coerce").fillna(0).clip(0, 1)
    f["airport_noise"] = _nonempty(_col(s, "Ov_Air_Noise", ""))
    f["opportunity"] = _col(s, "CTCAC", "").map(_ctcac_ordinal)
    f["poverty_area"] = _flag(_col(s, "Poverty", ""))
    f["minority_area"] = _flag(_col(s, "Minority", ""))
    f["recap"] = _flag(_col(s, "RECAP", ""))
    f["affordable_site"] = _flag(_col(s, "AFFORD", 0))
    f["requires_demo"] = _flag(_col(s, "DEMO", 0))

    lu = _col(s, "Extg_Landuse", "").map(_landuse_group)
    for g in [g for g, _ in LANDUSE_GROUPS] + ["other"]:
        f[f"lu_{g}"] = (lu == g).astype(int)

    # ---- spatial context (uses only baseline features: no label leakage) ----
    proj = sites.to_crs(2230)                             # CA State Plane VI, US feet
    cent = proj.geometry.centroid
    dt = gpd.GeoSeries(gpd.points_from_xy([config.DOWNTOWN_LATLON[1]],
                                          [config.DOWNTOWN_LATLON[0]]), crs=4326).to_crs(2230)
    f["dist_downtown_km"] = cent.distance(dt.iloc[0]) * 0.0003048
    pts = gpd.GeoDataFrame({"net": net.values}, geometry=cent.values, crs=2230)
    buf = gpd.GeoDataFrame(geometry=cent.buffer(1312).values, crs=2230)  # 400 m
    pairs = gpd.sjoin(buf, pts, predicate="contains")
    pairs = pairs[pairs.index != pairs.index_right]
    f["nbr_sites_400m"] = pairs.groupby(level=0).size().reindex(range(len(s)), fill_value=0).values
    f["nbr_log_capacity_400m"] = np.log1p(
        pairs.groupby(level=0)["net"].sum().reindex(range(len(s)), fill_value=0).values)
    return f


def build(data_dir) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.Series, pd.Series]:
    sites = gpd.read_file(data_dir / "sites.geojson").reset_index(drop=True)
    permits = gpd.read_file(data_dir / "permits.geojson")
    sites["APN_8"] = sites["APN_8"].astype(str).str.zfill(8)
    labels = build_labels(sites, permits)
    X = build_features(sites)
    y = labels["realized"].reset_index(drop=True)
    groups = sites[config.GROUP_COLUMN].fillna("Unknown").reset_index(drop=True)
    sites = sites.join(labels[["approved_units", "first_issue"]].reset_index(drop=True))
    return sites, X, y, groups
