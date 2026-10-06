"""Unit tests for outcome construction and leakage handling."""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsr.build_dataset import _ctcac_ordinal, build_features, build_labels  # noqa: E402


def _sites():
    return gpd.GeoDataFrame({
        "APN_8": ["11111111", "22222222", "33333333", "44444444"],
        "CPNAME": ["A", "A", "B", "B"], "ACRES": [0.5, 0.3, 1.0, 0.2],
        "ExtgUnits": [0, 1, 0, 0], "PotenUnits": [20, 10, 50, 5], "NetUnits": [20, 9, 50, 5],
        "STATUS_DESC": ["Permitted", "", "", ""],
        "approval_du_net_change_Sum": [20, 0, 0, 0],
    }, geometry=[box(-117.16, 32.71, -117.159, 32.711), box(-117.15, 32.72, -117.149, 32.721),
                 box(-117.14, 32.73, -117.139, 32.731), box(-117.13, 32.74, -117.129, 32.741)],
       crs="EPSG:4326")


def _permit(apn, date, du, adu=0, lon=0.0, lat=0.0, pid="P"):
    return {"APPROVAL_ID": pid, "JOB_APN": apn, "APPROVAL_ISSUE_DATE": pd.Timestamp(date),
            "TOTAL_DU": du, "TOTAL_ADU": adu, "TOTAL_JADU": 0, "LAT_JOB": lat, "LNG_JOB": lon}


def test_labels_apply_window_adu_rule_and_spatial_fallback():
    permits = pd.DataFrame([
        _permit("1111111100", "2022-03-01", 20, pid="p1"),             # counts (APN match)
        _permit("2222222200", "2022-03-01", 0, adu=1, pid="p2"),       # ADU only -> ignored
        _permit("3333333300", "2019-01-01", 40, pid="p3"),             # before window -> ignored
        _permit(None, "2023-05-01", 5, lon=-117.1295, lat=32.7405, pid="p4"),  # no APN -> spatial
    ])
    permits = gpd.GeoDataFrame(permits, geometry=[Point(0, 0)] * len(permits), crs="EPSG:4326")
    lab = build_labels(_sites(), permits).set_index("APN_8")
    assert lab.realized.to_dict() == {"11111111": 1, "22222222": 0, "33333333": 0, "44444444": 1}


def test_leakage_columns_never_become_features():
    X = build_features(_sites())
    assert not any("STATUS" in c or "approval_du" in c for c in X.columns)
    assert X.notna().all().all() or X["max_density"].isna().all()


def test_high_segregation_is_lowest_opportunity():
    assert _ctcac_ordinal("High Segregation & Poverty") == 0
    assert _ctcac_ordinal("Highest Resource") == 4
    assert _ctcac_ordinal("High Resource") == 3
