"""Central configuration: data sources, study design, and column roles."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"

# --------------------------------------------------------------------------
# Real data sources (City of San Diego ArcGIS REST services, public)
# --------------------------------------------------------------------------
HOUSING_SERVICE = (
    "https://webmaps.sandiego.gov/arcgis/rest/services/Planning/"
    "PLN_Housing_ServiceLayers/MapServer"
)
SITES_LAYER_URL = f"{HOUSING_SERVICE}/1"    # Housing Element Adequate Sites (2021-2029)
PERMITS_LAYER_URL = f"{HOUSING_SERVICE}/0"  # Housing Element Permit Data (DSD approvals)

# Bulk CSV alternative for permits (City open data portal):
PERMITS_CSV_URL = (
    "https://seshat.datasd.org/development_permits/approvals_issued_datasd.csv"
)

# --------------------------------------------------------------------------
# Study design
# --------------------------------------------------------------------------
# The 6th-cycle Housing Element was adopted in mid-2021. Features describe each
# site as inventoried; the outcome is housing approved AFTER this date.
OUTCOME_START = "2021-06-01"
OUTCOME_END = None            # None = through latest available data
MIN_NEW_UNITS = 1             # a site is "realized" if >= this many net new
                              # (non-ADU) dwelling units were approved on it
COUNT_ADUS = False            # ADU/JADU-only approvals do not count by default

DOWNTOWN_LATLON = (32.7157, -117.1611)
RANDOM_STATE = 42
N_FOLDS = 5

# --------------------------------------------------------------------------
# Column roles in the sites inventory
# --------------------------------------------------------------------------
# These fields are (or may be) updated AFTER the inventory was made and would
# leak the outcome into the features. They are dropped before modeling.
LEAKAGE_COLUMNS = [
    "STATUS", "STATUS_DESC", "DT_Status", "Comments", "TEMP",
    "approval_du_net_change_Sum", "approval_du_extremely_low_Sum",
    "approval_du_very_low_Sum", "approval_du_low_Sum",
    "approval_du_moderate_Sum", "approval_du_above_moderate_Sum",
    "created_user", "created_date", "last_edited_user", "last_edited_date",
]

GROUP_COLUMN = "CPNAME"  # community plan area -> spatial cross-validation groups
