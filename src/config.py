"""Central configuration and visual tokens for IDR-GE."""
from pathlib import Path
from typing import Dict, Any

# Filesystem layout
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = PROJECT_ROOT / "Data"  # Existing workspace raw data folder
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Processed file paths
GEOJSON_FILE = DATA_PROCESSED_DIR / "indonesia_regencies.geojson"
PARQUET_FILE = DATA_PROCESSED_DIR / "podes_irbi_clean.parquet"

# Coordinate boundaries for Indonesia
INDONESIA_BOUNDS = {
    "min_lat": -11.5,
    "max_lat": 6.5,
    "min_lon": 95.0,
    "max_lon": 141.5,
}
DEFAULT_MAP_CENTER = {"lat": -2.5, "lon": 118.0, "zoom": 4.2}

# Antislop-compliant Visual Design Tokens
DARK_COLOR_TOKENS: Dict[str, str] = {
    "bg-primary": "#1a2332",           # Dark analytical viewport background
    "bg-surface": "#161e2a",           # Panel, sidebar, and control surface
    "bg-surface-elevated": "#1e293b",  # Interactive controls, inputs, hover states
    "border-subtle": "#2a384c",        # Structural divider lines (1px crisp)
    "text-primary": "#f1f5f9",         # Primary labels, statistics, tooltips
    "text-muted": "#94a3b8",           # Supporting metadata, unit labels
    "accent-focus": "#38bdf8",         # Interactive active border / selection highlight
    "map-style": "carto-darkmatter",   # PyDeck basemap style
    "plotly-template": "plotly_dark",  # Plotly figure template
}

LIGHT_COLOR_TOKENS: Dict[str, str] = {
    "bg-primary": "#f8fafc",           # Bright carto background
    "bg-surface": "#ffffff",           # Panel, sidebar, and card surface
    "bg-surface-elevated": "#f1f5f9",  # Controls, inputs, hover states
    "border-subtle": "#cbd5e1",        # Structural divider lines
    "text-primary": "#0f172a",         # Primary dark typography
    "text-muted": "#64748b",           # Supporting metadata labels
    "accent-focus": "#0284c7",         # Interactive blue accent
    "map-style": "carto-positron",     # PyDeck basemap style
    "plotly-template": "plotly_white", # Plotly figure template
}

COLOR_TOKENS: Dict[str, str] = DARK_COLOR_TOKENS


def get_theme_tokens(theme_name: str = "Dark Theme") -> Dict[str, str]:
    """Return design tokens corresponding to visual theme name."""
    is_bright = theme_name == "Light Theme"
    return LIGHT_COLOR_TOKENS if is_bright else DARK_COLOR_TOKENS

# 3x3 Bivariate Risk vs Mitigation Color Matrix (Hex)
# Rows: Risk (Low R1, Med R2, High R3)
# Cols: Mitigation (Low M1, Med M2, High M3)
BIVARIATE_MATRIX_HEX: Dict[Any, str] = {
    ("R3", "M1"): "#d03838",  # High Risk / Low Mitigation: CRITICAL GAP
    ("R3", "M2"): "#c47c5d",
    ("R3", "M3"): "#82537e",  # High Risk / High Mitigation
    ("R2", "M1"): "#d88478",
    ("R2", "M2"): "#bfa29b",
    ("R2", "M3"): "#7a8ba5",
    ("R1", "M1"): "#e2dfd2",  # Low Risk / Low Mitigation
    ("R1", "M2"): "#9fb8be",
    ("R1", "M3"): "#3b82a6",  # Low Risk / High Mitigation: RESILIENT
    # String key lookups
    "R3-M1": "#d03838",
    "R3-M2": "#c47c5d",
    "R3-M3": "#82537e",
    "R2-M1": "#d88478",
    "R2-M2": "#bfa29b",
    "R2-M3": "#7a8ba5",
    "R1-M1": "#e2dfd2",
    "R1-M2": "#9fb8be",
    "R1-M3": "#3b82a6",
}

BIVARIATE_PALETTE: Dict[str, str] = {
    "critical-gap": "#d03838",
    "high-risk-high-mit": "#82537e",
    "low-risk-low-mit": "#e2dfd2",
    "resilient": "#3b82a6",
}

# LISA Cluster Color Palette
LISA_PALETTE: Dict[str, str] = {
    "high-high": "#ef4444",        # Hotspot (Vivid Crimson)
    "low-low": "#0284c7",          # Coldspot (Cobalt Blue)
    "high-low": "#f97316",         # Outlier (Amber Orange)
    "low-high": "#06b6d4",         # Outlier (Teal Cyan)
    "not-significant": "#334155",  # Subdued Muted Slate
    # Standard Title Case variants
    "High-High": "#ef4444",
    "Low-Low": "#0284c7",
    "High-Low": "#f97316",
    "Low-High": "#06b6d4",
    "Not Significant": "#334155",
}

# Continuous diverging threshold tokens for Net Readiness Balance (Mitigation - Risk)
BALANCE_COLOR_RAMP = [
    (-0.40, [185, 28, 28, 200]),   # Severe Deficit (Deep Red)
    (-0.10, [245, 158, 11, 200]),  # Moderate Deficit (Amber)
    (0.10,  [148, 163, 184, 180]), # Neutral / Balanced (Muted Slate)
    (1.00,  [2, 132, 199, 200]),   # Surplus (Teal/Blue)
]
