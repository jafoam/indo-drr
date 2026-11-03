"""Environment and dependencies validation test."""
import sys
import importlib
import pytest

REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "streamlit",
    "geopandas",
    "shapely",
    "pyproj",
    "libpysal",
    "esda",
    "plotly",
    "pydeck",
    "pyarrow",
    "openpyxl",
    "pydantic",
    "pandas",
    "numpy",
    "scipy",
]

@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_dependencies_importable(module_name: str):
    """Verify all pinned dependencies are installed and importable."""
    module = importlib.import_module(module_name)
    assert module is not None

def test_config_paths_and_tokens():
    """Verify src.config defines expected constants."""
    from src.config import (
        PROJECT_ROOT,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        COLOR_TOKENS,
        BIVARIATE_PALETTE,
        LISA_PALETTE,
    )
    assert PROJECT_ROOT.exists()
    assert DATA_RAW_DIR.exists()
    assert DATA_PROCESSED_DIR.exists()
    assert "bg-primary" in COLOR_TOKENS
    assert "critical-gap" in BIVARIATE_PALETTE
    assert "high-high" in LISA_PALETTE
