"""Tests for Ingestion & Validation Pipeline."""
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pytest

from src.config import (
    DATA_PROCESSED_DIR,
    GEOJSON_FILE,
    PARQUET_FILE,
    INDONESIA_BOUNDS,
)

MAX_GEOJSON_SIZE_BYTES = 5_242_880  # 5 MB


def test_parquet_exists_and_valid():
    """Verify clean parquet exists, contains exactly 514 rows, and has no nulls in core metrics."""
    assert PARQUET_FILE.exists(), f"Parquet file not found at {PARQUET_FILE}"

    df = pd.read_parquet(PARQUET_FILE)
    assert len(df) == 514, f"Expected 514 rows, found {len(df)}"

    core_cols = ["KODE_WIL", "IRBI_2024", "EWS", "RAMBU", "PERKAP"]
    for col in core_cols:
        assert col in df.columns, f"Column {col} missing from parquet"
        assert df[col].isnull().sum() == 0, f"Column {col} contains null values"

    assert "PROVINSI" in df.columns, "PROVINSI metadata missing from parquet"
    assert "KAB_KOTA" in df.columns, "KAB_KOTA metadata missing from parquet"


def test_geojson_exists_and_valid():
    """Verify GeoJSON exists, feature count is 514, CRS is EPSG:4326, size < 5MB, within bounds."""
    assert GEOJSON_FILE.exists(), f"GeoJSON file not found at {GEOJSON_FILE}"

    file_size = os.path.getsize(GEOJSON_FILE)
    assert file_size < MAX_GEOJSON_SIZE_BYTES, (
        f"GeoJSON file size {file_size} bytes exceeds 5MB limit ({MAX_GEOJSON_SIZE_BYTES} bytes)"
    )

    gdf = gpd.read_file(GEOJSON_FILE)
    assert len(gdf) == 514, f"Expected 514 features, found {len(gdf)}"

    # CRS must be EPSG:4326
    assert gdf.crs is not None, "GeoJSON has no CRS"
    assert gdf.crs.to_epsg() == 4326 or "4326" in str(gdf.crs), f"Expected EPSG:4326, got {gdf.crs}"

    # Geometries must all be valid and non-empty
    assert gdf.geometry.is_valid.all(), "Some geometries in GeoJSON are invalid"
    assert (~gdf.geometry.is_empty).all(), "Some geometries in GeoJSON are empty"

    # Coordinates fall within Indonesia bounding box
    minx, miny, maxx, maxy = gdf.total_bounds
    # Aceh westernmost island tips extend to ~94.97°E, allow 0.1 deg maritime margin
    assert minx >= INDONESIA_BOUNDS["min_lon"] - 0.1, f"Min lon {minx} out of bounds"
    assert maxx <= INDONESIA_BOUNDS["max_lon"] + 0.1, f"Max lon {maxx} out of bounds"
    assert miny >= INDONESIA_BOUNDS["min_lat"] - 0.1, f"Min lat {miny} out of bounds"
    assert maxy <= INDONESIA_BOUNDS["max_lat"] + 0.1, f"Max lat {maxy} out of bounds"

    # Centroids must strictly reside within Indonesia bounds
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*geographic CRS.*")
        centroids = gdf.geometry.centroid
    assert (centroids.x >= INDONESIA_BOUNDS["min_lon"]).all(), "Feature centroid longitude < 95.0"
    assert (centroids.x <= INDONESIA_BOUNDS["max_lon"]).all(), "Feature centroid longitude > 141.5"
    assert (centroids.y >= INDONESIA_BOUNDS["min_lat"]).all(), "Feature centroid latitude < -11.5"
    assert (centroids.y <= INDONESIA_BOUNDS["max_lat"]).all(), "Feature centroid latitude > 6.5"
