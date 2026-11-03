"""Validator module for tabular metrics and spatial GeoJSON outputs."""
import os
from pathlib import Path
from typing import List, Optional
import geopandas as gpd
import pandas as pd

from src.config import INDONESIA_BOUNDS

EXPECTED_UNIT_COUNT = 514
CORE_METRIC_COLUMNS = ["KODE_WIL", "IRBI_2024", "EWS", "RAMBU", "PERKAP"]
MAX_GEOJSON_BYTES = 5_242_880  # 5 MB


class DataValidationError(ValueError):
    """Raised when data fails integrity or schema checks."""
    pass


def validate_tabular_data(
    df: pd.DataFrame,
    expected_rows: int = EXPECTED_UNIT_COUNT,
    required_cols: Optional[List[str]] = None,
) -> bool:
    """Validate 514 administrative rows, uniqueness, and absence of nulls in core metrics."""
    if required_cols is None:
        required_cols = CORE_METRIC_COLUMNS

    if len(df) != expected_rows:
        raise DataValidationError(f"Expected {expected_rows} rows, found {len(df)}")

    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns in tabular dataset: {missing_cols}")

    # Check nulls in required columns
    for col in required_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise DataValidationError(f"Column '{col}' has {null_count} missing/null values")

    # Check uniqueness of primary key KODE_WIL
    if df["KODE_WIL"].nunique() != expected_rows:
        raise DataValidationError(
            f"KODE_WIL must be unique. Found {df['KODE_WIL'].nunique()} unique values out of {expected_rows}"
        )

    # Check presence and non-emptiness of metadata
    for meta_col in ["KAB_KOTA", "PROVINSI"]:
        if meta_col in df.columns:
            empty_count = (df[meta_col].astype(str).str.strip() == "").sum()
            if empty_count > 0:
                raise DataValidationError(f"Metadata column '{meta_col}' has {empty_count} blank entries")

    return True


def validate_spatial_data(
    gdf: gpd.GeoDataFrame,
    expected_features: int = EXPECTED_UNIT_COUNT,
    bounds: Optional[dict] = None,
) -> bool:
    """Validate GeoDataFrame completeness, CRS, geometry validity, and bounding box compliance."""
    if bounds is None:
        bounds = INDONESIA_BOUNDS

    if len(gdf) != expected_features:
        raise DataValidationError(f"Expected {expected_features} spatial features, found {len(gdf)}")

    # Validate CRS is EPSG:4326
    if gdf.crs is None or (gdf.crs.to_epsg() != 4326 and "4326" not in str(gdf.crs)):
        raise DataValidationError(f"Expected CRS EPSG:4326, got {gdf.crs}")

    # Validate geometry validity and non-emptiness
    if not gdf.geometry.is_valid.all():
        invalid_count = (~gdf.geometry.is_valid).sum()
        raise DataValidationError(f"Found {invalid_count} invalid geometries in spatial dataset")

    if gdf.geometry.is_empty.any():
        empty_count = gdf.geometry.is_empty.sum()
        raise DataValidationError(f"Found {empty_count} empty geometries in spatial dataset")

    # Validate bounding box: total bounds (allowing 0.1 deg margin for Aceh outer islands)
    minx, miny, maxx, maxy = gdf.total_bounds
    margin = 0.1
    if minx < bounds["min_lon"] - margin or maxx > bounds["max_lon"] + margin:
        raise DataValidationError(
            f"Longitude bounds [{minx:.4f}, {maxx:.4f}] exceed allowed range [{bounds['min_lon']}, {bounds['max_lon']}]"
        )
    if miny < bounds["min_lat"] - margin or maxy > bounds["max_lat"] + margin:
        raise DataValidationError(
            f"Latitude bounds [{miny:.4f}, {maxy:.4f}] exceed allowed range [{bounds['min_lat']}, {bounds['max_lat']}]"
        )

    # Validate feature centroids strictly fall within Indonesia bounds
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*geographic CRS.*")
        centroids = gdf.geometry.centroid
    if not ((centroids.x >= bounds["min_lon"]) & (centroids.x <= bounds["max_lon"])).all():
        raise DataValidationError("One or more feature centroids have longitude outside [95.0, 141.5]")
    if not ((centroids.y >= bounds["min_lat"]) & (centroids.y <= bounds["max_lat"])).all():
        raise DataValidationError("One or more feature centroids have latitude outside [-11.5, 6.5]")

    return True


def validate_file_size(file_path: Path, max_bytes: int = MAX_GEOJSON_BYTES) -> bool:
    """Validate that the file exists and is strictly under the byte size limit."""
    if not file_path.exists():
        raise FileNotFoundError(f"Target file not found at {file_path}")

    actual_size = os.path.getsize(file_path)
    if actual_size >= max_bytes:
        raise DataValidationError(
            f"File size {actual_size} bytes exceeds maximum allowed limit {max_bytes} bytes"
        )
    return True


def validate_pipeline_outputs(
    parquet_path: Path,
    geojson_path: Path,
    max_geojson_bytes: int = MAX_GEOJSON_BYTES,
) -> bool:
    """End-to-end validator for generated Parquet and GeoJSON files."""
    # 1. Parquet validation
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file missing: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    validate_tabular_data(df)

    # 2. GeoJSON validation
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file missing: {geojson_path}")
    validate_file_size(geojson_path, max_bytes=max_geojson_bytes)
    gdf = gpd.read_file(geojson_path)
    validate_spatial_data(gdf)

    return True
