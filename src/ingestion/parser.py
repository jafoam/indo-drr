"""Parser module for PODES/IRBI CSV datasets and boundary shapefiles."""
from pathlib import Path
from typing import Optional, Tuple
import geopandas as gpd
import pandas as pd
from shapely import set_precision
from shapely.ops import unary_union
from shapely.validation import make_valid

from src.config import DATA_RAW_DIR


def load_csv_metrics(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and clean tabular metrics from DATA_V#.csv."""
    if csv_path is None:
        csv_path = DATA_RAW_DIR / "DATA_V#.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Clean KODE_WIL to standard integer
    df["KODE_WIL"] = pd.to_numeric(df["KODE_WIL"], errors="raise").astype(int)

    # Convert string-encoded count columns with empty spaces to numeric
    for col in ["KEJADIAN_BENCANA", "TERDAMPAK"]:
        if col in df.columns:
            cleaned = df[col].astype(str).str.strip()
            df[col] = pd.to_numeric(cleaned, errors="coerce").fillna(0).astype(int)

    return df


def load_shapefile_geometries(shp_path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load shapefile geometries and ensure EPSG:4326 CRS."""
    if shp_path is None:
        shp_path = DATA_RAW_DIR / "GEODA" / "MITIGASI BENCANA.shp"

    if not shp_path.exists():
        raise FileNotFoundError(f"Raw Shapefile not found at {shp_path}")

    gdf = gpd.read_file(shp_path)

    # Clean KODE_WIL to integer
    gdf["KODE_WIL"] = pd.to_numeric(gdf["KODE_WIL"], errors="raise").astype(int)

    # Set or convert CRS to EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def repair_and_simplify_geometries(
    gdf: gpd.GeoDataFrame,
    tolerance: float = 0.005,
    grid_size: float = 0.0001,
) -> gpd.GeoDataFrame:
    """Repair invalid topologies, simplify boundaries, and apply coordinate precision."""
    gdf = gdf.copy()

    # Step 1: Repair invalid input topologies
    gdf["geometry"] = gdf.geometry.apply(make_valid)

    # Step 2: Simplify with topology preservation
    gdf["geometry"] = gdf.geometry.simplify(tolerance=tolerance, preserve_topology=True)

    # Step 3: Set precision to optimize coordinate byte size while retaining ~10m resolution
    if grid_size is not None:
        gdf["geometry"] = gdf.geometry.apply(lambda g: set_precision(g, grid_size=grid_size))

    # Step 4: Final validity repair and filter to pure Polygons/MultiPolygons
    def _clean_geometry(geom):
        valid_geom = make_valid(geom)
        if valid_geom.geom_type in ("Polygon", "MultiPolygon"):
            return valid_geom
        if valid_geom.geom_type == "GeometryCollection":
            polys = [p for p in valid_geom.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
            if polys:
                return unary_union(polys)
        return valid_geom

    gdf["geometry"] = gdf["geometry"].apply(_clean_geometry)
    return gdf


def parse_and_align_datasets(
    csv_path: Optional[Path] = None,
    shp_path: Optional[Path] = None,
    tolerance: float = 0.005,
    grid_size: float = 0.0001,
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Parse, repair, align, and join CSV metrics with Shapefile geometry metadata.

    Returns:
        Tuple[pd.DataFrame, gpd.GeoDataFrame]:
            - clean_df: 514-row tabular DataFrame with enriched KAB_KOTA and PROVINSI names.
            - clean_gdf: 514-feature GeoDataFrame with EPSG:4326 simplified geometries.
    """
    df_raw = load_csv_metrics(csv_path)
    gdf_raw = load_shapefile_geometries(shp_path)

    # Extract clean regency/province metadata from shapefile
    metadata_cols = ["KODE_WIL", "KAB_KOTA", "PROVINSI"]
    meta_df = gdf_raw[metadata_cols].drop_duplicates(subset=["KODE_WIL"]).copy()

    # Merge metadata to CSV tabular dataset
    # Drop original CSV KAB_KOTA (which contains dummy zeros) before merging
    cols_to_merge = df_raw.drop(columns=["KAB_KOTA"], errors="ignore")
    clean_df = pd.merge(meta_df, cols_to_merge, on="KODE_WIL", how="inner")

    # Add IS_KOTA indicator (1 for City/Kota, 0 for Regency/Kabupaten)
    clean_df["IS_KOTA"] = clean_df["KAB_KOTA"].astype(str).str.startswith("Kota ").astype(int)

    # Reorder columns: identifiers and names first, then metrics
    primary_cols = ["KODE_WIL", "KAB_KOTA", "PROVINSI", "IS_KOTA"]
    remaining_cols = [c for c in clean_df.columns if c not in primary_cols]
    clean_df = clean_df[primary_cols + remaining_cols].sort_values("KODE_WIL").reset_index(drop=True)

    # Prepare spatial GeoDataFrame
    clean_gdf = gdf_raw[["KODE_WIL", "KAB_KOTA", "PROVINSI", "geometry"]].copy()
    clean_gdf["IS_KOTA"] = clean_gdf["KAB_KOTA"].astype(str).str.startswith("Kota ").astype(int)
    clean_gdf = clean_gdf.sort_values("KODE_WIL").reset_index(drop=True)

    # Simplify and repair geometries for lightweight GeoJSON distribution
    clean_gdf = repair_and_simplify_geometries(clean_gdf, tolerance=tolerance, grid_size=grid_size)

    return clean_df, clean_gdf
