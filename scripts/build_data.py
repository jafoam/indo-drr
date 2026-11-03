"""Data pipeline build script: Ingest raw PODES/IRBI and boundary shapefiles.

Outputs:
  - data/processed/podes_irbi_clean.parquet (clean analytical tabular data)
  - data/processed/indonesia_regencies.geojson (EPSG:4326 simplified spatial polygons < 5MB)
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, GEOJSON_FILE, PARQUET_FILE
from src.ingestion.parser import parse_and_align_datasets
from src.ingestion.validator import (
    validate_tabular_data,
    validate_spatial_data,
    validate_pipeline_outputs,
)


def run_pipeline():
    start_time = time.time()
    print("=" * 60)
    print("Starting Ingestion & Validation Pipeline...")
    print("=" * 60)

    # 1. Parse and align datasets
    print("[1/4] Parsing and simplifying raw CSV and Shapefile datasets...")
    clean_df, clean_gdf = parse_and_align_datasets(
        tolerance=0.005,
        grid_size=0.0001,
    )
    print(f"      Parsed {len(clean_df)} tabular rows and {len(clean_gdf)} spatial features.")

    # 2. In-memory validation
    print("[2/4] Validating in-memory datasets...")
    validate_tabular_data(clean_df)
    validate_spatial_data(clean_gdf)
    print("      In-memory integrity validation passed.")

    # 3. Write outputs to disk
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] Persisting outputs...")
    print(f"      Writing parquet -> {PARQUET_FILE}")
    clean_df.to_parquet(PARQUET_FILE, index=False)

    print(f"      Writing geojson -> {GEOJSON_FILE}")
    clean_gdf.to_file(GEOJSON_FILE, driver="GeoJSON")

    # 4. End-to-end on-disk validation
    print("[4/4] Validating written artifacts on disk...")
    validate_pipeline_outputs(PARQUET_FILE, GEOJSON_FILE)

    parquet_size = PARQUET_FILE.stat().st_size
    geojson_size = GEOJSON_FILE.stat().st_size
    elapsed = time.time() - start_time

    print("=" * 60)
    print("Pipeline Execution Completed Successfully!")
    print(f"Elapsed Time: {elapsed:.2f} seconds")
    print(f"Parquet Path : {PARQUET_FILE}")
    print(f"Parquet Size : {parquet_size:,} bytes ({parquet_size / 1024:.2f} KB)")
    print(f"GeoJSON Path : {GEOJSON_FILE}")
    print(f"GeoJSON Size : {geojson_size:,} bytes ({geojson_size / 1024 / 1024:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
