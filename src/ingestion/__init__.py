"""Ingestion module for loading, parsing, and validating PODES, IRBI, and spatial boundaries."""

from src.ingestion.parser import parse_and_align_datasets, load_csv_metrics, load_shapefile_geometries
from src.ingestion.validator import (
    validate_tabular_data,
    validate_spatial_data,
    validate_file_size,
    validate_pipeline_outputs,
)

__all__ = [
    "parse_and_align_datasets",
    "load_csv_metrics",
    "load_shapefile_geometries",
    "validate_tabular_data",
    "validate_spatial_data",
    "validate_file_size",
    "validate_pipeline_outputs",
]
