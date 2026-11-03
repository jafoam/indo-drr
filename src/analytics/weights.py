"""Spatial weights matrix construction for Indonesia archipelagic geography."""
import warnings
from typing import Dict, Optional, Tuple
import geopandas as gpd
import libpysal
from libpysal.weights import W

_WEIGHTS_CACHE: Dict[Tuple[int, int, Tuple], W] = {}


def build_spatial_weights(
    gdf: gpd.GeoDataFrame,
    k: int = 5,
    use_cache: bool = True,
    transform: str = "R",
) -> W:
    """Build spatial weight matrix using K-Nearest Neighbors to guarantee island connectivity.

    In archipelagic Indonesia, Queen/Rook contiguity produces disconnected island singletons.
    KNN (k=5) ensures every regency has exactly 5 neighbors and zero islands.

    Args:
        gdf: GeoDataFrame containing administrative unit polygons or centroids.
        k: Number of nearest neighbors (default: 5).
        use_cache: If True, reuse previously built spatial weights for matching input.
        transform: Standardization scheme (default 'R' for row-standardized).

    Returns:
        libpysal.weights.W row-standardized spatial weights matrix.
    """
    if len(gdf) == 0:
        raise ValueError("Cannot construct spatial weights for empty GeoDataFrame.")

    cache_key = None
    if use_cache:
        # Cache key based on length, k, and identifier tuple if available
        id_tuple = (
            tuple(gdf["KODE_WIL"].tolist())
            if "KODE_WIL" in gdf.columns
            else tuple(gdf.index.tolist())
        )
        cache_key = (len(gdf), k, id_tuple)
        if cache_key in _WEIGHTS_CACHE:
            return _WEIGHTS_CACHE[cache_key]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # KNN.from_dataframe automatically computes centroids for polygon geometries
        w = libpysal.weights.KNN.from_dataframe(gdf, k=k)

    # Apply row-standardization
    w.transform = transform

    if use_cache and cache_key is not None:
        _WEIGHTS_CACHE[cache_key] = w

    return w


def clear_weights_cache() -> None:
    """Clear in-memory spatial weights cache."""
    _WEIGHTS_CACHE.clear()
