"""Regional Corridor Mapping and Localized Spatial Weights Analytics.

Groups Indonesia's 38 provinces into 7 analytical corridors and manages
localized spatial connectivity matrices guaranteeing non-singular, zero-island
spatial autocorrelation analysis for archipelagic sub-regions.
"""
from typing import Dict, List, Optional, Union
import geopandas as gpd
import pandas as pd
from libpysal.weights import W

from src.analytics.weights import build_spatial_weights
from src.analytics.moran import compute_morans_i, MoranResult

# Standard 7 Archipelagic Regional Corridors
CORRIDORS: List[str] = [
    "Sumatera",
    "Jawa-Bali",
    "Kalimantan",
    "Sulawesi",
    "Nusa Tenggara",
    "Maluku",
    "Papua",
]

# Official 38-Province to Regional Corridor Registry
CORRIDOR_MAPPING: Dict[str, str] = {
    # Sumatera (10 provinces, 154 regencies)
    "Aceh": "Sumatera",
    "Sumatera Utara": "Sumatera",
    "Sumatera Barat": "Sumatera",
    "Riau": "Sumatera",
    "Kepulauan Riau": "Sumatera",
    "Jambi": "Sumatera",
    "Sumatera Selatan": "Sumatera",
    "Kepulauan Bangka Belitung": "Sumatera",
    "Bengkulu": "Sumatera",
    "Lampung": "Sumatera",
    # Jawa-Bali (7 provinces, 128 regencies)
    "DKI Jakarta": "Jawa-Bali",
    "Jawa Barat": "Jawa-Bali",
    "Jawa Tengah": "Jawa-Bali",
    "Daerah Istimewa Yogyakarta": "Jawa-Bali",
    "Jawa Timur": "Jawa-Bali",
    "Banten": "Jawa-Bali",
    "Bali": "Jawa-Bali",
    # Kalimantan (5 provinces, 56 regencies)
    "Kalimantan Barat": "Kalimantan",
    "Kalimantan Tengah": "Kalimantan",
    "Kalimantan Selatan": "Kalimantan",
    "Kalimantan Timur": "Kalimantan",
    "Kalimantan Utara": "Kalimantan",
    # Sulawesi (6 provinces, 81 regencies)
    "Sulawesi Utara": "Sulawesi",
    "Gorontalo": "Sulawesi",
    "Sulawesi Tengah": "Sulawesi",
    "Sulawesi Barat": "Sulawesi",
    "Sulawesi Selatan": "Sulawesi",
    "Sulawesi Tenggara": "Sulawesi",
    # Nusa Tenggara (2 provinces, 32 regencies)
    "Nusa Tenggara Barat": "Nusa Tenggara",
    "Nusa Tenggara Timur": "Nusa Tenggara",
    # Maluku (2 provinces, 21 regencies)
    "Maluku": "Maluku",
    "Maluku Utara": "Maluku",
    # Papua (6 provinces, 42 regencies)
    "Papua": "Papua",
    "Papua Barat": "Papua",
    "Papua Barat Daya": "Papua",
    "Papua Pegunungan": "Papua",
    "Papua Selatan": "Papua",
    "Papua Tengah": "Papua",
    # Common variant aliases for robustness
    "DI Yogyakarta": "Jawa-Bali",
    "D.I. Yogyakarta": "Jawa-Bali",
    "Yogyakarta": "Jawa-Bali",
    "Kep Bangka Belitung": "Sumatera",
    "Bangka Belitung": "Sumatera",
    "Kep. Bangka Belitung": "Sumatera",
    "Kep Riau": "Sumatera",
    "Kep. Riau": "Sumatera",
    "NTB": "Nusa Tenggara",
    "NTT": "Nusa Tenggara",
}


def assign_corridors(
    df: Union[gpd.GeoDataFrame, pd.DataFrame]
) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """Assign each administrative record to one of the 7 regional corridors.

    Args:
        df: GeoDataFrame or DataFrame containing a 'PROVINSI' column.

    Returns:
        Augmented copy with 'corridor' column populated.

    Raises:
        KeyError: If 'PROVINSI' column is missing.
        ValueError: If any province fails to map to a recognized corridor.
    """
    if "PROVINSI" not in df.columns:
        raise KeyError("Dataframe must contain 'PROVINSI' column for corridor assignment.")

    res = df.copy()
    res["corridor"] = res["PROVINSI"].map(CORRIDOR_MAPPING)

    unassigned = res[res["corridor"].isna()]
    if len(unassigned) > 0:
        missing_provs = unassigned["PROVINSI"].unique().tolist()
        raise ValueError(f"Unmapped provinces found: {missing_provs}")

    return res


def get_corridor_subgraph(
    gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
    corridor_name: str,
) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """Extract filtered subgraph corresponding to a regional corridor.

    Args:
        gdf: GeoDataFrame or DataFrame of administrative units.
        corridor_name: One of the 7 corridors in CORRIDORS.

    Returns:
        Filtered GeoDataFrame or DataFrame.

    Raises:
        ValueError: If corridor_name is invalid or has 0 matching units.
    """
    if corridor_name not in CORRIDORS:
        raise ValueError(
            f"Invalid corridor '{corridor_name}'. Must be one of: {CORRIDORS}"
        )

    if "corridor" not in gdf.columns:
        work_gdf = assign_corridors(gdf)
    else:
        work_gdf = gdf

    sub_gdf = work_gdf[work_gdf["corridor"] == corridor_name].copy()
    if len(sub_gdf) == 0:
        raise ValueError(f"No records found for corridor '{corridor_name}'.")

    return sub_gdf


def get_corridor_weights(
    corridor_gdf: gpd.GeoDataFrame,
    k: Optional[int] = None,
    use_cache: bool = True,
    transform: str = "R",
) -> W:
    """Build localized spatial weights matrix bounded to sub-regional graph.

    Archipelagic Connectivity Constraint:
        k is bounded dynamically to min(5, N_sub - 1) to guarantee zero isolated
        islands even in smaller archipelagic sub-graphs (e.g. Maluku).

    Args:
        corridor_gdf: GeoDataFrame containing administrative units in corridor.
        k: Optional neighbor count override. Defaults to min(5, len(corridor_gdf) - 1).
        use_cache: If True, uses cached weights matrix for identical geometry subsets.
        transform: Row standardization scheme (default 'R').

    Returns:
        libpysal.weights.W spatial weights matrix with zero islands.
    """
    n_units = len(corridor_gdf)
    if n_units <= 1:
        raise ValueError(f"Corridor requires >= 2 spatial units for weights, got {n_units}.")

    if k is None:
        effective_k = min(5, n_units - 1)
    else:
        effective_k = min(k, n_units - 1)

    return build_spatial_weights(
        corridor_gdf,
        k=effective_k,
        use_cache=use_cache,
        transform=transform,
    )


def compute_corridor_moran(
    corridor_gdf: gpd.GeoDataFrame,
    variable: str = "IRBI_2024",
    weights: Optional[W] = None,
    permutations: int = 999,
) -> MoranResult:
    """Compute localized Global Moran's I spatial autocorrelation for a regional corridor.

    Args:
        corridor_gdf: GeoDataFrame subset for a specific corridor.
        variable: Indicator column name (default 'IRBI_2024').
        weights: Optional precomputed spatial weights. If None, generated via get_corridor_weights.
        permutations: Number of Monte Carlo permutations for p-value estimation.

    Returns:
        MoranResult with localized moran_i, p_value, z_score, and expected_i.
    """
    if weights is None:
        weights = get_corridor_weights(corridor_gdf)

    return compute_morans_i(
        corridor_gdf,
        column=variable,
        weights=weights,
        permutations=permutations,
    )
