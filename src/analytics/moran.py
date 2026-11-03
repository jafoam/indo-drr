"""Global Moran's I and Local Indicators of Spatial Association (LISA) analytics."""
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union
import geopandas as gpd
import numpy as np
import pandas as pd
import esda.moran
from libpysal.weights import W

from src.config import LISA_PALETTE
from src.analytics.weights import build_spatial_weights


@dataclass
class MoranResult:
    """Dataclass encapsulating Global Moran's I spatial autocorrelation metrics."""
    variable: str
    moran_i: float
    p_value: float
    z_score: float
    expected_i: float
    simulations: int = 999

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style key indexing for backwards compatibility."""
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary."""
        return asdict(self)


def compute_morans_i(
    df: Union[gpd.GeoDataFrame, pd.DataFrame],
    column: str,
    weights: Optional[W] = None,
    permutations: int = 999,
) -> MoranResult:
    """Compute Global Moran's I spatial autocorrelation for a given column.

    Args:
        df: GeoDataFrame or DataFrame containing the column to evaluate.
        column: Column name of interest (e.g. 'IRBI_2024', 'EWS').
        weights: PySAL spatial weights matrix. If None and df is a GeoDataFrame, builds KNN(k=5).
        permutations: Number of random permutations for Monte Carlo p-value simulation.

    Returns:
        MoranResult with moran_i, p_value, z_score, and expected_i.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe.")

    if weights is None:
        if not isinstance(df, gpd.GeoDataFrame) or "geometry" not in df.columns:
            raise ValueError("Spatial weights matrix must be provided when df has no geometry.")
        weights = build_spatial_weights(df, k=5)

    y = df[column].to_numpy(dtype=float)

    # Handle edge case: Zero variance / constant variable across units
    if np.all(np.isnan(y)) or np.nanstd(y) == 0.0:
        n = len(y)
        ei = -1.0 / (n - 1) if n > 1 else 0.0
        return MoranResult(
            variable=column,
            moran_i=0.0,
            p_value=1.0,
            z_score=0.0,
            expected_i=float(ei),
            simulations=permutations,
        )

    # PySAL Moran computation
    mi = esda.moran.Moran(y, weights, permutations=permutations)

    return MoranResult(
        variable=column,
        moran_i=float(mi.I),
        p_value=float(mi.p_sim),
        z_score=float(mi.z_sim),
        expected_i=float(mi.EI),
        simulations=permutations,
    )


def compute_lisa(
    gdf: gpd.GeoDataFrame,
    column: str,
    weights: Optional[W] = None,
    p_threshold: float = 0.05,
    seed: int = 42,
    permutations: int = 999,
) -> gpd.GeoDataFrame:
    """Compute Local Indicators of Spatial Association (LISA) for local cluster identification.

    Categories:
        - High-High: Hotspot (High value surrounded by high values)
        - Low-Low: Coldspot (Low value surrounded by low values)
        - High-Low: Spatial outlier (High value surrounded by low values)
        - Low-High: Spatial outlier (Low value surrounded by high values)
        - Not Significant: p > p_threshold

    Args:
        gdf: GeoDataFrame containing target column and geometries.
        column: Attribute column to analyze.
        weights: PySAL spatial weights matrix. If None, builds KNN(k=5).
        p_threshold: Significance alpha cutoff (default: 0.05).
        seed: Random seed for reproducible Monte Carlo permutations.
        permutations: Number of permutations for local simulation.

    Returns:
        GeoDataFrame copy augmented with:
            lisa_i, p_value, z_score, lisa_q, lisa_cluster, color.
    """
    if column not in gdf.columns:
        raise KeyError(f"Column '{column}' not found in GeoDataFrame.")

    if weights is None:
        weights = build_spatial_weights(gdf, k=5)

    y = gdf[column].to_numpy(dtype=float)

    # Run Local Moran's I
    lm = esda.moran.Moran_Local(
        y,
        weights,
        transformation="r",
        permutations=permutations,
        seed=seed,
    )

    # Quadrant mapping: PySAL scheme (geoda_quads=False)
    # 1: High-High, 2: Low-High, 3: Low-Low, 4: High-Low
    quad_labels = {
        1: "High-High",
        2: "Low-High",
        3: "Low-Low",
        4: "High-Low",
    }

    n = len(gdf)
    clusters = []
    colors = []

    for i in range(n):
        p_sim = lm.p_sim[i]
        q = lm.q[i]

        if p_sim <= p_threshold and q in quad_labels:
            label = quad_labels[q]
        else:
            label = "Not Significant"

        clusters.append(label)
        # Match color from palette (handles both Title-Case and kebab-case keys)
        color = LISA_PALETTE.get(label) or LISA_PALETTE.get(label.lower().replace(" ", "-"), "#334155")
        colors.append(color)

    result = gdf.copy()
    result["lisa_i"] = lm.Is.tolist()
    result["p_value"] = lm.p_sim.tolist()
    result["z_score"] = lm.z_sim.tolist()
    result["lisa_q"] = lm.q.tolist()
    result["lisa_cluster"] = clusters
    result["color"] = colors

    return result
