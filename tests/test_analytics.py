"""Tests for Core Spatial Analytics Module (Task 1.3)."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from src.config import (
    GEOJSON_FILE,
    PARQUET_FILE,
    LISA_PALETTE,
    BIVARIATE_MATRIX_HEX,
)
from src.analytics.weights import build_spatial_weights
from src.analytics.moran import compute_morans_i, compute_lisa
from src.analytics.bivariate import compute_bivariate_scores


@pytest.fixture(scope="module")
def merged_dataset() -> gpd.GeoDataFrame:
    """Load and merge 514-regency GeoJSON geometry with tabular indicators."""
    gdf = gpd.read_file(GEOJSON_FILE)
    df = pd.read_parquet(PARQUET_FILE)
    # Merge ensuring 514 records and consistent ordering
    merged = gdf.merge(df, on="KODE_WIL", suffixes=("", "_df"))
    assert len(merged) == 514
    return merged


def test_spatial_weights_k5_properties(merged_dataset):
    """Verify spatial weights matrix has 514 nodes, 0 islands, degree > 0, and row standardization."""
    w = build_spatial_weights(merged_dataset, k=5)

    # 514 nodes
    assert w.n == 514, f"Expected 514 nodes in weight matrix, got {w.n}"

    # 0 islands
    assert len(w.islands) == 0, f"Expected 0 islands, found {len(w.islands)} islands"

    # Degree > 0 for each node
    for node, neighbors in w.neighbors.items():
        assert len(neighbors) == 5, f"Node {node} expected degree 5, got {len(neighbors)}"

    # Row-standardized transformation ('R')
    assert w.transform.upper() == "R", f"Expected transform 'R', got {w.transform}"

    # Verify caching: repeated call with same data returns valid weights
    w_cached = build_spatial_weights(merged_dataset, k=5)
    assert w_cached.n == 514


def test_global_morans_i_irbi_and_ews(merged_dataset):
    """Verify Global Moran's I calculation for IRBI_2024 and EWS."""
    w = build_spatial_weights(merged_dataset, k=5)

    # Test on IRBI_2024
    res_irbi = compute_morans_i(merged_dataset, "IRBI_2024", weights=w)
    assert hasattr(res_irbi, "moran_i") or "moran_i" in res_irbi
    moran_i_val = res_irbi["moran_i"] if isinstance(res_irbi, dict) else res_irbi.moran_i
    p_val = res_irbi["p_value"] if isinstance(res_irbi, dict) else res_irbi.p_value
    z_score = res_irbi["z_score"] if isinstance(res_irbi, dict) else res_irbi.z_score
    expected_i = res_irbi["expected_i"] if isinstance(res_irbi, dict) else res_irbi.expected_i

    assert -1.0 <= moran_i_val <= 1.0, f"Moran's I out of bounds [-1, 1]: {moran_i_val}"
    assert 0.0 <= p_val <= 1.0, f"p-value out of bounds [0, 1]: {p_val}"
    assert isinstance(z_score, float)
    assert isinstance(expected_i, float)
    # IRBI exhibits positive spatial autocorrelation
    assert moran_i_val > 0.0

    # Test on EWS
    res_ews = compute_morans_i(merged_dataset, "EWS", weights=w)
    moran_i_ews = res_ews["moran_i"] if isinstance(res_ews, dict) else res_ews.moran_i
    p_val_ews = res_ews["p_value"] if isinstance(res_ews, dict) else res_ews.p_value

    assert -1.0 <= moran_i_ews <= 1.0, f"Moran's I out of bounds [-1, 1]: {moran_i_ews}"
    assert 0.0 <= p_val_ews <= 1.0, f"p-value out of bounds [0, 1]: {p_val_ews}"
    # EWS exhibits positive spatial autocorrelation (infrastructure clusters)
    assert moran_i_ews > 0.0


def test_lisa_calculation_and_clustering(merged_dataset):
    """Verify LISA local spatial autocorrelation returns 514 records, valid clusters, and palette colors."""
    w = build_spatial_weights(merged_dataset, k=5)
    lisa_df = compute_lisa(merged_dataset, "IRBI_2024", weights=w, p_threshold=0.05)

    assert len(lisa_df) == 514, f"Expected 514 LISA records, got {len(lisa_df)}"
    assert "lisa_cluster" in lisa_df.columns, "Missing 'lisa_cluster' column"
    assert "p_value" in lisa_df.columns, "Missing 'p_value' column"
    assert "color" in lisa_df.columns, "Missing 'color' column"

    valid_clusters = {"High-High", "Low-Low", "High-Low", "Low-High", "Not Significant"}
    unique_clusters = set(lisa_df["lisa_cluster"].unique())
    assert unique_clusters.issubset(valid_clusters), f"Unexpected cluster values: {unique_clusters - valid_clusters}"

    # Verify colors match LISA_PALETTE
    for _, row in lisa_df.iterrows():
        cluster = row["lisa_cluster"]
        expected_color = (
            LISA_PALETTE.get(cluster)
            or LISA_PALETTE.get(cluster.lower().replace(" ", "-"))
        )
        assert row["color"] == expected_color, f"Color mismatch for {cluster}: {row['color']} != {expected_color}"

    # Verify not all observations are Not Significant
    assert (lisa_df["lisa_cluster"] != "Not Significant").sum() > 0


def test_bivariate_scores_and_matrix(merged_dataset):
    """Verify composite risk, mitigation, gap scores, tercile categorization, and matrix colors."""
    bivariate_df = compute_bivariate_scores(merged_dataset)

    assert len(bivariate_df) == 514

    required_cols = [
        "risk_score",
        "mitigation_score",
        "gap_score",
        "bivariate_category",
        "bivariate_color",
    ]
    for col in required_cols:
        assert col in bivariate_df.columns, f"Missing required column {col}"

    # Risk score in [0, 1]
    assert (bivariate_df["risk_score"] >= 0.0).all(), "risk_score < 0.0 detected"
    assert (bivariate_df["risk_score"] <= 1.0).all(), "risk_score > 1.0 detected"

    # Mitigation score in [0, 1]
    assert (bivariate_df["mitigation_score"] >= 0.0).all(), "mitigation_score < 0.0 detected"
    assert (bivariate_df["mitigation_score"] <= 1.0).all(), "mitigation_score > 1.0 detected"

    # Gap score = risk - mitigation in [-1, 1]
    assert (bivariate_df["gap_score"] >= -1.0).all(), "gap_score < -1.0 detected"
    assert (bivariate_df["gap_score"] <= 1.0).all(), "gap_score > 1.0 detected"
    diff = (bivariate_df["risk_score"] - bivariate_df["mitigation_score"]) - bivariate_df["gap_score"]
    assert np.allclose(diff, 0.0), "gap_score does not match (risk_score - mitigation_score)"

    # Bivariate 3x3 categories: R{1..3}-M{1..3}
    valid_categories = {
        f"R{r}-M{m}" for r in (1, 2, 3) for m in (1, 2, 3)
    }
    unique_cats = set(bivariate_df["bivariate_category"].unique())
    assert unique_cats.issubset(valid_categories), f"Unexpected categories: {unique_cats - valid_categories}"

    # Verify bivariate color matching BIVARIATE_MATRIX_HEX
    for _, row in bivariate_df.iterrows():
        cat = row["bivariate_category"]
        r_part, m_part = cat.split("-")
        expected_color = (
            BIVARIATE_MATRIX_HEX.get((r_part, m_part))
            or BIVARIATE_MATRIX_HEX.get(cat)
        )
        assert row["bivariate_color"] == expected_color, f"Color mismatch for {cat}: {row['bivariate_color']} != {expected_color}"

    # Verify critical gap flag exists
    if "is_critical_gap" in bivariate_df.columns:
        critical_mask = bivariate_df["bivariate_category"] == "R3-M1"
        assert (bivariate_df["is_critical_gap"] == critical_mask).all()


def test_moran_zero_variance_handling(merged_dataset):
    """Verify Moran's I handles zero-variance edge cases gracefully without uncaught exceptions."""
    w = build_spatial_weights(merged_dataset, k=5)
    df_copy = merged_dataset.copy()
    df_copy["CONSTANT_VAL"] = 42.0

    res = compute_morans_i(df_copy, "CONSTANT_VAL", weights=w)
    moran_i_val = res["moran_i"] if isinstance(res, dict) else res.moran_i
    assert moran_i_val == 0.0 or np.isnan(moran_i_val)


def test_hybrid_threshold_matrix_tanimbar(merged_dataset):
    """Verify hybrid threshold matrix applies normative cutoffs and classifies Kepulauan Tanimbar (8103) as R3-M1."""
    bivariate_df = compute_bivariate_scores(merged_dataset)

    # Kepulauan Tanimbar (8103) has low mitigation (<0.30) and high hazard risk (R3)
    tanimbar = bivariate_df[bivariate_df["KODE_WIL"] == 8103].iloc[0]
    assert tanimbar["mitigation_score"] < 0.30
    assert tanimbar["mitigation_tercile"] == "M1"
    assert tanimbar["risk_tercile"] == "R3"
    assert tanimbar["bivariate_category"] == "R3-M1"
    assert tanimbar["is_critical_gap"] is True or tanimbar["is_critical_gap"] == 1

    # National distribution checks
    cat_counts = bivariate_df["bivariate_category"].value_counts().to_dict()
    assert cat_counts["R3-M1"] == 148
    assert cat_counts["R2-M1"] == 137
    assert cat_counts["R1-M1"] == 125


def test_readiness_balance_and_adequacy_ratio(merged_dataset):
    """Verify Net Readiness Balance and Readiness Adequacy Ratio calculation."""
    bivariate_df = compute_bivariate_scores(merged_dataset)

    assert "readiness_balance" in bivariate_df.columns
    assert "adequacy_ratio" in bivariate_df.columns

    # readiness_balance = mitigation_score - risk_score in [-1.0, 1.0]
    expected_balance = np.clip(
        bivariate_df["mitigation_score"] - bivariate_df["risk_score"], -1.0, 1.0
    )
    assert np.allclose(bivariate_df["readiness_balance"], expected_balance)

    # adequacy_ratio in [0.0, 200.0]
    assert (bivariate_df["adequacy_ratio"] >= 0.0).all()
    assert (bivariate_df["adequacy_ratio"] <= 200.0).all()

    # For positive risk, verify ratio is approximately (M / R) * 100
    pos_risk = bivariate_df[bivariate_df["risk_score"] > 0.05]
    expected_ratio = np.clip((pos_risk["mitigation_score"] / pos_risk["risk_score"]) * 100.0, 0.0, 200.0)
    assert np.allclose(pos_risk["adequacy_ratio"], expected_ratio)


