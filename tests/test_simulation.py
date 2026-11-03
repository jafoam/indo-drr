"""Tests for Policy Intervention Simulation and Regional Corridor Analytics (Milestone 2 - Task 2.1)."""
import geopandas as gpd
import pandas as pd
import pytest

from src.config import GEOJSON_FILE, PARQUET_FILE, BIVARIATE_MATRIX_HEX
from src.analytics.bivariate import compute_bivariate_scores
from src.analytics.simulation import (
    simulate_policy_intervention,
    SimulationResult,
)
from src.analytics.corridors import (
    CORRIDOR_MAPPING,
    CORRIDORS,
    assign_corridors,
    get_corridor_subgraph,
    get_corridor_weights,
    compute_corridor_moran,
)
from src.analytics.moran import MoranResult


@pytest.fixture(scope="module")
def merged_dataset() -> gpd.GeoDataFrame:
    """Load and merge 514-regency GeoJSON geometry with tabular indicators."""
    gdf = gpd.read_file(GEOJSON_FILE)
    df = pd.read_parquet(PARQUET_FILE)
    merged = gdf.merge(df, on="KODE_WIL", suffixes=("", "_df"))
    assert len(merged) == 514
    return merged


@pytest.fixture(scope="module")
def bivariate_dataset(merged_dataset) -> gpd.GeoDataFrame:
    """Compute baseline bivariate scores for the 514 units."""
    return compute_bivariate_scores(merged_dataset)


def test_simulate_policy_intervention_baseline_zero(bivariate_dataset):
    """At 0% intervention (0, 0), returns baseline exactly (148 Critical Gap units, delta=0)."""
    result = simulate_policy_intervention(
        bivariate_dataset,
        signal_reduction_pct=0.0,
        gotong_royong_boost_pct=0.0,
    )

    assert isinstance(result, SimulationResult)
    assert result.baseline_critical_gaps == 148
    assert result.simulated_critical_gaps == 148
    assert result.resolved_gaps == 0

    sim_df = result.df
    assert len(sim_df) == 514
    assert sim_df["is_critical_gap_simulated"].sum() == 148

    # Simulated mitigation scores match baseline exactly
    pd.testing.assert_series_equal(
        sim_df["simulated_mitigation_score"],
        bivariate_dataset["mitigation_score"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sim_df["simulated_gap_score"],
        bivariate_dataset["gap_score"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sim_df["simulated_bivariate_category"],
        bivariate_dataset["bivariate_category"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sim_df["simulated_bivariate_color"],
        bivariate_dataset["bivariate_color"],
        check_names=False,
    )


def test_simulate_policy_intervention_active(bivariate_dataset):
    """At 50% signal reduction & 25% gotong royong boost, critical gap strictly decreases (resolves >= 10)."""
    result = simulate_policy_intervention(
        bivariate_dataset,
        signal_reduction_pct=50.0,
        gotong_royong_boost_pct=25.0,
    )

    assert isinstance(result, SimulationResult)
    assert result.baseline_critical_gaps == 148
    assert result.simulated_critical_gaps < 148
    assert result.resolved_gaps >= 10
    assert result.resolved_gaps == result.baseline_critical_gaps - result.simulated_critical_gaps

    sim_df = result.df
    assert len(sim_df) == 514

    # Verify score bounds
    assert sim_df["simulated_mitigation_score"].between(0.0, 1.0).all()
    assert sim_df["simulated_gap_score"].between(-1.0, 1.0).all()

    # Required simulated output columns
    required_cols = [
        "simulated_mitigation_score",
        "simulated_gap_score",
        "simulated_bivariate_category",
        "simulated_bivariate_color",
        "is_critical_gap_simulated",
    ]
    for col in required_cols:
        assert col in sim_df.columns, f"Missing column: {col}"

    # Verify valid categories and colors
    valid_categories = {
        "R1-M1", "R1-M2", "R1-M3",
        "R2-M1", "R2-M2", "R2-M3",
        "R3-M1", "R3-M2", "R3-M3",
    }
    assert set(sim_df["simulated_bivariate_category"].unique()).issubset(valid_categories)
    assert sim_df["simulated_bivariate_color"].isin(BIVARIATE_MATRIX_HEX.values()).all()


def test_simulate_policy_intervention_clipping(bivariate_dataset):
    """Ensure extreme interventions remain strictly bounded in [0.0, 1.0]."""
    result = simulate_policy_intervention(
        bivariate_dataset,
        signal_reduction_pct=100.0,
        gotong_royong_boost_pct=100.0,
    )
    sim_df = result.df
    assert (sim_df["simulated_mitigation_score"] <= 1.0).all()
    assert (sim_df["simulated_mitigation_score"] >= 0.0).all()
    assert (sim_df["simulated_gap_score"] <= 1.0).all()
    assert (sim_df["simulated_gap_score"] >= -1.0).all()


def test_corridor_mapping_provinces(merged_dataset):
    """Maps all 38 provinces into 7 corridors with zero unknown/null."""
    all_provinces = set(merged_dataset["PROVINSI"].unique())
    assert len(all_provinces) == 38

    # All 38 provinces must be explicitly mapped in CORRIDOR_MAPPING
    for prov in all_provinces:
        assert prov in CORRIDOR_MAPPING, f"Province '{prov}' missing from CORRIDOR_MAPPING"
        assert CORRIDOR_MAPPING[prov] in CORRIDORS, f"Invalid corridor for province '{prov}': {CORRIDOR_MAPPING[prov]}"

    # Specifically check Yogyakarta mapping to 'Jawa-Bali'
    assert CORRIDOR_MAPPING["Daerah Istimewa Yogyakarta"] == "Jawa-Bali"

    # Assign corridors to GeoDataFrame
    assigned = assign_corridors(merged_dataset)
    assert len(assigned) == 514
    assert "corridor" in assigned.columns
    assert assigned["corridor"].isna().sum() == 0

    # Ensure 100% of 514 regencies map to a valid corridor
    unique_corridors = set(assigned["corridor"].unique())
    assert unique_corridors == set(CORRIDORS)


def test_corridor_unit_counts(merged_dataset):
    """Verify regency unit counts per corridor sum to 514."""
    assigned = assign_corridors(merged_dataset)
    counts = assigned.groupby("corridor").size().to_dict()

    expected_counts = {
        "Sumatera": 154,
        "Jawa-Bali": 128,
        "Kalimantan": 56,
        "Sulawesi": 81,
        "Nusa Tenggara": 32,
        "Maluku": 21,
        "Papua": 42,
    }
    assert counts == expected_counts
    assert sum(counts.values()) == 514


def test_get_corridor_weights_zero_islands(merged_dataset):
    """For every corridor, generates valid KNN spatial weights with degree > 0 for all units (0 isolated islands)."""
    assigned = assign_corridors(merged_dataset)

    for corridor_name in CORRIDORS:
        corridor_gdf = get_corridor_subgraph(assigned, corridor_name)
        assert len(corridor_gdf) > 0

        w = get_corridor_weights(corridor_gdf)
        assert w.n == len(corridor_gdf), f"Weights nodes {w.n} != gdf len {len(corridor_gdf)} for {corridor_name}"
        assert len(w.islands) == 0, f"Corridor {corridor_name} has {len(w.islands)} islands!"

        for node, neighbors in w.neighbors.items():
            assert len(neighbors) > 0, f"Node {node} in {corridor_name} has degree 0"


def test_compute_corridor_moran(merged_dataset):
    """Computes localized Moran's I and p-value on corridor subsets."""
    assigned = assign_corridors(merged_dataset)

    for corridor_name in ["Sumatera", "Jawa-Bali", "Sulawesi", "Maluku"]:
        corridor_gdf = get_corridor_subgraph(assigned, corridor_name)
        res = compute_corridor_moran(corridor_gdf, variable="IRBI_2024")

        assert isinstance(res, MoranResult)
        assert res.variable == "IRBI_2024"
        assert -1.0 <= res.moran_i <= 1.0
        assert 0.0 <= res.p_value <= 1.0
        assert isinstance(res.z_score, float)
