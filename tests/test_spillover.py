"""Tests for Spatial Spillover Econometric Module (Task 3.2)."""
import pandas as pd
import pytest

from src.api.routes import get_data_manager
from src.analytics.weights import build_spatial_weights
from src.analytics.bivariate import compute_bivariate_scores
from src.analytics.spillover import (
    estimate_spatial_spillover,
    SpilloverResult,
    SpilloverEffect,
)


@pytest.fixture(scope="module")
def data_context():
    """Load cached spatial DataManager and ensure bivariate scoring."""
    dm = get_data_manager()
    dm.load()
    gdf = compute_bivariate_scores(dm.gdf)
    return gdf


def test_estimate_spatial_spillover_lag_default(data_context):
    """Verify spatial lag model estimation with default regressors on national scope."""
    gdf = data_context
    x_vars = ["SINYAL_BURUK", "GOTONG_ROYONG", "IPM"]
    y_var = "gap_score"

    result = estimate_spatial_spillover(
        gdf,
        weights=None,
        y_var=y_var,
        x_vars=x_vars,
        model_type="lag",
    )

    # 1. Structure assertions
    assert isinstance(result, SpilloverResult)
    assert result.model_type == "lag"
    assert result.y_var == y_var
    assert result.x_vars == x_vars
    assert result.n_obs == 514

    # 2. Spatial autoregressive parameter rho
    assert isinstance(result.rho, float)
    assert -1.0 < result.rho < 1.0
    assert result.rho_p_value is not None
    assert 0.0 <= result.rho_p_value <= 1.0
    assert result.rho_z_stat is not None

    # 3. Model fit
    assert isinstance(result.r2, float)
    assert 0.0 <= result.r2 <= 1.0

    # 4. Effects list inspection
    assert len(result.effects) == 3
    var_names = [e.variable for e in result.effects]
    assert var_names == x_vars

    for effect in result.effects:
        assert isinstance(effect, SpilloverEffect)
        # Direct + Indirect = Total (abs diff < 1e-4)
        assert abs(effect.total_effect - (effect.direct_effect + effect.indirect_effect)) < 1e-4
        # Spillover ratio bounded
        assert 0.0 <= effect.spillover_ratio <= 1.0
        # p-value valid
        assert 0.0 <= effect.p_value <= 1.0
        # Non-empty interpretation string
        assert isinstance(effect.interpretation, str)
        assert len(effect.interpretation) > 15

    # 5. Summary DataFrame
    df = result.summary_df
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "Variable" in df.columns
    assert "Direct Effect" in df.columns
    assert "Indirect (Spillover)" in df.columns
    assert "Total Effect" in df.columns
    assert "p-value" in df.columns


def test_estimate_spatial_spillover_slx(data_context):
    """Verify Spatial Lag of X (SLX) model estimation."""
    gdf = data_context
    x_vars = ["SINYAL_BURUK", "GOTONG_ROYONG", "IPM"]
    y_var = "gap_score"

    result = estimate_spatial_spillover(
        gdf,
        weights=None,
        y_var=y_var,
        x_vars=x_vars,
        model_type="slx",
    )

    assert isinstance(result, SpilloverResult)
    assert result.model_type == "slx"
    assert result.n_obs == 514
    assert result.rho is None or result.rho == 0.0
    assert len(result.effects) == 3

    for effect in result.effects:
        assert isinstance(effect, SpilloverEffect)
        assert abs(effect.total_effect - (effect.direct_effect + effect.indirect_effect)) < 1e-4
        assert -1.0 <= effect.spillover_ratio <= 1.0 or 0.0 <= effect.spillover_ratio <= 1.0
        assert 0.0 <= effect.p_value <= 1.0
        assert len(effect.interpretation) > 15


def test_estimate_spatial_spillover_corridor_sumatera(data_context):
    """Verify corridor-level estimation bounded to Sumatera sub-graph."""
    gdf = data_context

    result = estimate_spatial_spillover(
        gdf,
        y_var="gap_score",
        x_vars=["SINYAL_BURUK", "GOTONG_ROYONG", "IPM"],
        model_type="lag",
        corridor="Sumatera",
    )

    assert isinstance(result, SpilloverResult)
    assert result.corridor == "Sumatera"
    assert result.n_obs == 154
    assert isinstance(result.rho, float)
    assert -1.0 < result.rho < 1.0
    assert len(result.effects) == 3

    for effect in result.effects:
        assert abs(effect.total_effect - (effect.direct_effect + effect.indirect_effect)) < 1e-4


def test_estimate_spatial_spillover_precomputed_weights(data_context):
    """Verify estimate_spatial_spillover accepts precomputed spatial weights."""
    gdf = data_context
    w = build_spatial_weights(gdf, k=5)

    result = estimate_spatial_spillover(
        gdf,
        weights=w,
        y_var="gap_score",
        x_vars=["SINYAL_BURUK", "GOTONG_ROYONG", "IPM"],
        model_type="lag",
    )

    assert result.n_obs == 514
    assert -1.0 < result.rho < 1.0


def test_estimate_spatial_spillover_invalid_inputs(data_context):
    """Verify appropriate exceptions for invalid arguments."""
    gdf = data_context

    with pytest.raises(ValueError, match="Invalid corridor"):
        estimate_spatial_spillover(gdf, corridor="Atlantis")

    with pytest.raises(ValueError, match="Invalid model_type"):
        estimate_spatial_spillover(gdf, model_type="random_forest")

    with pytest.raises(KeyError, match="not found in data"):
        estimate_spatial_spillover(gdf, x_vars=["NON_EXISTENT_VAR"])
