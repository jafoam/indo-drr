"""Unit and integration tests for Streamlit viewport components."""
import pytest
import pydeck as pdk
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd

from src.api.routes import get_data_manager
from src.analytics.moran import compute_morans_i, compute_lisa
from src.app.components.choropleth import (
    hex_to_rgba,
    prepare_choropleth_data,
    create_choropleth_deck,
    render_bivariate_legend_html,
    render_lisa_legend_html,
)
from src.app.components.stats_panel import (
    create_moran_scatterplot,
    render_moran_metrics_html,
    get_top_critical_gap_table,
)


@pytest.fixture(scope="module")
def loaded_data():
    """Fixture providing initialized DataManager with 514 units."""
    dm = get_data_manager()
    dm.load()
    return dm


def test_hex_to_rgba():
    """Verify hex to RGBA conversion."""
    rgba = hex_to_rgba("#d03838", alpha=200)
    assert rgba == [208, 56, 56, 200]

    # Test fallback
    fallback = hex_to_rgba("invalid", alpha=180)
    assert len(fallback) == 4
    assert fallback[3] == 180


def test_prepare_choropleth_data_bivariate(loaded_data):
    """Verify choropleth data preparation for Bivariate mode."""
    prepared = prepare_choropleth_data(
        gdf=loaded_data.gdf,
        bivariate_df=loaded_data.bivariate_df,
        layer_mode="Bivariate Risk vs Mitigation (3x3)",
    )
    assert len(prepared) == 514
    assert "fill_color" in prepared.columns
    assert "formatted_risk" in prepared.columns
    assert "formatted_mitigation" in prepared.columns
    assert "formatted_gap" in prepared.columns
    assert "category_label" in prepared.columns
    assert len(prepared.iloc[0]["fill_color"]) == 4


def test_prepare_choropleth_data_lisa(loaded_data):
    """Verify choropleth data preparation for LISA mode."""
    prepared = prepare_choropleth_data(
        gdf=loaded_data.gdf,
        bivariate_df=loaded_data.bivariate_df,
        layer_mode="LISA Spatial Clusters",
        lisa_variable="IRBI_2024",
        p_threshold=0.05,
    )
    assert len(prepared) == 514
    assert "fill_color" in prepared.columns
    assert "category_label" in prepared.columns
    # Check valid cluster labels
    valid_labels = {"High-High", "Low-Low", "High-Low", "Low-High", "Not Significant"}
    assert all(cat in valid_labels for cat in prepared["category_label"].unique())


def test_create_choropleth_deck(loaded_data):
    """Verify PyDeck Deck creation, initial view state, CRS enforcement, and basemap style."""
    prepared = prepare_choropleth_data(
        gdf=loaded_data.gdf,
        bivariate_df=loaded_data.bivariate_df,
        layer_mode="Bivariate Risk vs Mitigation (3x3)",
    )
    assert prepared.crs.to_epsg() == 4326

    deck = create_choropleth_deck(prepared)
    assert isinstance(deck, pdk.Deck)
    assert len(deck.layers) == 1
    assert deck.layers[0].type == "GeoJsonLayer"
    assert deck.initial_view_state.latitude == -2.5
    assert deck.initial_view_state.longitude == 118.0
    assert deck.initial_view_state.zoom == 4.2
    assert deck.map_provider == "carto"
    assert "cartocdn.com" in deck.map_style or deck.map_style == "carto-darkmatter"

    # Test non-4326 CRS is reprojected to EPSG:4326
    projected_prep = prepare_choropleth_data(
        gdf=loaded_data.gdf.to_crs(3857),
        bivariate_df=loaded_data.bivariate_df,
    )
    assert projected_prep.crs.to_epsg() == 4326

    reproj_deck = create_choropleth_deck(loaded_data.gdf.to_crs(3857))
    sample_pt = reproj_deck.layers[0].data[0]["geometry"]["coordinates"][0][0][0]
    # Check coordinates are in longitude/latitude degrees (~95 to 142), not EPSG:3857 meters (>1,000,000)
    assert 90.0 < sample_pt[0] < 145.0


def test_legends_markup():
    """Verify HTML legend markup generation and ensure no raw code leak from indentation."""
    biv_html = render_bivariate_legend_html()
    assert "3×3 Risk vs. Mitigation" in biv_html
    assert "#d03838" in biv_html
    assert "#3b82a6" in biv_html
    assert biv_html.startswith("<div")
    biv_bad_lines = [l for l in biv_html.splitlines() if l.startswith("    ") or l.startswith("\t")]
    assert len(biv_bad_lines) == 0, f"Bivariate legend has {len(biv_bad_lines)} indented lines that leak as raw code block"

    lisa_html = render_lisa_legend_html()
    assert "LISA Spatial Clusters" in lisa_html
    assert "High-High (Hotspot)" in lisa_html
    assert "Low-Low (Coldspot)" in lisa_html
    assert lisa_html.startswith("<div")
    lisa_bad_lines = [l for l in lisa_html.splitlines() if l.startswith("    ") or l.startswith("\t")]
    assert len(lisa_bad_lines) == 0, f"LISA legend has {len(lisa_bad_lines)} indented lines that leak as raw code block"


def test_moran_metrics_markup(loaded_data):
    """Verify Moran metrics HTML markup has no code-block indentation leaks."""
    from src.analytics.moran import compute_morans_i
    res = compute_morans_i(loaded_data.gdf, column="IRBI_2024", weights=loaded_data.weights)
    metrics_html = render_moran_metrics_html(res)
    assert metrics_html.startswith("<div")
    assert "Global Autocorrelation" in metrics_html
    bad_lines = [l for l in metrics_html.splitlines() if l.startswith("    ") or l.startswith("\t")]
    assert len(bad_lines) == 0, f"Moran metrics has {len(bad_lines)} indented lines that leak as raw code block"


def test_create_moran_scatterplot(loaded_data):
    """Verify Moran scatterplot figure generation."""
    lisa_res = compute_lisa(loaded_data.gdf, column="IRBI_2024", weights=loaded_data.weights)
    fig = create_moran_scatterplot(
        gdf=loaded_data.gdf,
        variable="IRBI_2024",
        weights=loaded_data.weights,
        lisa_clusters=lisa_res["lisa_cluster"],
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # scatter points trace + OLS line trace
    assert fig.data[0].mode == "markers"
    assert fig.data[1].mode == "lines"


def test_render_moran_metrics_html(loaded_data):
    """Verify Moran metric block HTML generation."""
    moran_res = compute_morans_i(loaded_data.gdf, column="IRBI_2024", weights=loaded_data.weights)
    html = render_moran_metrics_html(moran_res)
    assert "Global Autocorrelation" in html
    assert f"{moran_res.moran_i:+.4f}" in html
    assert "Monte Carlo permutations" in html


def test_get_top_critical_gap_table(loaded_data):
    """Verify top critical gap table ranking."""
    top_10 = get_top_critical_gap_table(loaded_data.bivariate_df, n=10)
    assert isinstance(top_10, pd.DataFrame)
    assert len(top_10) == 10
    assert "Regency / City" in top_10.columns
    assert "Province" in top_10.columns
    assert "Gap (R - M)" in top_10.columns


def test_headless_import_app():
    """Verify headless dry-run import of Streamlit app module."""
    import src.app.app as app_module
    assert hasattr(app_module, "main")
    assert hasattr(app_module, "apply_antislop_styling")


def test_antislop_css_padding_and_line_height():
    """Verify CSS fixes top clipping with adequate padding and line-height."""
    from src.app.app import get_antislop_css
    css = get_antislop_css()
    assert "padding-top: 2" in css
    assert "!important" in css
    assert "line-height: 1.4" in css
    assert ".idr-title" in css


def test_spillover_panel_components(loaded_data):
    """Verify spillover metric card, waterfall chart, and effects table components."""
    from src.analytics.spillover import estimate_spatial_spillover
    from src.app.components.spillover_panel import (
        render_spillover_metric_card_html,
        render_spillover_waterfall_chart,
        render_spillover_effects_table,
    )

    res = estimate_spatial_spillover(loaded_data.gdf, weights=loaded_data.weights, model_type="lag")

    # Card HTML markup assertions
    card_html = render_spillover_metric_card_html(res)
    assert card_html.startswith("<div")
    assert "MODEL TYPE" in card_html
    assert "SPATIAL RHO" in card_html
    assert "SPATIAL MULTIPLIER" in card_html
    assert "SPILLOVER RATIO" in card_html
    bad_lines = [l for l in card_html.splitlines() if l.startswith("    ") or l.startswith("\t")]
    assert len(bad_lines) == 0, f"Card markup has {len(bad_lines)} indented lines that leak as raw code block"

    # Waterfall figure assertions
    fig = render_spillover_waterfall_chart(res)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].orientation == "h"
    assert fig.data[1].orientation == "h"
    assert fig.data[0].name == "Direct Effect (Local)"
    assert fig.data[1].name == "Indirect (Spillover)"

    # Table assertions
    df = render_spillover_effects_table(res)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "Variable" in df.columns
    assert "Direct Effect" in df.columns
    assert "Indirect (Spillover)" in df.columns
    assert "Total Effect" in df.columns
    assert "p-value" in df.columns


def test_inspector_components(loaded_data):
    """Verify district inspector card and radar chart components."""
    from src.analytics.profile import compute_regency_profile
    from src.app.components.inspector import (
        render_inspector_card_html,
        render_radar_chart,
    )

    profile = compute_regency_profile(loaded_data.gdf, loaded_data.bivariate_df, kode_wil=1101)

    card_html = render_inspector_card_html(profile)
    assert card_html.startswith("<div")
    assert "Simeulue" in card_html
    assert "Aceh" in card_html
    bad_lines = [l for l in card_html.splitlines() if l.startswith("    ") or l.startswith("\t")]
    assert len(bad_lines) == 0, f"Inspector card has {len(bad_lines)} indented lines that leak as raw code block"

    radar = render_radar_chart(profile)
    assert isinstance(radar, go.Figure)
    assert len(radar.data) == 3
    assert radar.data[0].name == "Simeulue"

