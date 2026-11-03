"""Unit and integration tests for Policy Sandbox and Dual-Layer Viewport Components (Milestone 2 - Task 2.3)."""
import re
from html.parser import HTMLParser
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import pytest

from src.analytics.bivariate import compute_bivariate_scores
from src.analytics.simulation import SimulationResult, simulate_policy_intervention
from src.api.routes import get_data_manager
from src.app.components.choropleth import (
    create_choropleth_deck,
    hex_to_rgba,
    prepare_choropleth_data,
)
from src.app.components.sandbox_panel import (
    render_sandbox_controls,
    render_simulation_delta_html,
)
from src.app.components.stats_panel import create_moran_scatterplot


class HTMLTagValidator(HTMLParser):
    """Simple parser to validate balanced tags in HTML snippets."""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []
        self.void_elements = {"br", "hr", "img", "input", "meta", "link"}

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag not in self.void_elements:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.void_elements:
            return
        if not self.stack:
            self.errors.append(f"Unexpected closing tag </{tag}>")
        elif self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(
                f"Mismatched closing tag </{tag}>, expected </{self.stack[-1]}>"
            )

    def is_valid(self) -> bool:
        if self.stack:
            self.errors.append(f"Unclosed tags remaining: {self.stack}")
        return len(self.errors) == 0


@pytest.fixture(scope="module")
def loaded_data():
    """Fixture providing initialized DataManager with 514 units."""
    dm = get_data_manager()
    dm.load()
    return dm


def test_render_simulation_delta_html_format(loaded_data):
    """Verify simulation delta KPI strip HTML formatting and anti-slop rules."""
    sim_result = simulate_policy_intervention(
        loaded_data.bivariate_df,
        signal_reduction_pct=50.0,
        gotong_royong_boost_pct=25.0,
    )
    assert isinstance(sim_result, SimulationResult)

    html = render_simulation_delta_html(sim_result)
    assert isinstance(html, str)
    assert html.startswith("<div")

    # Anti-slop check: zero lines indented with 4 spaces or tabs
    bad_lines = [
        line
        for line in html.splitlines()
        if line.startswith("    ") or line.startswith("\t")
    ]
    assert len(bad_lines) == 0, f"Found {len(bad_lines)} indented lines in KPI HTML"

    # Must contain critical metric labels and counts
    assert "Baseline Critical Gaps" in html
    assert "Simulated Critical Gaps" in html
    assert "Net Resolved" in html
    assert str(sim_result.baseline_critical_gaps) in html
    assert str(sim_result.simulated_critical_gaps) in html
    assert str(sim_result.resolved_gaps) in html

    # Tag balance validation
    validator = HTMLTagValidator()
    validator.feed(html)
    assert validator.is_valid(), f"HTML tag errors: {validator.errors}"


def test_prepare_choropleth_data_with_simulation(loaded_data):
    """Verify prepare_choropleth_data augments dataset with simulation columns."""
    sim_result = simulate_policy_intervention(
        loaded_data.bivariate_df,
        signal_reduction_pct=40.0,
        gotong_royong_boost_pct=30.0,
    )

    prepared = prepare_choropleth_data(
        gdf=loaded_data.gdf,
        bivariate_df=loaded_data.bivariate_df,
        layer_mode="Bivariate Risk vs Mitigation (3x3)",
        simulated_df=sim_result.df,
    )

    assert len(prepared) == 514
    assert "formatted_sim_gap" in prepared.columns
    assert "sim_category_label" in prepared.columns
    assert "fill_color" in prepared.columns
    assert "simulated_gap_score" in prepared.columns
    assert "simulated_category" in prepared.columns
    assert pd.notna(prepared["simulated_gap_score"].iloc[0])

    # Verify that in Bivariate mode, fill_color reflects simulated category color
    # Pick a unit where category changed
    sim_df = sim_result.df
    changed_units = sim_df[
        sim_df["bivariate_category"] != sim_df["simulated_bivariate_category"]
    ]
    assert len(changed_units) > 0

    first_changed_kode = changed_units.iloc[0]["KODE_WIL"]
    row = prepared[prepared["KODE_WIL"] == first_changed_kode].iloc[0]
    expected_sim_color = changed_units.iloc[0]["simulated_bivariate_color"]
    expected_rgba = hex_to_rgba(expected_sim_color, alpha=210)
    assert row["fill_color"] == expected_rgba
    assert row["sim_category_label"] == changed_units.iloc[0]["simulated_bivariate_label"]


def test_prepare_choropleth_data_backward_compatible(loaded_data):
    """Without simulated_df, prepare_choropleth_data preserves baseline behavior."""
    prepared = prepare_choropleth_data(
        gdf=loaded_data.gdf,
        bivariate_df=loaded_data.bivariate_df,
        layer_mode="Bivariate Risk vs Mitigation (3x3)",
    )
    assert len(prepared) == 514
    assert "formatted_gap" in prepared.columns
    assert "category_label" in prepared.columns
    assert "simulated_gap_score" in prepared.columns
    assert "simulated_category" in prepared.columns


def test_create_choropleth_deck_simulation_tooltip(loaded_data):
    """Verify PyDeck Deck tooltip contains simulated comparison and has zero unclosed tags."""
    sim_result = simulate_policy_intervention(
        loaded_data.bivariate_df,
        signal_reduction_pct=50.0,
        gotong_royong_boost_pct=25.0,
    )
    prepared = prepare_choropleth_data(
        gdf=loaded_data.gdf,
        bivariate_df=loaded_data.bivariate_df,
        layer_mode="Bivariate Risk vs Mitigation (3x3)",
        simulated_df=sim_result.df,
    )

    deck = create_choropleth_deck(prepared, is_simulated=True)
    assert isinstance(deck, pdk.Deck)
    assert "tooltip" in deck.__dict__ or hasattr(deck, "tooltip")
    tooltip = deck.tooltip
    assert isinstance(tooltip, dict)
    tooltip_html = tooltip["html"]

    # Verify zero leading 4-space or tab indentations in tooltip HTML
    bad_lines = [
        line
        for line in tooltip_html.splitlines()
        if line.startswith("    ") or line.startswith("\t")
    ]
    assert len(bad_lines) == 0, f"Found {len(bad_lines)} indented lines in tooltip HTML"

    # Verify simulation comparison keywords in tooltip
    assert "Simulated Gap" in tooltip_html
    assert "{formatted_sim_gap}" in tooltip_html
    assert "Simulated Category" in tooltip_html
    assert "{sim_category_label}" in tooltip_html

    # Validate HTML tag closure
    validator = HTMLTagValidator()
    validator.feed(tooltip_html)
    assert validator.is_valid(), f"Tooltip HTML tag syntax errors: {validator.errors}"


def test_create_choropleth_deck_baseline_tooltip(loaded_data):
    """Verify baseline PyDeck Deck tooltip does not include simulation tags."""
    prepared = prepare_choropleth_data(
        gdf=loaded_data.gdf,
        bivariate_df=loaded_data.bivariate_df,
        layer_mode="Bivariate Risk vs Mitigation (3x3)",
    )
    deck = create_choropleth_deck(prepared, is_simulated=False)
    tooltip_html = deck.tooltip["html"]

    assert "Simulated Gap" not in tooltip_html
    assert "{formatted_gap}" in tooltip_html

    validator = HTMLTagValidator()
    validator.feed(tooltip_html)
    assert validator.is_valid(), f"Baseline tooltip HTML tag errors: {validator.errors}"


def test_create_moran_scatterplot_corridor_annotation(loaded_data):
    """Verify Moran scatterplot accepts corridor_name and displays corridor annotation."""
    fig = create_moran_scatterplot(
        gdf=loaded_data.gdf,
        variable="IRBI_2024",
        weights=loaded_data.weights,
        corridor_name="Sumatera",
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title is not None
    title_text = fig.layout.title.text
    assert "Sumatera" in title_text


def test_render_sandbox_controls_mocked():
    """Verify sandbox control sliders return parameters and active flag."""
    with patch("streamlit.slider", side_effect=[35.0, 20.0]):
        signal_pct, gotong_pct, is_active = render_sandbox_controls()
        assert signal_pct == 35.0
        assert gotong_pct == 20.0
        assert is_active is True

    with patch("streamlit.slider", side_effect=[0.0, 0.0]):
        signal_pct, gotong_pct, is_active = render_sandbox_controls()
        assert signal_pct == 0.0
        assert gotong_pct == 0.0
        assert is_active is False
