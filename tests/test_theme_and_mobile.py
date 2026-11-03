"""Unit and integration tests for Theme Switcher, Viewport Footer, and Mobile Responsiveness."""
import pytest
import geopandas as gpd
import pandas as pd
import pydeck as pdk

from src.api.routes import get_data_manager


@pytest.fixture(scope="module")
def loaded_data():
    """Fixture providing initialized DataManager with 514 units."""
    dm = get_data_manager()
    dm.load()
    return dm


def test_theme_tokens_definition():
    """Verify dark and light theme tokens and get_theme_tokens helper."""
    from src.config import (
        COLOR_TOKENS,
        DARK_COLOR_TOKENS,
        LIGHT_COLOR_TOKENS,
        get_theme_tokens,
    )

    # Check DARK tokens
    assert "bg-primary" in DARK_COLOR_TOKENS
    assert DARK_COLOR_TOKENS["bg-primary"] in ["#0f141c", "#1a2332"]
    assert DARK_COLOR_TOKENS["map-style"] == "carto-darkmatter"
    assert DARK_COLOR_TOKENS["plotly-template"] == "plotly_dark"

    # Check LIGHT tokens
    assert "bg-primary" in LIGHT_COLOR_TOKENS
    assert LIGHT_COLOR_TOKENS["bg-primary"] == "#f8fafc"
    assert LIGHT_COLOR_TOKENS["text-primary"] == "#0f172a"
    assert LIGHT_COLOR_TOKENS["map-style"] == "carto-positron"
    assert LIGHT_COLOR_TOKENS["plotly-template"] == "plotly_white"

    # Check get_theme_tokens selector
    dark_res = get_theme_tokens("Dark Theme")
    assert dark_res["map-style"] == "carto-darkmatter"

    light_res = get_theme_tokens("Light Theme")
    assert light_res["map-style"] == "carto-positron"
    assert light_res["text-primary"] == "#0f172a"


def test_responsive_css_media_queries():
    """Verify responsive CSS media queries for mobile layout and touch heights."""
    from src.app.app import get_antislop_css

    css = get_antislop_css()
    assert "@media (max-width: 768px)" in css
    # Single column stack collapse
    assert "flex-direction: column !important" in css
    # Touch-friendly pydeck map container height constraint
    assert "height: 380px" in css or "max-height: 420px" in css
    assert "touch-action" in css


def test_viewport_footer_two_subcolumns(loaded_data):
    """Verify 2-subcolumn viewport footer markup, contingency counts, KPI pills, and hazard bars."""
    from src.app.components.choropleth import render_viewport_footer_html

    footer_html = render_viewport_footer_html(
        bivariate_df=loaded_data.bivariate_df,
        gdf=loaded_data.gdf,
    )
    assert isinstance(footer_html, str)
    assert footer_html.startswith("<div")

    # Anti-slop check: zero 4-space / tab leading indentations
    bad_lines = [l for l in footer_html.splitlines() if l.startswith("    ") or l.startswith("\t")]
    assert len(bad_lines) == 0, f"Viewport footer has {len(bad_lines)} indented lines leaking as raw code"

    # Sub-column Left elements
    assert "3×3 Risk vs. Mitigation" in footer_html
    assert "Critical Gap (R3-M1)" in footer_html
    assert "Resilient (R1-M3)" in footer_html

    # Sub-column Right elements: KPI metric pills
    assert "Gap Score" in footer_html
    assert "Bad Telecom" in footer_html
    assert "Social Capital" in footer_html

    # Multi-hazard progress bars
    assert "Floods / Water Hazard" in footer_html or "Floods" in footer_html
    assert "Tsunami EWS" in footer_html
    assert "Landslides / Slopes" in footer_html or "Landslides" in footer_html


def test_viewport_footer_corridor_filtering(loaded_data):
    """Verify contingency matrix frequency counts sum to active corridor sample size."""
    from src.app.components.choropleth import (
        render_viewport_footer_html,
        get_contingency_matrix_counts,
    )

    # National (N=514) - Hybrid Threshold Matrix
    nat_counts = get_contingency_matrix_counts(loaded_data.bivariate_df)
    assert sum(nat_counts.values()) == 514
    assert nat_counts["R3-M1"] == 148

    # Sumatera corridor (N=154)
    sumatera_df = loaded_data.bivariate_df[loaded_data.bivariate_df["corridor"] == "Sumatera"]
    sum_counts = get_contingency_matrix_counts(sumatera_df)
    assert sum(sum_counts.values()) == 154
    assert sum_counts["R3-M1"] > 0

    sum_html = render_viewport_footer_html(
        bivariate_df=sumatera_df,
        gdf=loaded_data.gdf[loaded_data.gdf["corridor"] == "Sumatera"],
    )
    assert "154" in sum_html


def test_theme_switching_in_components(loaded_data):
    """Verify theme token swapping in PyDeck and Plotly components."""
    from src.config import DARK_COLOR_TOKENS, LIGHT_COLOR_TOKENS
    from src.app.components.choropleth import (
        prepare_choropleth_data,
        create_choropleth_deck,
    )
    from src.app.components.stats_panel import create_moran_scatterplot

    prepared = prepare_choropleth_data(
        gdf=loaded_data.gdf,
        bivariate_df=loaded_data.bivariate_df,
        layer_mode="Bivariate Risk vs Mitigation (3x3)",
    )

    # PyDeck with dark theme
    dark_deck = create_choropleth_deck(prepared, theme_tokens=DARK_COLOR_TOKENS)
    assert dark_deck.map_style in (
        "carto-darkmatter",
        "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )
    assert dark_deck.map_provider == "carto"

    # PyDeck with light theme
    light_deck = create_choropleth_deck(prepared, theme_tokens=LIGHT_COLOR_TOKENS)
    assert light_deck.map_style in (
        "carto-positron",
        "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )
    assert light_deck.map_provider == "carto"

    # Plotly Moran scatterplot with light theme
    fig_light = create_moran_scatterplot(
        gdf=loaded_data.gdf,
        variable="IRBI_2024",
        weights=loaded_data.weights,
        theme_tokens=LIGHT_COLOR_TOKENS,
    )
    assert fig_light.layout.paper_bgcolor == LIGHT_COLOR_TOKENS["bg-surface"]


def test_light_theme_high_contrast_css():
    """Verify scoped high-contrast CSS overrides for Light Theme."""
    from src.app.app import get_antislop_css
    from src.config import LIGHT_COLOR_TOKENS

    css_light = get_antislop_css(LIGHT_COLOR_TOKENS, theme="Light Theme")

    # High-contrast sidebar styling
    assert 'section[data-testid="stSidebar"]' in css_light
    assert "#f1f5f9 !important" in css_light
    assert "#cbd5e1 !important" in css_light
    assert "#0f172a !important" in css_light

    # Right analytics panel toggle label high-contrast text
    assert 'div[data-testid="stRadio"] label p' in css_light


def test_inspector_adaptive_drawer_and_radar(loaded_data):
    """Verify district drawer and polar radar adapt cleanly to Dark and Light Themes."""
    from src.analytics.profile import compute_regency_profile
    from src.app.components.inspector import (
        render_inspector_card_html,
        render_radar_chart,
    )

    profile = compute_regency_profile(loaded_data.gdf, loaded_data.bivariate_df, kode_wil=1101)

    # 1. Dark Theme Inspector Card
    card_dark = render_inspector_card_html(profile, is_bright=False)
    assert "#1a2332" in card_dark
    assert "#f1f5f9" in card_dark
    assert "#2a384c" in card_dark
    # Zero indented lines
    bad_dark = [l for l in card_dark.splitlines() if l.startswith("    ") or l.startswith("\t")]
    assert len(bad_dark) == 0

    # 2. Light Theme Inspector Card
    card_light = render_inspector_card_html(profile, is_bright=True)
    assert "#ffffff" in card_light
    assert "#0f172a" in card_light
    assert "#cbd5e1" in card_light
    assert "#1a2332" not in card_light
    assert "#0b0f19" not in card_light
    assert "color: #ffffff" not in card_light
    bad_light = [l for l in card_light.splitlines() if l.startswith("    ") or l.startswith("\t")]
    assert len(bad_light) == 0

    # 3. Polar Radar - Dark Theme
    radar_dark = render_radar_chart(profile, is_bright=False)
    assert radar_dark.layout.paper_bgcolor in ["#161e2a", "#1a2332"]
    assert radar_dark.layout.polar.bgcolor in ["#0f141c", "#1a2332"]

    # 4. Polar Radar - Light Theme
    radar_light = render_radar_chart(profile, is_bright=True)
    assert radar_light.layout.paper_bgcolor == "#ffffff"
    assert radar_light.layout.plot_bgcolor == "#ffffff"
    assert radar_light.layout.polar.angularaxis.tickfont.color == "#0f172a"
    assert radar_light.layout.polar.angularaxis.linecolor == "#0f172a"
    assert radar_light.layout.polar.radialaxis.gridcolor == "#cbd5e1"
    assert radar_light.layout.font.color == "#0f172a"


def test_moran_scatterplot_light_theme_contrast(loaded_data):
    """Verify Plotly axes tickfont and title font colors are #0f172a with #e2e8f0 gridlines in Light Theme."""
    from src.app.components.stats_panel import create_moran_scatterplot

    fig = create_moran_scatterplot(
        gdf=loaded_data.gdf,
        variable="IRBI_2024",
        weights=loaded_data.weights,
        is_bright=True,
    )

    assert fig.layout.paper_bgcolor == "#ffffff"
    assert fig.layout.plot_bgcolor == "#ffffff"
    assert fig.layout.xaxis.tickfont.color == "#0f172a"
    assert fig.layout.xaxis.title.font.color == "#0f172a"
    assert fig.layout.xaxis.gridcolor == "#e2e8f0"
    assert fig.layout.xaxis.linecolor == "#0f172a"
    assert fig.layout.yaxis.tickfont.color == "#0f172a"
    assert fig.layout.yaxis.title.font.color == "#0f172a"
    assert fig.layout.yaxis.gridcolor == "#e2e8f0"
    assert fig.layout.yaxis.linecolor == "#0f172a"


def test_top_10_table_and_global_kpi_theme_adaptation(loaded_data):
    """Verify Top 10 Critical Gap table and Global KPI card adapt dynamically between Light and Dark themes."""
    from src.analytics.moran import compute_morans_i
    from src.app.components.stats_panel import (
        render_moran_metrics_html,
        render_top_critical_gap_table_html,
    )
    from src.app.app import get_antislop_css
    from src.config import LIGHT_COLOR_TOKENS, DARK_COLOR_TOKENS

    moran_res = compute_morans_i(loaded_data.gdf, column="IRBI_2024", weights=loaded_data.weights)

    # 1. Global KPI card in Light Theme
    kpi_light = render_moran_metrics_html(moran_res, is_bright=True)
    assert "#ffffff" in kpi_light
    assert "#cbd5e1" in kpi_light
    assert "#0f172a" in kpi_light
    assert "#f1f5f9" in kpi_light

    # 2. Global KPI card in Dark Theme
    kpi_dark = render_moran_metrics_html(moran_res, is_bright=False)
    assert "#161e2a" in kpi_dark or "#1a2332" in kpi_dark
    assert "#2a384c" in kpi_dark
    assert "#f1f5f9" in kpi_dark

    # 3. Top 10 Table HTML in Light Theme
    table_light = render_top_critical_gap_table_html(loaded_data.bivariate_df, n=10, is_bright=True)
    assert "#ffffff" in table_light
    assert "#f1f5f9" in table_light
    assert "#0f172a" in table_light
    assert "#e2e8f0" in table_light
    assert "#1a2332" not in table_light
    assert "#0b0f19" not in table_light
    assert "color: #ffffff" not in table_light

    # 4. Top 10 Table HTML in Dark Theme
    table_dark = render_top_critical_gap_table_html(loaded_data.bivariate_df, n=10, is_bright=False)
    assert "#1a2332" in table_dark
    assert "#0f172a" in table_dark
    assert "#f1f5f9" in table_dark
    assert "#2a384c" in table_dark

    # 5. CSS overrides for Top 10 Table in Streamlit dataframe / table components
    css_light = get_antislop_css(LIGHT_COLOR_TOKENS, theme="Light Theme")
    assert 'div[data-testid="stDataFrame"], div[data-testid="stTable"]' in css_light
    assert "#ffffff !important" in css_light
    assert "#f1f5f9 !important" in css_light
    assert "#0f172a !important" in css_light
    assert "#e2e8f0 !important" in css_light

    css_dark = get_antislop_css(DARK_COLOR_TOKENS, theme="Dark Theme")
    assert 'div[data-testid="stDataFrame"], div[data-testid="stTable"]' in css_dark
    assert "#1a2332 !important" in css_dark
    assert "#0f172a !important" in css_dark
    assert "#f1f5f9 !important" in css_dark
    assert "#2a384c !important" in css_dark


def test_provincial_hotspots_strip(loaded_data):
    """Verify Top Provincial Priority Hotspots analytical strip in dark and light themes."""
    from src.app.components.choropleth import render_provincial_hotspots_strip_html

    # Light Theme
    strip_light = render_provincial_hotspots_strip_html(
        bivariate_df=loaded_data.bivariate_df,
        gdf=loaded_data.gdf,
        is_bright=True,
    )
    assert isinstance(strip_light, str)
    assert strip_light.startswith("<div")
    assert "Top Provincial Priority Hotspots" in strip_light
    assert "#ffffff" in strip_light
    assert "#b91c1c" in strip_light
    assert "#0f172a" in strip_light
    assert "#1a2332" not in strip_light
    assert "#0b0f19" not in strip_light
    assert "color: #ffffff" not in strip_light
    bad_lines = [l for l in strip_light.splitlines() if l.startswith("    ") or l.startswith("\t")]
    assert len(bad_lines) == 0

    # Dark Theme
    strip_dark = render_provincial_hotspots_strip_html(
        bivariate_df=loaded_data.bivariate_df,
        gdf=loaded_data.gdf,
        is_bright=False,
    )
    assert isinstance(strip_dark, str)
    assert "#161e2a" in strip_dark
    assert "#f87171" in strip_dark


def test_simulation_delta_html_theme_adaptation():
    """Verify Policy Intervention sandbox delta HTML eliminates dark styles in Light Theme."""
    from src.analytics.simulation import SimulationResult
    from src.app.components.sandbox_panel import render_simulation_delta_html
    import pandas as pd

    sim_res = SimulationResult(
        baseline_critical_gaps=54,
        simulated_critical_gaps=16,
        resolved_gaps=38,
        df=pd.DataFrame(),
    )

    delta_light = render_simulation_delta_html(sim_res, is_bright=True)
    assert "#ffffff" in delta_light
    assert "#0f172a" in delta_light
    assert "#dc2626" in delta_light
    assert "#161e2a" not in delta_light
    assert "#0f141c" not in delta_light
    assert "#f1f5f9" not in delta_light

    delta_dark = render_simulation_delta_html(sim_res, is_bright=False)
    assert "#161e2a" in delta_dark
    assert "#f1f5f9" in delta_dark


def test_i18n_localization_and_components(loaded_data):
    """Verify bilingual (EN/ID) translations and localized UI components."""
    from src.app.i18n import t
    from src.app.components.stats_panel import (
        render_moran_metrics_html,
        get_top_critical_gap_table,
        render_top_critical_gap_table_html,
    )
    from src.app.components.spillover_panel import (
        render_spillover_metric_card_html,
        render_spillover_effects_table,
    )
    from src.app.components.inspector import render_inspector_card_html
    from src.app.components.choropleth import render_viewport_footer_html, render_provincial_hotspots_strip_html
    from src.analytics.profile import compute_regency_profile
    from src.analytics.moran import compute_morans_i
    from src.analytics.spillover import estimate_spatial_spillover

    # 1. Dictionary translation checks
    assert "Disaster Readiness" in t("app_title", "en")
    assert "Kesiapsiagaan" in t("app_title", "id")
    assert "Active Viewport" in t("app_subtitle_prefix", "en")
    assert "Tampilan Aktif" in t("app_subtitle_prefix", "id")

    # 2. Localized Moran metrics HTML
    moran_res = compute_morans_i(loaded_data.gdf, column="IRBI_2024", weights=loaded_data.weights)
    moran_id = render_moran_metrics_html(moran_res, lang="id")
    assert "Autokorelasi Global" in moran_id
    assert "Permutasi Monte Carlo" in moran_id

    # 3. Localized Top Critical Gap Table
    top_10_id = get_top_critical_gap_table(loaded_data.bivariate_df, n=5, lang="id")
    assert "Kabupaten / Kota" in top_10_id.columns
    assert "Provinsi" in top_10_id.columns
    assert "Kesenjangan (R - M)" in top_10_id.columns

    table_html_id = render_top_critical_gap_table_html(loaded_data.bivariate_df, n=5, lang="id")
    assert "Kabupaten / Kota" in table_html_id
    assert "Kesenjangan (R - M)" in table_html_id

    # 4. Localized Spillover Metric Card & Table
    spill_res = estimate_spatial_spillover(loaded_data.gdf, weights=loaded_data.weights, model_type="lag")
    card_id = render_spillover_metric_card_html(spill_res, lang="id")
    assert "Dekomposisi Ekonometrika Spasial" in card_id
    assert "TIPE MODEL" in card_id

    spill_tbl_id = render_spillover_effects_table(spill_res, lang="id")
    assert "Dampak Langsung" in spill_tbl_id.columns
    assert "Dampak Total" in spill_tbl_id.columns

    # 5. Localized Inspector Card
    profile = compute_regency_profile(loaded_data.gdf, loaded_data.bivariate_df, 1101)
    card_insp_id = render_inspector_card_html(profile, lang="id")
    assert "PROVINSI:" in card_insp_id
    assert "KORIDOR:" in card_insp_id
    assert "PERINGKAT GAP:" in card_insp_id

    # 6. Localized Viewport Footer & Hotspots Strip
    footer_id = render_viewport_footer_html(bivariate_df=loaded_data.bivariate_df, gdf=loaded_data.gdf, lang="id")
    assert "3×3 Risiko vs. Mitigasi" in footer_id
    assert "Kesenjangan Kritis (R3-M1)" in footer_id
    assert "Ketahanan Teritorial & Pemicu Bahaya" in footer_id

    strip_id = render_provincial_hotspots_strip_html(bivariate_df=loaded_data.bivariate_df, gdf=loaded_data.gdf, lang="id")
    assert "Titik Prioritas Provinsi Tertinggi" in strip_id


def test_readiness_balance_choropleth_and_legend(loaded_data):
    """Verify Readiness Balance choropleth layer data preparation and legend."""
    from src.app.components.choropleth import (
        prepare_choropleth_data,
        render_balance_legend_html,
        get_balance_rgba,
    )
    from src.config import BALANCE_COLOR_RAMP

    # 1. Continuous RGBA evaluation
    rgba_severe = get_balance_rgba(-0.50)
    assert rgba_severe == [185, 28, 28, 200]
    rgba_surplus = get_balance_rgba(0.40)
    assert rgba_surplus[0] < 150 and rgba_surplus[2] > 180

    # 2. Data preparation
    prep_gdf = prepare_choropleth_data(
        loaded_data.gdf,
        loaded_data.bivariate_df,
        layer_mode="Readiness Balance (Deficit-Surplus)",
    )
    assert "readiness_balance" in prep_gdf.columns
    assert "adequacy_ratio" in prep_gdf.columns
    assert "fill_color" in prep_gdf.columns
    assert len(prep_gdf["fill_color"].iloc[0]) == 4

    # 3. Legend HTML in Dark and Light themes
    legend_dark = render_balance_legend_html(is_bright=False, lang="id")
    assert "Defisit Ekstrem" in legend_dark
    assert "Surplus Mitigasi" in legend_dark
    assert "#161e2a" in legend_dark

    legend_light = render_balance_legend_html(is_bright=True, lang="en")
    assert "Severe Deficit" in legend_light
    assert "Mitigation Surplus" in legend_light
    assert "#ffffff" in legend_light


def test_parity_quadrant_chart(loaded_data):
    """Verify 4-Quadrant Parity Diagram rendering and theme compliance."""
    from src.app.components.stats_panel import create_parity_quadrant_chart

    # Dark Theme with target_kode
    fig_dark = create_parity_quadrant_chart(
        loaded_data.bivariate_df,
        target_kode=1101,
        is_bright=False,
        lang="id",
    )
    assert fig_dark is not None
    # Verify trace names
    trace_names = [t.name for t in fig_dark.data if t.name]
    assert any("Garis Paritas" in n for n in trace_names)
    assert any("Target" in n for n in trace_names)

    # Light Theme
    fig_light = create_parity_quadrant_chart(
        loaded_data.bivariate_df,
        target_kode=None,
        is_bright=True,
        lang="en",
    )
    assert fig_light.layout.paper_bgcolor == "#ffffff"
    trace_names_en = [t.name for t in fig_light.data if t.name]
    assert any("Parity Line" in n for n in trace_names_en)


def test_inspector_adequacy_kpi_card(loaded_data):
    """Verify Adequacy Ratio KPI Card in District Inspector."""
    from src.analytics.profile import compute_regency_profile
    from src.app.components.inspector import render_inspector_card_html
    from src.analytics.exporter import generate_district_policy_brief

    profile = compute_regency_profile(loaded_data.gdf, loaded_data.bivariate_df, 1101)
    assert hasattr(profile, "adequacy_ratio")
    assert hasattr(profile, "readiness_balance")

    # Dark theme HTML
    card_dark = render_inspector_card_html(profile, is_bright=False, lang="id")
    assert "Rasio Kecukupan Kesiapsiagaan" in card_dark
    assert f"{profile.adequacy_ratio:.1f}%" in card_dark
    assert "Kapasitas mitigasi memenuhi" in card_dark

    # Light theme HTML
    card_light = render_inspector_card_html(profile, is_bright=True, lang="en")
    assert "Readiness Adequacy Ratio" in card_light
    assert f"{profile.adequacy_ratio:.1f}%" in card_light
    assert "Mitigation capacity meets" in card_light
    assert "#ffffff" in card_light

    # Exporter Policy Brief
    brief_id = generate_district_policy_brief(profile, lang="id")
    assert "Neraca Kesiapsiagaan Bersih" in brief_id
    assert "Rasio Kecukupan Kesiapsiagaan" in brief_id

    brief_en = generate_district_policy_brief(profile, lang="en")
    assert "Net Readiness Balance" in brief_en
    assert "Readiness Adequacy Ratio" in brief_en



