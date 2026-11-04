"""Indonesian Disaster Readiness & Gap Explorer (IDR-GE) — Streamlit Viewport."""
import sys
from pathlib import Path

# Ensure project root is on sys.path when launched directly via streamlit
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import textwrap
from typing import Optional, Dict, Any
import streamlit as st
import geopandas as gpd
import pandas as pd

from src.config import (
    COLOR_TOKENS,
    DEFAULT_MAP_CENTER,
    DARK_COLOR_TOKENS,
    LIGHT_COLOR_TOKENS,
    get_theme_tokens,
)
from src.api.routes import get_data_manager
from src.analytics.corridors import (
    CORRIDORS,
    get_corridor_subgraph,
    get_corridor_weights,
    compute_corridor_moran,
)
from src.analytics.simulation import (
    simulate_policy_intervention,
    SimulationResult,
)
from src.analytics.moran import compute_morans_i, compute_lisa
from src.analytics.profile import compute_regency_profile
from src.analytics.provincial import (
    compute_provincial_profile,
    compute_regional_profile,
    compute_national_profile,
)
from src.analytics.spillover import estimate_spatial_spillover
from src.analytics.exporter import (
    generate_district_policy_brief_pdf,
    generate_provincial_policy_brief_pdf,
    generate_regional_policy_brief_pdf,
    generate_national_policy_brief_pdf,
)
from src.app.components.choropleth import (
    prepare_choropleth_data,
    create_choropleth_deck,
    render_bivariate_legend_html,
    render_lisa_legend_html,
    render_balance_legend_html,
    render_viewport_footer_html,
    render_provincial_hotspots_strip_html,
    get_contingency_matrix_counts,
)
from src.app.components.stats_panel import (
    create_moran_scatterplot,
    create_parity_quadrant_chart,
    render_moran_metrics_html,
    render_top_critical_gap_table_html,
    get_top_critical_gap_table,
)
from src.app.components.sandbox_panel import (
    render_sandbox_controls,
    render_simulation_delta_html,
)
from src.app.components.inspector import (
    render_inspector_card_html,
    render_radar_chart,
)
from src.app.components.spillover_panel import (
    render_spillover_metric_card_html,
    render_spillover_waterfall_chart,
    render_spillover_effects_table,
    render_spillover_effects_table_html,
)
from src.app.i18n import t, get_indicator_label


@st.cache_resource(show_spinner=False)
def get_cached_data_manager():
    """Load and cache the spatial datasets and weights matrix in memory."""
    dm = get_data_manager()
    dm.load()
    return dm


def get_antislop_css(
    theme_tokens: Optional[Dict[str, str]] = None,
    theme: Optional[str] = None,
) -> str:
    """Return strict anti-slop CSS overrides matching dark or light analytical design tokens."""
    tokens = theme_tokens or COLOR_TOKENS
    is_bright = (theme == "Light Theme") if theme is not None else (tokens.get("bg-primary") == "#f8fafc")
    bg_primary = tokens.get("bg-primary", "#f8fafc" if is_bright else "#1a2332")
    bg_surface = tokens.get("bg-surface", "#ffffff" if is_bright else "#161e2a")
    border_subtle = tokens.get("border-subtle", "#cbd5e1" if is_bright else "#2a384c")
    text_primary = tokens.get("text-primary", "#0f172a" if is_bright else "#f1f5f9")
    text_muted = tokens.get("text-muted", "#64748b" if is_bright else "#94a3b8")

    if is_bright:
        theme_specific_css = f"""
        /* High-contrast Light Theme sidebar overrides */
        section[data-testid="stSidebar"] {{
            background-color: #f1f5f9 !important;
            border-right: 1px solid #cbd5e1 !important;
        }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label {{
            color: #0f172a !important;
        }}
        /* Right analytics panel toggle & form controls */
        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] label span,
        div[data-testid="stRadio"] div[role="radiogroup"] label {{
            color: #0f172a !important;
        }}
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stSlider"] label p {{
            color: #0f172a !important;
        }}
        /* Expander overrides */
        div[data-testid="stExpander"],
        div.stExpander,
        details {{
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
        }}
        div[data-testid="stExpander"] summary,
        div.stExpander summary,
        details summary {{
            background-color: #f8fafc !important;
            color: #0f172a !important;
        }}
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary svg,
        div.stExpander summary span,
        div.stExpander summary p,
        div.stExpander summary svg,
        details summary span,
        details summary p,
        details summary svg {{
            color: #0f172a !important;
            fill: #0f172a !important;
        }}
        div[data-testid="stExpanderDetails"] {{
            background-color: #ffffff !important;
            color: #0f172a !important;
        }}
        /* High-contrast Light Theme Top 10 Table & Dataframe overrides */
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            color: #0f172a !important;
        }}
        div[data-testid="stTable"] th,
        table th {{
            background-color: #f1f5f9 !important;
            color: #0f172a !important;
            border-bottom: 1px solid #e2e8f0 !important;
        }}
        div[data-testid="stTable"] td,
        table td {{
            border-bottom: 1px solid #e2e8f0 !important;
            color: #0f172a !important;
        }}
        /* Enhanced Selectbox affordance - Light Mode */
        div[data-testid="stSelectbox"] > div,
        div[data-baseweb="select"] > div,
        div.stSelectbox div[role="combobox"],
        div.stSelectbox div[class*="react-aria-Group"],
        div[class*="react-aria-Group"],
        div.stSelectbox input,
        input[class*="react-aria-Input"] {{
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            padding-left: 4px !important;
            padding-right: 4px !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }}
        div[data-testid="stSelectbox"] > div:hover,
        div[data-baseweb="select"] > div:hover,
        div[data-testid="stSelectbox"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {{
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 1px #38bdf8 !important;
        }}
        div[data-testid="stSelectbox"] svg,
        div[data-baseweb="select"] svg {{
            color: #64748b !important;
            fill: #64748b !important;
        }}
        div[data-baseweb="select"] span,
        div.stSelectbox span,
        div[class*="react-aria-Group"] span {{
            color: #0f172a !important;
        }}
        div[data-baseweb="popover"] ul,
        div[data-testid="stSelectboxVirtualDropdown"],
        ul[class*="react-aria-ListBox"] {{
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }}
        div[data-baseweb="popover"] li,
        li[class*="react-aria-ListBoxItem"] {{
            color: #0f172a !important;
            background-color: #ffffff !important;
        }}
        button[kind="secondary"] {{
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }}
        /* Streamlit Top Header Styling - Light Theme */
        header[data-testid="stHeader"] {{
            background-color: #ffffff !important;
            border-bottom: 1px solid #e2e8f0 !important;
        }}
        header[data-testid="stHeader"] svg,
        header[data-testid="stHeader"] button {{
            color: #0f172a !important;
            fill: #0f172a !important;
        }}
        .idr-header {{
            border-bottom-color: #e2e8f0 !important;
        }}
        .idr-title {{
            color: #0f172a !important;
        }}
        .idr-subtitle {{
            color: #475569 !important;
        }}
        """
    else:
        theme_specific_css = f"""
        section[data-testid="stSidebar"] {{
            background-color: {bg_surface} !important;
            border-right: 1px solid {border_subtle} !important;
        }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {text_primary};
        }}
        /* Expander dark mode */
        div[data-testid="stExpander"],
        div.stExpander,
        details {{
            background-color: {bg_surface} !important;
            border: 1px solid {border_subtle} !important;
        }}
        div[data-testid="stExpander"] summary,
        div.stExpander summary,
        details summary {{
            background-color: {bg_surface} !important;
            color: {text_primary} !important;
        }}
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary svg,
        div.stExpander summary span,
        div.stExpander summary p,
        div.stExpander summary svg,
        details summary span,
        details summary p,
        details summary svg {{
            color: {text_primary} !important;
            fill: {text_primary} !important;
        }}
        div[data-testid="stExpanderDetails"] {{
            background-color: {bg_surface} !important;
            color: {text_primary} !important;
        }}
        /* Enhanced Selectbox affordance - Dark Mode */
        div[data-testid="stSelectbox"] > div,
        div[data-baseweb="select"] > div,
        div.stSelectbox div[role="combobox"],
        div.stSelectbox div[class*="react-aria-Group"],
        div[class*="react-aria-Group"],
        div.stSelectbox input,
        input[class*="react-aria-Input"] {{
            background-color: #1e293b !important;
            color: {text_primary} !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            padding-left: 4px !important;
            padding-right: 4px !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }}
        div[data-testid="stSelectbox"] > div:hover,
        div[data-baseweb="select"] > div:hover,
        div[data-testid="stSelectbox"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {{
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 1px #38bdf8 !important;
        }}
        div[data-testid="stSelectbox"] svg,
        div[data-baseweb="select"] svg {{
            color: #94a3b8 !important;
            fill: #94a3b8 !important;
        }}
        div[data-baseweb="select"] span,
        div.stSelectbox span,
        div[class*="react-aria-Group"] span {{
            color: {text_primary} !important;
        }}
        div[data-baseweb="popover"] ul,
        div[data-testid="stSelectboxVirtualDropdown"],
        ul[class*="react-aria-ListBox"] {{
            background-color: #161e2a !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }}
        div[data-baseweb="popover"] li,
        li[class*="react-aria-ListBoxItem"] {{
            color: {text_primary} !important;
            background-color: #161e2a !important;
        }}
        /* High-contrast Dark Theme Top 10 Table & Dataframe overrides */
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            background-color: #1a2332 !important;
            border: 1px solid #2a384c !important;
            color: #f1f5f9 !important;
        }}
        div[data-testid="stTable"] th,
        table th {{
            background-color: #0f172a !important;
            color: #f1f5f9 !important;
            border-bottom: 1px solid #2a384c !important;
        }}
        div[data-testid="stTable"] td,
        table td {{
            border-bottom: 1px solid #2a384c !important;
            color: #f1f5f9 !important;
        }}
        /* Streamlit Top Header Styling - Dark Theme */
        header[data-testid="stHeader"] {{
            background-color: #0f141c !important;
            border-bottom: 1px solid #1e293b !important;
        }}
        header[data-testid="stHeader"] svg,
        header[data-testid="stHeader"] button {{
            color: #f1f5f9 !important;
            fill: #f1f5f9 !important;
        }}
        .idr-header {{
            border-bottom-color: #1e293b !important;
        }}
        .idr-title {{
            color: #f1f5f9 !important;
        }}
        .idr-subtitle {{
            color: #94a3b8 !important;
        }}
        """

    return textwrap.dedent(f"""
        <style>
        /* Base typography & backgrounds */
        .stApp {{
            background-color: {bg_primary};
            color: {text_primary};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        /* Text selection highlight - prevent browser brown strip */
        ::selection {{
            background-color: rgba(56, 189, 248, 0.25) !important;
            color: inherit !important;
        }}
        {theme_specific_css}
        /* Clean vertical clearance without excessive blank space (padding-top: 2rem minimum baseline preserved) */
        .block-container,
        div[data-testid="stMainBlockContainer"] {{
            padding-top: 4rem !important;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }}
        /* Tables & dataframes: flat matte borders, zero floating shadow */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {border_subtle};
            border-radius: 2px;
            background-color: {bg_surface};
        }}
        /* Plotly container clean edge */
        .js-plotly-plot {{
            border: 1px solid {border_subtle};
            border-radius: 2px;
        }}
        /* PyDeck map container clean edge and strict boundary containment */
        .deck-widget-container,
        div[data-testid="stDeckGlChart"],
        .stDeckGlJsonChart {{
            border: 1px solid {border_subtle};
            border-radius: 4px;
            overflow: hidden !important;
            box-sizing: border-box !important;
            max-height: 520px;
        }}
        .deck-widget-container iframe,
        div[data-testid="stDeckGlChart"] iframe,
        div[data-testid="stDeckGlChart"] canvas,
        .deck-widget-container canvas {{
            width: 100% !important;
            height: 100% !important;
            max-height: 500px !important;
            object-fit: contain;
            box-sizing: border-box;
        }}
        /* Top-of-sidebar language segmented control clean styling */
        div[data-testid="stSegmentedControl"],
        div[data-testid="stButtonGroup"] {{
            width: 100% !important;
            margin-bottom: 6px !important;
        }}
        /* Header bar in normal document flow with clean vertical clearance */
        .idr-header {{
            border-bottom: 1px solid {border_subtle};
            padding-top: 4px;
            padding-bottom: 12px;
            margin-top: 0.5rem;
            margin-bottom: 16px;
            background: transparent !important;
            background-color: transparent !important;
        }}
        .idr-title {{
            font-size: 20px;
            font-weight: 700;
            line-height: 1.4;
            color: {text_primary};
            letter-spacing: -0.3px;
            margin: 0;
            padding: 0;
            background: transparent !important;
            background-color: transparent !important;
        }}
        .idr-subtitle {{
            font-size: 12px;
            line-height: 1.5;
            color: {text_muted};
            font-family: monospace;
            margin-top: 4px;
            background: transparent !important;
            background-color: transparent !important;
        }}
        /* Responsive Mobile Layout (@media max-width: 768px) */
        @media (max-width: 768px) {{
            /* Collapse multi-column layout into vertical single-column stack */
            div[data-testid="stHorizontalBlock"] {{
                flex-direction: column !important;
                gap: 16px !important;
            }}
            div[data-testid="column"] {{
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }}
            /* Fixed touch-friendly mobile PyDeck map height with smooth navigation */
            .deck-widget-container, div[data-testid="stDeckGlChart"], iframe[title="pydeck"] {{
                height: 380px !important;
                min-height: 380px !important;
                max-height: 420px !important;
                touch-action: pan-x pan-y pinch-zoom;
            }}
            /* Mobile responsive adjustments for the footer container */
            .idr-viewport-footer {{
                flex-direction: column !important;
                width: 100% !important;
            }}
            .idr-footer-left, .idr-footer-right {{
                width: 100% !important;
                min-width: 100% !important;
                border-left: none !important;
                padding-left: 0 !important;
                margin-top: 8px !important;
            }}
            .block-container,
            div[data-testid="stMainBlockContainer"] {{
                padding-top: 4.5rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}
            .idr-header {{
                margin-top: 0.5rem !important;
            }}
        }}
        </style>
    """).strip()


def get_active_theme_base() -> str:
    """Detect active theme base ('dark' or 'light').

    1. Checks st.context.theme.base (from Streamlit's native 3-dots menu: System / Light / Dark)
    2. Fallback: query parameter ?theme=dark or ?theme=light
    3. Fallback: session state st.session_state.get('theme', 'dark')
    """
    # 1. Native theme context (from 3-dots menu)
    try:
        if hasattr(st, "context") and hasattr(st.context, "theme") and st.context.theme is not None:
            ctx_theme = st.context.theme
            base = getattr(ctx_theme, "base", None)
            if not base and hasattr(ctx_theme, "type"):
                base = getattr(ctx_theme, "type", None)
            if not base and hasattr(ctx_theme, "get"):
                base = ctx_theme.get("base") or ctx_theme.get("type")
            if base in ("dark", "light"):
                return base
    except Exception:
        pass

    # 2. Query parameter fallback (?theme=dark or ?theme=light)
    try:
        if hasattr(st, "query_params"):
            qp = st.query_params.get("theme", "").lower()
            if qp in ("dark", "light"):
                return qp
    except Exception:
        pass

    # 3. Session state fallback
    try:
        sess = str(st.session_state.get("theme", "")).lower()
        if sess in ("dark", "light"):
            return sess
        if "dark_mode" in st.session_state:
            return "dark" if st.session_state["dark_mode"] else "light"
    except Exception:
        pass

    return "dark"


def apply_antislop_styling(
    theme_tokens: Optional[Dict[str, str]] = None,
    theme: Optional[str] = None,
):
    """Inject strict anti-slop CSS overrides matching dark or light analytical design tokens."""
    st.markdown(get_antislop_css(theme_tokens, theme), unsafe_allow_html=True)


def main():
    """Main Streamlit execution entrypoint."""
    st.set_page_config(
        page_title="IDR-GE | Disaster Readiness & Gap Explorer",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load spatial dataset singleton
    dm = get_cached_data_manager()
    gdf = dm.gdf
    bivariate_df = dm.bivariate_df
    weights = dm.weights

    # -----------------------------------------------------------------------
    # Native Theme Detection & State Synchronization
    # -----------------------------------------------------------------------
    theme_base = get_active_theme_base()
    is_bright = (theme_base == "light")
    st.session_state["theme"] = theme_base
    st.session_state["dark_mode"] = not is_bright

    theme_choice = "Light Theme" if is_bright else "Dark Theme"
    tokens = get_theme_tokens(theme_choice)
    apply_antislop_styling(tokens, theme=theme_choice)

    # -----------------------------------------------------------------------
    # Clean Language Selector Placement (Top of Sidebar)
    # -----------------------------------------------------------------------
    if "lang" not in st.session_state:
        qp_lang = st.query_params.get("lang", "").lower() if hasattr(st, "query_params") else ""
        if qp_lang in ("id", "en"):
            st.session_state["lang"] = qp_lang
        else:
            st.session_state["lang"] = st.session_state.get("app_lang", "id")

    with st.sidebar:
        if hasattr(st, "segmented_control"):
            lang_choice = st.segmented_control(
                label="Language / Bahasa",
                options=["ID", "EN"],
                default=st.session_state.get("lang", "id").upper(),
                format_func=lambda opt: f"🇮🇩 {opt}" if opt == "ID" else f"🇬🇧 {opt}",
                label_visibility="collapsed",
                key="app_lang_selector",
            )
        else:
            lang_choice = st.radio(
                label="Language / Bahasa",
                options=["ID", "EN"],
                index=0 if st.session_state.get("lang", "id").lower() == "id" else 1,
                format_func=lambda opt: f"🇮🇩 {opt}" if opt == "ID" else f"🇬🇧 {opt}",
                horizontal=True,
                label_visibility="collapsed",
                key="app_lang_selector",
            )

        if not lang_choice:
            lang_choice = st.session_state.get("lang", "id").upper()

        st.session_state["lang"] = lang_choice.lower()
        st.session_state["app_lang"] = st.session_state["lang"]
        lang_code = st.session_state["lang"]

        sidebar_brand = textwrap.dedent(f"""
            <div style='padding-bottom: 8px; margin-bottom: 12px; border-bottom: 1px solid {tokens["border-subtle"]};'>
                <div style='font-size: 15px; font-weight: 700; color: {tokens["text-primary"]};'>IDR-GE</div>
                <div style='font-size: 10px; color: {tokens["text-muted"]}; font-family: monospace;'>{t("sidebar_brand_sub", lang_code)}</div>
            </div>
        """).strip()
        st.markdown(sidebar_brand, unsafe_allow_html=True)

    layer_biv_label = t("layer_bivariate", lang_code)
    layer_balance_label = t("layer_balance", lang_code)
    layer_lisa_label = t("layer_lisa", lang_code)
    layer_mode_display = st.sidebar.radio(
        t("map_layer_mode", lang_code),
        options=[layer_biv_label, layer_balance_label, layer_lisa_label],
        index=0,
    )
    if layer_mode_display == layer_biv_label:
        layer_mode = "Bivariate Risk vs Mitigation (3x3)"
    elif layer_mode_display == layer_balance_label:
        layer_mode = "Readiness Balance (Deficit-Surplus)"
    else:
        layer_mode = "LISA Spatial Clusters"

    available_vars = [
        "IRBI_2024",
        "EWS",
        "RAMBU",
        "PERKAP",
        "RAWAT_SUNGAI",
        "ADA_BENCANA",
        "TERDAMPAK",
        "KEMISKINAN",
        "IPM",
        "GOTONG_ROYONG",
    ]

    if layer_mode == "LISA Spatial Clusters":
        selected_var = st.sidebar.selectbox(
            t("spatial_indicator", lang_code),
            options=available_vars,
            format_func=lambda v: f"{v} ({get_indicator_label(v, lang_code)})",
            index=0,
        )
        p_threshold = st.sidebar.select_slider(
            t("significance_level", lang_code),
            options=[0.01, 0.05, 0.10],
            value=0.05,
        )
    else:
        selected_var = "IRBI_2024"
        p_threshold = 0.05

    # -----------------------------------------------------------------------
    # Policy Intervention Sandbox (Sidebar)
    # -----------------------------------------------------------------------
    with st.sidebar.expander(t("policy_sandbox", lang_code), expanded=True):
        signal_reduction, gotong_royong_boost, is_sim_active = render_sandbox_controls(lang=lang_code)

    if is_sim_active:
        sim_result = simulate_policy_intervention(
            bivariate_df,
            signal_reduction_pct=signal_reduction,
            gotong_royong_boost_pct=gotong_royong_boost,
        )
        simulated_df = sim_result.df
    else:
        sim_result = None
        simulated_df = None

    # -----------------------------------------------------------------------
    # Regional Corridor & Province Filters
    # -----------------------------------------------------------------------
    nat_label = t("national_all_corridors", lang_code)
    corridor_options = [nat_label] + CORRIDORS
    selected_corridor_display = st.sidebar.selectbox(t("regional_corridor", lang_code), options=corridor_options, index=0)
    selected_corridor = "National (All 7 Corridors)" if selected_corridor_display == nat_label else selected_corridor_display

    if selected_corridor == "National (All 7 Corridors)":
        corridor_gdf = gdf
        corridor_biv = bivariate_df
        corridor_weights = weights

        all_prov_label = t("all_provinces", lang_code)
        all_provinces = [all_prov_label] + sorted(gdf["PROVINSI"].unique().tolist())
        selected_province_display = st.sidebar.selectbox(t("filter_province", lang_code), options=all_provinces, index=0)
        selected_province = "All 38 Provinces" if selected_province_display == all_prov_label else selected_province_display

        if selected_province != "All 38 Provinces":
            filtered_gdf = gdf[gdf["PROVINSI"] == selected_province].copy()
            filtered_biv = bivariate_df[bivariate_df["PROVINSI"] == selected_province].copy()
        else:
            filtered_gdf = gdf
            filtered_biv = bivariate_df

        center_lat = DEFAULT_MAP_CENTER["lat"]
        center_lon = DEFAULT_MAP_CENTER["lon"]
        zoom = DEFAULT_MAP_CENTER["zoom"]
    else:
        corridor_gdf = get_corridor_subgraph(gdf, selected_corridor)
        corridor_biv = get_corridor_subgraph(bivariate_df, selected_corridor)
        corridor_weights = get_corridor_weights(corridor_gdf)

        corr_all_prov_label = f"{t('all_prefix', lang_code)} {selected_corridor} {t('provinces_suffix', lang_code)}"
        prov_options = [corr_all_prov_label] + sorted(corridor_gdf["PROVINSI"].unique().tolist())
        selected_province_display = st.sidebar.selectbox(t("filter_province", lang_code), options=prov_options, index=0)
        selected_province = f"All {selected_corridor} Provinces" if selected_province_display == corr_all_prov_label else selected_province_display

        if selected_province != f"All {selected_corridor} Provinces":
            filtered_gdf = corridor_gdf[corridor_gdf["PROVINSI"] == selected_province].copy()
            filtered_biv = corridor_biv[corridor_biv["PROVINSI"] == selected_province].copy()
        else:
            filtered_gdf = corridor_gdf
            filtered_biv = corridor_biv

        # Dynamically compute bounds and centroid
        bounds = filtered_gdf.total_bounds
        center_lon = float((bounds[0] + bounds[2]) / 2.0)
        center_lat = float((bounds[1] + bounds[3]) / 2.0)
        span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        if span > 15:
            zoom = 5.0
        elif span > 8:
            zoom = 5.8
        elif span > 4:
            zoom = 6.5
        else:
            zoom = 7.0

    st.sidebar.markdown("---")
    sidebar_meta = textwrap.dedent(f"""
        <div style='font-family: monospace; font-size: 11px; color: {tokens["text-muted"]}; line-height: 1.6;'>
            <div>{t("units_in_view", lang_code)}: <b style='color: {tokens["text-primary"]};'>{len(filtered_gdf)} {t("units_label", lang_code)}</b></div>
            <div>{t("corridor_meta", lang_code)}: <b style='color: {tokens["accent-focus"]};'>{selected_corridor_display}</b></div>
            <div>{t("topology_meta", lang_code)}: <b style='color: {tokens["text-primary"]};'>KNN (0 Islands)</b></div>
            <div>CRS: <b style='color: {tokens["text-primary"]};'>EPSG:4326 (WGS84)</b></div>
            <div>Engine: <b style='color: {tokens["text-primary"]};'>PySAL + FastAPI</b></div>
        </div>
    """).strip()
    st.sidebar.markdown(sidebar_meta, unsafe_allow_html=True)

    is_specific_province = (
        selected_province != "All 38 Provinces"
        and not selected_province.startswith("All ")
    )

    # National Executive Policy Brief PDF Download Button
    if selected_corridor == "National (All 7 Corridors)" and not is_specific_province:
        try:
            nat_profile = compute_national_profile(gdf)
            nat_pdf_bytes = generate_national_policy_brief_pdf(
                nat_profile,
                lang=lang_code,
            )
            st.sidebar.download_button(
                label=t("download_national_pdf_btn", lang_code),
                data=nat_pdf_bytes,
                file_name=f"Policy_Brief_Nasional_Indonesia_{lang_code.upper()}.pdf",
                mime="application/pdf",
                key="dl_nat_pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.sidebar.caption(f"National brief notice: {exc}")

    # Regional Executive Policy Brief PDF Download Button
    if selected_corridor in CORRIDORS:
        try:
            reg_profile = compute_regional_profile(gdf, selected_corridor)
            safe_reg_slug = selected_corridor.lower().replace(" ", "_").replace(".", "").replace("-", "_")
            reg_pdf_bytes = generate_regional_policy_brief_pdf(
                reg_profile,
                lang=lang_code,
            )
            st.sidebar.download_button(
                label=t("download_region_pdf_btn", lang_code, region=selected_corridor),
                data=reg_pdf_bytes,
                file_name=f"Policy_Brief_Wilayah_{safe_reg_slug}_{lang_code.upper()}.pdf",
                mime="application/pdf",
                key=f"dl_reg_pdf_{safe_reg_slug}",
                use_container_width=True,
            )
        except Exception as exc:
            st.sidebar.caption(f"Regional brief notice: {exc}")

    # Provincial Executive Policy Brief PDF Download Button
    if is_specific_province:
        try:
            prov_profile = compute_provincial_profile(gdf, selected_province)
            safe_prov_slug = selected_province.lower().replace(" ", "_").replace(".", "")
            prov_pdf_bytes = generate_provincial_policy_brief_pdf(
                prov_profile,
                lang=lang_code,
            )
            st.sidebar.download_button(
                label=t("download_prov_pdf_btn", lang_code, prov=selected_province),
                data=prov_pdf_bytes,
                file_name=f"Policy_Brief_Provinsi_{safe_prov_slug}_{lang_code.upper()}.pdf",
                mime="application/pdf",
                key=f"dl_prov_pdf_{safe_prov_slug}",
                use_container_width=True,
            )
        except Exception as exc:
            st.sidebar.caption(f"Provincial brief notice: {exc}")

    # -----------------------------------------------------------------------
    # Main Header
    # -----------------------------------------------------------------------
    scope_prov_display = selected_province if not selected_province.startswith("All ") else (t("all_provinces", lang_code) if selected_corridor == "National (All 7 Corridors)" else f"{t('all_prefix', lang_code)} {selected_corridor} {t('provinces_suffix', lang_code)}")
    scope_name = (
        f"{selected_corridor_display} ({scope_prov_display})"
        if selected_corridor != "National (All 7 Corridors)"
        else scope_prov_display
    )
    sim_badge = t("sim_active_badge", lang_code) if is_sim_active else ""
    main_header = textwrap.dedent(f"""
        <div class="idr-header">
            <div class="idr-title" style="background: transparent !important; background-color: transparent !important;">{t("app_title", lang_code)}{sim_badge}</div>
            <div class="idr-subtitle" style="background: transparent !important; background-color: transparent !important;">{t("app_subtitle_prefix", lang_code)}: {layer_mode_display} | {t("app_scope_prefix", lang_code)}: {scope_name} ({len(filtered_gdf)} {t("units_label", lang_code)})</div>
        </div>
    """).strip()
    st.markdown(main_header, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 70% Map Viewport / 30% Analytics Panel Split
    # -----------------------------------------------------------------------
    col_map, col_stats = st.columns([7, 3], gap="medium")

    with col_map:
        # Monospace delta readout placed prominently above map when simulation active
        if is_sim_active and sim_result is not None:
            st.markdown(render_simulation_delta_html(sim_result, is_bright=is_bright, theme_tokens=tokens), unsafe_allow_html=True)

        # Prepare spatial choropleth data (passing simulated_df if active)
        prepared_data = prepare_choropleth_data(
            gdf=filtered_gdf,
            bivariate_df=filtered_biv,
            layer_mode=layer_mode,
            lisa_variable=selected_var,
            p_threshold=p_threshold,
            simulated_df=simulated_df,
        )

        deck = create_choropleth_deck(
            prepared_data,
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=zoom,
            is_simulated=is_sim_active,
            theme_tokens=tokens,
        )
        st.pydeck_chart(deck, use_container_width=True)

        # High-density matte legend & integrated 2-subcolumn footer below map
        if layer_mode == "Bivariate Risk vs Mitigation (3x3)":
            st.markdown(
                render_viewport_footer_html(
                    bivariate_df=filtered_biv,
                    gdf=filtered_gdf,
                    theme_tokens=tokens,
                    is_bright=is_bright,
                    lang=lang_code,
                ),
                unsafe_allow_html=True,
            )
        elif layer_mode == "Readiness Balance (Deficit-Surplus)":
            st.markdown(
                render_balance_legend_html(theme_tokens=tokens, is_bright=is_bright, lang=lang_code),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(render_lisa_legend_html(theme_tokens=tokens), unsafe_allow_html=True)

        # Compact horizontal analytical strip: Top Provincial Priority Hotspots
        st.markdown(
            render_provincial_hotspots_strip_html(
                bivariate_df=filtered_biv,
                gdf=filtered_gdf,
                is_bright=is_bright,
                theme_tokens=tokens,
                lang=lang_code,
            ),
            unsafe_allow_html=True,
        )

    with col_stats:
        opt_moran = t("view_moran", lang_code)
        opt_parity = t("view_parity", lang_code)
        opt_spill = t("view_spillover", lang_code)
        analytics_mode_display = st.radio(
            t("analytics_view", lang_code),
            options=[opt_moran, opt_parity, opt_spill],
            index=0,
            horizontal=True,
        )
        if analytics_mode_display == opt_moran:
            analytics_mode = "Spatial Autocorrelation (Moran)"
        elif analytics_mode_display == opt_parity:
            analytics_mode = "4-Quadrant Parity (M vs R)"
        else:
            analytics_mode = "Spatial Spillover & Multipliers"

        if analytics_mode == "Spatial Autocorrelation (Moran)":
            # Moran metrics computation on localized corridor or national connectivity
            if selected_corridor == "National (All 7 Corridors)":
                moran_res = compute_morans_i(corridor_gdf, column=selected_var, weights=corridor_weights)
                lisa_res = compute_lisa(corridor_gdf, column=selected_var, weights=corridor_weights, p_threshold=p_threshold)
                scatter_fig = create_moran_scatterplot(
                    gdf=corridor_gdf,
                    variable=selected_var,
                    weights=corridor_weights,
                    lisa_clusters=lisa_res["lisa_cluster"],
                    corridor_name=None,
                    theme_tokens=tokens,
                    is_bright=is_bright,
                )
            else:
                moran_res = compute_corridor_moran(corridor_gdf, variable=selected_var, weights=corridor_weights)
                lisa_res = compute_lisa(corridor_gdf, column=selected_var, weights=corridor_weights, p_threshold=p_threshold)
                scatter_fig = create_moran_scatterplot(
                    gdf=corridor_gdf,
                    variable=selected_var,
                    weights=corridor_weights,
                    lisa_clusters=lisa_res["lisa_cluster"],
                    corridor_name=selected_corridor,
                    theme_tokens=tokens,
                    is_bright=is_bright,
                )

            st.markdown(render_moran_metrics_html(moran_res, theme_tokens=tokens, is_bright=is_bright, lang=lang_code), unsafe_allow_html=True)
            st.plotly_chart(scatter_fig, use_container_width=True, config={"displayModeBar": False})

            # Critical Gap ranking table
            table_title = (
                t("top_critical_gap_sim_title", lang_code)
                if is_sim_active
                else t("top_critical_gap_title", lang_code)
            )
            st.markdown(
                f"<div style='font-size: 11px; font-weight: 700; color: {tokens['text-primary']}; text-transform: uppercase; margin-top: 8px; margin-bottom: 6px; font-family: monospace;'>{table_title}</div>",
                unsafe_allow_html=True,
            )

            if is_sim_active and simulated_df is not None:
                # Update critical gap table to show simulated rankings when active
                sim_filtered = simulated_df[simulated_df["KODE_WIL"].isin(filtered_gdf["KODE_WIL"])].copy()
                table_source = sim_filtered.copy()
                table_source["gap_score"] = table_source["simulated_gap_score"]
                table_source["mitigation_score"] = table_source["simulated_mitigation_score"]
                table_source["bivariate_label"] = table_source["simulated_bivariate_label"]
                st.markdown(
                    render_top_critical_gap_table_html(
                        table_source,
                        n=10,
                        is_bright=is_bright,
                        lang=lang_code,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    render_top_critical_gap_table_html(
                        filtered_biv,
                        n=10,
                        is_bright=is_bright,
                        lang=lang_code,
                    ),
                    unsafe_allow_html=True,
                )

        elif analytics_mode == "4-Quadrant Parity (M vs R)":
            target_kw = st.session_state.get("inspector_district_kw")
            fig_parity = create_parity_quadrant_chart(
                filtered_biv,
                target_kode=target_kw,
                theme_tokens=tokens,
                is_bright=is_bright,
                lang=lang_code,
            )
            st.plotly_chart(fig_parity, use_container_width=True, config={"displayModeBar": False})

            # Critical Gap ranking table
            table_title = (
                t("top_critical_gap_sim_title", lang_code)
                if is_sim_active
                else t("top_critical_gap_title", lang_code)
            )
            st.markdown(
                f"<div style='font-size: 11px; font-weight: 700; color: {tokens['text-primary']}; text-transform: uppercase; margin-top: 8px; margin-bottom: 6px; font-family: monospace;'>{table_title}</div>",
                unsafe_allow_html=True,
            )
            if is_sim_active and simulated_df is not None:
                sim_filtered = simulated_df[simulated_df["KODE_WIL"].isin(filtered_gdf["KODE_WIL"])].copy()
                table_source = sim_filtered.copy()
                table_source["gap_score"] = table_source["simulated_gap_score"]
                table_source["mitigation_score"] = table_source["simulated_mitigation_score"]
                table_source["bivariate_label"] = table_source["simulated_bivariate_label"]
                st.markdown(
                    render_top_critical_gap_table_html(
                        table_source,
                        n=10,
                        is_bright=is_bright,
                        lang=lang_code,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    render_top_critical_gap_table_html(
                        filtered_biv,
                        n=10,
                        is_bright=is_bright,
                        lang=lang_code,
                    ),
                    unsafe_allow_html=True,
                )

        else:
            # Spatial Spillover & Multipliers Mode
            active_corr = None if selected_corridor == "National (All 7 Corridors)" else selected_corridor
            active_w = corridor_weights if selected_corridor == "National (All 7 Corridors)" else None

            spec_sar = t("spec_sar", lang_code)
            spec_slx = t("spec_slx", lang_code)
            model_opt = st.selectbox(
                t("econometric_spec", lang_code),
                options=[spec_sar, spec_slx],
                index=0,
            )
            model_type = "lag" if "SAR" in model_opt else "slx"

            spillover_res = estimate_spatial_spillover(
                gdf=gdf,
                corridor=active_corr,
                weights=active_w,
                model_type=model_type,
            )

            st.markdown(render_spillover_metric_card_html(spillover_res, is_bright=is_bright, lang=lang_code), unsafe_allow_html=True)
            spillover_fig = render_spillover_waterfall_chart(spillover_res, is_bright=is_bright)
            st.plotly_chart(spillover_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"<div style='font-size: 11px; font-weight: 700; color: {tokens['text-primary']}; text-transform: uppercase; margin-top: 8px; margin-bottom: 6px; font-family: monospace;'>{t('cross_boundary_title', lang_code)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                render_spillover_effects_table_html(
                    spillover_res,
                    is_bright=is_bright,
                    lang=lang_code,
                ),
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # District Micro-Profile Inspector (Expandable Drawer)
    # -----------------------------------------------------------------------
    st.markdown("---")
    with st.expander(t("district_inspector", lang_code), expanded=True):
        # Use filtered_gdf so options strictly match selected province
        target_gdf = filtered_gdf if len(filtered_gdf) > 0 else gdf
        reg_records = target_gdf.sort_values("KAB_KOTA")[["KODE_WIL", "KAB_KOTA", "PROVINSI"]].to_dict("records")
        reg_display = {
            r["KODE_WIL"]: f"{r['KAB_KOTA']} ({r['PROVINSI']}) [{r['KODE_WIL']}]"
            for r in reg_records
        }
        keys_list = list(reg_display.keys())

        # Determine safe default index: keep previous selection if still in filtered options, else 0
        default_idx = 0
        current_inspector_kw = st.session_state.get("inspector_district_kw")
        if current_inspector_kw in keys_list:
            default_idx = keys_list.index(current_inspector_kw)

        col_ctrl, _ = st.columns([1, 1])
        with col_ctrl:
            selected_kode = st.selectbox(
                t("select_unit_inspect", lang_code, count=len(target_gdf)),
                options=keys_list,
                format_func=lambda kw: reg_display[kw],
                index=default_idx,
            )
            st.session_state["inspector_district_kw"] = selected_kode

        profile = compute_regency_profile(gdf, bivariate_df, kode_wil=selected_kode)
        radar_fig = render_radar_chart(profile, is_bright=is_bright)

        col_insp_card, col_insp_radar = st.columns([1, 1], gap="medium")
        with col_insp_card:
            st.markdown(render_inspector_card_html(profile, is_bright=is_bright, lang=lang_code), unsafe_allow_html=True)
            # One-click Executive Policy Brief Exporters (PDF & Markdown)
            try:
                insp_spill = estimate_spatial_spillover(gdf, corridor=profile.corridor)
            except Exception:
                insp_spill = None

            active_lang = st.session_state.get("lang", st.session_state.get("app_lang", lang_code))
            safe_name = profile.kab_kota.lower().replace(" ", "_").replace(".", "")

            try:
                pdf_bytes = generate_district_policy_brief_pdf(
                    profile,
                    spatial_context=insp_spill,
                    lang=active_lang,
                    radar_fig=radar_fig,
                )
                st.download_button(
                    label=t("download_pdf_brief_btn", active_lang, kab=profile.kab_kota),
                    data=pdf_bytes,
                    file_name=f"Policy_Brief_{profile.kode_wil}_{safe_name}_{active_lang.upper()}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{profile.kode_wil}",
                    use_container_width=True,
                )
            except Exception as exc:
                st.caption(f"PDF exporter notice: {exc}")
        with col_insp_radar:
            st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})


if __name__ == "__main__":
    main()
