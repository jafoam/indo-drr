"""Choropleth spatial visualization component using PyDeck and Plotly."""
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.config import (
    COLOR_TOKENS,
    DEFAULT_MAP_CENTER,
    BIVARIATE_MATRIX_HEX,
    BIVARIATE_PALETTE,
    LISA_PALETTE,
    BALANCE_COLOR_RAMP,
)
from src.analytics.bivariate import compute_bivariate_scores, BIVARIATE_CATEGORY_LABELS
from src.analytics.moran import compute_lisa
import numpy as np


def get_balance_rgba(balance: float) -> List[int]:
    """Map readiness balance (-1.0 to 1.0) into continuous RGBA using BALANCE_COLOR_RAMP."""
    if balance <= -0.40:
        return [185, 28, 28, 200]
    elif balance <= -0.10:
        t = (balance - (-0.40)) / (-0.10 - (-0.40))
        r = int(185 + t * (245 - 185))
        g = int(28 + t * (158 - 28))
        b = int(28 + t * (11 - 28))
        return [r, g, b, 200]
    elif balance <= 0.10:
        t = (balance - (-0.10)) / (0.10 - (-0.10))
        r = int(245 + t * (148 - 245))
        g = int(158 + t * (163 - 158))
        b = int(11 + t * (184 - 11))
        return [r, g, b, 190]
    else:
        t = min(1.0, (balance - 0.10) / (0.50 - 0.10))
        r = int(148 + t * (2 - 148))
        g = int(163 + t * (132 - 163))
        b = int(184 + t * (199 - 184))
        return [r, g, b, 200]


def get_balance_label(balance: float) -> str:
    """Categorical label for readiness balance."""
    if balance < -0.40:
        return "Defisit Ekstrem (< -0.40)"
    elif balance <= -0.10:
        return "Defisit Moderat (-0.40 s/d -0.10)"
    elif balance <= 0.10:
        return "Seimbang (-0.10 s/d +0.10)"
    else:
        return "Surplus Mitigasi (> +0.10)"


def hex_to_rgba(hex_color: str, alpha: int = 200) -> List[int]:
    """Convert hex color string (e.g., '#d03838') to RGBA integer list."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return [r, g, b, alpha]
    return [51, 65, 85, alpha]  # fallback muted slate


def prepare_choropleth_data(
    gdf: gpd.GeoDataFrame,
    bivariate_df: pd.DataFrame,
    layer_mode: str = "Bivariate Risk vs Mitigation (3x3)",
    lisa_variable: str = "IRBI_2024",
    p_threshold: float = 0.05,
    simulated_df: Optional[Any] = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Prepare GeoDataFrame with computed colors and formatted tooltip properties."""
    plot_gdf = gdf.copy()

    # Ensure GeoDataFrame is explicitly projected to WGS84 (EPSG:4326) for PyDeck GeoJsonLayer
    if hasattr(plot_gdf, "crs") and plot_gdf.crs is not None:
        plot_gdf = plot_gdf.to_crs(epsg=4326)
    else:
        plot_gdf = plot_gdf.set_crs(epsg=4326)

    # Support simulated_df passed as kwarg or attribute on result container
    if simulated_df is None and "simulated_df" in kwargs:
        simulated_df = kwargs.get("simulated_df")

    if simulated_df is not None and hasattr(simulated_df, "df"):
        sim_data = simulated_df.df
    elif simulated_df is not None:
        sim_data = simulated_df
    else:
        sim_data = None

    # Ensure KODE_WIL matches as integer
    plot_gdf["KODE_WIL"] = plot_gdf["KODE_WIL"].astype(int)
    biv = bivariate_df.copy()
    biv["KODE_WIL"] = biv["KODE_WIL"].astype(int)

    # Attach bivariate attributes
    biv_cols = [
        "risk_score",
        "mitigation_score",
        "gap_score",
        "readiness_balance",
        "adequacy_ratio",
        "bivariate_category",
        "bivariate_color",
        "bivariate_label",
        "is_critical_gap",
    ]
    for col in biv_cols:
        if col in biv.columns and col not in plot_gdf.columns:
            plot_gdf[col] = biv[col].values

    if "readiness_balance" not in plot_gdf.columns:
        plot_gdf["readiness_balance"] = np.clip(
            plot_gdf["mitigation_score"] - plot_gdf["risk_score"], -1.0, 1.0
        )
    if "adequacy_ratio" not in plot_gdf.columns:
        risk_v = plot_gdf["risk_score"].to_numpy()
        mit_v = plot_gdf["mitigation_score"].to_numpy()
        plot_gdf["adequacy_ratio"] = np.where(
            risk_v > 0,
            np.clip((mit_v / risk_v) * 100.0, 0.0, 200.0),
            100.0,
        )

    # Attach simulated attributes if provided
    if sim_data is not None:
        sim_df = sim_data.copy()
        sim_df["KODE_WIL"] = sim_df["KODE_WIL"].astype(int)

        # Standard simulated columns mapping
        sim_cols = [
            "simulated_mitigation_score",
            "simulated_gap_score",
            "simulated_bivariate_category",
            "simulated_bivariate_color",
            "simulated_bivariate_label",
            "is_critical_gap_simulated",
            "is_resilient_simulated",
        ]
        for col in sim_cols:
            if col in sim_df.columns:
                mapping = dict(zip(sim_df["KODE_WIL"], sim_df[col]))
                plot_gdf[col] = plot_gdf["KODE_WIL"].map(mapping)

        # Ensure explicit simulated_gap_score and simulated_category are attached
        if "simulated_gap_score" in sim_df.columns:
            plot_gdf["simulated_gap_score"] = plot_gdf["KODE_WIL"].map(
                dict(zip(sim_df["KODE_WIL"], sim_df["simulated_gap_score"]))
            )
        elif "gap_score" in sim_df.columns:
            plot_gdf["simulated_gap_score"] = plot_gdf["KODE_WIL"].map(
                dict(zip(sim_df["KODE_WIL"], sim_df["gap_score"]))
            )

        if "simulated_bivariate_category" in sim_df.columns:
            plot_gdf["simulated_category"] = plot_gdf["KODE_WIL"].map(
                dict(zip(sim_df["KODE_WIL"], sim_df["simulated_bivariate_category"]))
            )
        elif "simulated_category" in sim_df.columns:
            plot_gdf["simulated_category"] = plot_gdf["KODE_WIL"].map(
                dict(zip(sim_df["KODE_WIL"], sim_df["simulated_category"]))
            )
        else:
            plot_gdf["simulated_category"] = plot_gdf.get("bivariate_category")

        plot_gdf["formatted_sim_gap"] = plot_gdf["simulated_gap_score"].apply(
            lambda v: f"{v:+.3f}" if pd.notna(v) else ""
        )
        plot_gdf["sim_category_label"] = (
            plot_gdf["simulated_bivariate_label"].fillna("Unknown")
            if "simulated_bivariate_label" in plot_gdf.columns
            else plot_gdf.get("bivariate_label", "")
        )
    else:
        plot_gdf["simulated_gap_score"] = None
        plot_gdf["simulated_category"] = None
        plot_gdf["formatted_sim_gap"] = ""
        plot_gdf["sim_category_label"] = ""

    if layer_mode == "Bivariate Risk vs Mitigation (3x3)":
        if sim_data is not None and "simulated_bivariate_color" in plot_gdf.columns:
            plot_gdf["category_label"] = plot_gdf["sim_category_label"]
            plot_gdf["display_color"] = plot_gdf["simulated_bivariate_color"].fillna("#334155")
        else:
            plot_gdf["category_label"] = plot_gdf["bivariate_label"].fillna("Unknown")
            plot_gdf["display_color"] = plot_gdf["bivariate_color"].fillna("#334155")
        plot_gdf["fill_color"] = plot_gdf["display_color"].apply(lambda h: hex_to_rgba(h, alpha=210))
    elif layer_mode == "Readiness Balance (Deficit-Surplus)":
        plot_gdf["category_label"] = plot_gdf["readiness_balance"].apply(get_balance_label)
        plot_gdf["fill_color"] = plot_gdf["readiness_balance"].apply(get_balance_rgba)
        plot_gdf["display_color"] = plot_gdf["fill_color"].apply(
            lambda rgba: f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}"
        )
    else:  # LISA Spatial Clusters
        lisa_res = compute_lisa(
            plot_gdf,
            column=lisa_variable,
            p_threshold=p_threshold,
        )
        plot_gdf["category_label"] = lisa_res["lisa_cluster"].values
        plot_gdf["display_color"] = lisa_res["color"].values
        plot_gdf["fill_color"] = plot_gdf["display_color"].apply(lambda h: hex_to_rgba(h, alpha=210))

    # Format numeric properties for tooltips
    plot_gdf["formatted_risk"] = plot_gdf["risk_score"].apply(lambda v: f"{v:.3f}")
    plot_gdf["formatted_mitigation"] = plot_gdf["mitigation_score"].apply(lambda v: f"{v:.3f}")
    plot_gdf["formatted_gap"] = plot_gdf["gap_score"].apply(lambda v: f"{v:+.3f}")
    plot_gdf["formatted_balance"] = plot_gdf["readiness_balance"].apply(lambda v: f"{v:+.3f}")
    plot_gdf["formatted_adequacy"] = plot_gdf["adequacy_ratio"].apply(lambda v: f"{v:.1f}%")

    # Keep essential columns including simulated metrics
    keep_cols = [
        "geometry",
        "KODE_WIL",
        "KAB_KOTA",
        "PROVINSI",
        "formatted_risk",
        "formatted_mitigation",
        "formatted_gap",
        "formatted_balance",
        "formatted_adequacy",
        "readiness_balance",
        "adequacy_ratio",
        "simulated_gap_score",
        "simulated_category",
        "formatted_sim_gap",
        "sim_category_label",
        "category_label",
        "display_color",
        "fill_color",
    ]
    return plot_gdf[keep_cols]


def create_choropleth_deck(
    data_gdf: gpd.GeoDataFrame,
    center_lat: float = DEFAULT_MAP_CENTER["lat"],
    center_lon: float = DEFAULT_MAP_CENTER["lon"],
    zoom: float = DEFAULT_MAP_CENTER["zoom"],
    is_simulated: bool = False,
    theme_tokens: Optional[Dict[str, str]] = None,
    map_style: Optional[str] = None,
) -> pdk.Deck:
    """Construct PyDeck visualization deck with strict anti-slop color tokens."""
    # Ensure GeoDataFrame is strictly projected to EPSG:4326 WGS84 for GeoJsonLayer
    if hasattr(data_gdf, "crs") and data_gdf.crs is not None:
        data_gdf = data_gdf.to_crs(epsg=4326)
    elif hasattr(data_gdf, "set_crs"):
        data_gdf = data_gdf.set_crs(epsg=4326)

    tokens = theme_tokens or COLOR_TOKENS
    border_color = hex_to_rgba(tokens.get("border-subtle", "#2a384c"), 255)
    highlight_color = hex_to_rgba(tokens.get("accent-focus", "#38bdf8"), 180)

    # Style resolution for Carto basemaps: resolve to style JSON URLs to bypass Mapbox API key requirement
    CARTO_DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    CARTO_LIGHT_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

    raw_map_style = map_style or tokens.get("map-style", "carto-darkmatter")
    if raw_map_style in ("carto-darkmatter", "dark"):
        resolved_map_style = CARTO_DARK_STYLE
    elif raw_map_style in ("carto-positron", "light"):
        resolved_map_style = CARTO_LIGHT_STYLE
    else:
        resolved_map_style = raw_map_style

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=data_gdf,
        opacity=0.9,
        stroked=True,
        filled=True,
        get_fill_color="fill_color",
        get_line_color=border_color,
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=highlight_color,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        min_zoom=3.0,
        max_zoom=10.0,
        pitch=0,
        bearing=0,
    )

    bg_tip = tokens.get("bg-surface", "#161e2a")
    text_tip = tokens.get("text-primary", "#f1f5f9")
    border_tip = tokens.get("border-subtle", "#2a384c")
    muted_tip = tokens.get("text-muted", "#94a3b8")
    accent_tip = tokens.get("accent-focus", "#38bdf8")

    if is_simulated:
        tooltip_html = f"""<div style="background-color: {bg_tip}; color: {text_tip}; padding: 8px 12px; border: 1px solid {border_tip}; border-radius: 4px; font-family: monospace; font-size: 12px; line-height: 1.4; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
<div style="font-weight: 700; font-size: 13px; color: {accent_tip}; margin-bottom: 2px;">{{KAB_KOTA}}</div>
<div style="color: {muted_tip}; font-size: 11px; margin-bottom: 6px;">{{PROVINSI}}</div>
<div style="display: flex; justify-content: space-between; gap: 12px;"><span>Risk:</span><b>{{formatted_risk}}</b></div>
<div style="display: flex; justify-content: space-between; gap: 12px;"><span>Mitigation:</span><b>{{formatted_mitigation}}</b></div>
<div style="display: flex; justify-content: space-between; gap: 12px;"><span>Baseline Gap:</span><b>{{formatted_gap}}</b></div>
<div style="display: flex; justify-content: space-between; gap: 12px; color: {accent_tip};"><span>Simulated Gap:</span><b>{{formatted_sim_gap}}</b></div>
<div style="margin-top: 6px; padding-top: 4px; border-top: 1px solid {border_tip}; color: {text_tip};">
Category: <b>{{category_label}}</b>
</div>
<div style="color: {accent_tip}; margin-top: 2px;">
Simulated Category: <b>{{sim_category_label}}</b>
</div>
</div>"""
    else:
        tooltip_html = f"""<div style="background-color: {bg_tip}; color: {text_tip}; padding: 8px 12px; border: 1px solid {border_tip}; border-radius: 4px; font-family: monospace; font-size: 12px; line-height: 1.4; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
<div style="font-weight: 700; font-size: 13px; color: {accent_tip}; margin-bottom: 2px;">{{KAB_KOTA}}</div>
<div style="color: {muted_tip}; font-size: 11px; margin-bottom: 6px;">{{PROVINSI}}</div>
<div style="display: flex; justify-content: space-between; gap: 12px;"><span>Risk:</span><b>{{formatted_risk}}</b></div>
<div style="display: flex; justify-content: space-between; gap: 12px;"><span>Mitigation:</span><b>{{formatted_mitigation}}</b></div>
<div style="display: flex; justify-content: space-between; gap: 12px;"><span>Gap Score:</span><b>{{formatted_gap}}</b></div>
<div style="margin-top: 6px; padding-top: 4px; border-top: 1px solid {border_tip}; color: {text_tip};">
Category: <b>{{category_label}}</b>
</div>
</div>"""

    tooltip = {
        "html": tooltip_html,
        "style": {
            "backgroundColor": "transparent",
            "border": "none",
            "padding": "0px",
            "zIndex": "1000",
        },
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=resolved_map_style,
        map_provider="carto",
        tooltip=tooltip,
    )
    deck.tooltip = tooltip
    return deck


def get_contingency_matrix_counts(bivariate_df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """Calculate frequency counts across the 3x3 bivariate matrix."""
    all_categories = [
        "R3-M1", "R3-M2", "R3-M3",
        "R2-M1", "R2-M2", "R2-M3",
        "R1-M1", "R1-M2", "R1-M3",
    ]
    counts = {cat: 0 for cat in all_categories}
    if bivariate_df is not None and len(bivariate_df) > 0:
        if "bivariate_category" in bivariate_df.columns:
            val_counts = bivariate_df["bivariate_category"].value_counts().to_dict()
            for cat in all_categories:
                counts[cat] = int(val_counts.get(cat, 0))
    else:
        # Default national distribution if dataframe is not provided (Hybrid Matrix: absolute mitigation thresholds)
        default_counts = {
            "R3-M1": 148, "R2-M1": 137, "R1-M1": 125,
            "R1-M2": 33,  "R2-M2": 27,  "R3-M2": 21,
            "R1-M3": 14,  "R2-M3": 7,   "R3-M3": 2,
        }
        counts.update(default_counts)
    return counts


def render_viewport_footer_html(
    bivariate_df: Optional[pd.DataFrame] = None,
    gdf: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,
    theme_tokens: Optional[Dict[str, str]] = None,
    is_bright: bool = False,
    lang: str = "en",
) -> str:
    """Generate integrated 2-subcolumn viewport footer markup.

    Sub-column Left (~45%): 3x3 color grid with contingency frequency matrix counts and badges.
    Sub-column Right (~55%): 3 monospace KPI metric pills and multi-hazard progress bars.
    """
    if theme_tokens is not None:
        tokens = theme_tokens
        if not is_bright and tokens.get("bg-primary") == "#f8fafc":
            is_bright = True
    elif is_bright:
        from src.config import LIGHT_COLOR_TOKENS
        tokens = LIGHT_COLOR_TOKENS
    else:
        tokens = COLOR_TOKENS

    bg_surface = "#ffffff" if is_bright else tokens.get("bg-surface", "#161e2a")
    border_subtle = "#cbd5e1" if is_bright else tokens.get("border-subtle", "#2a384c")
    text_primary = "#0f172a" if is_bright else tokens.get("text-primary", "#f1f5f9")
    text_muted = "#64748b" if is_bright else tokens.get("text-muted", "#94a3b8")
    bg_elevated = "#f8fafc" if is_bright else tokens.get("bg-surface-elevated", "#1e293b")
    accent = "#0284c7" if is_bright else tokens.get("accent-focus", "#38bdf8")

    # Contingency matrix counts
    counts = get_contingency_matrix_counts(bivariate_df)
    total_n = sum(counts.values()) if sum(counts.values()) > 0 else 514
    crit_count = counts.get("R3-M1", 0)
    crit_pct = (crit_count / total_n * 100.0) if total_n > 0 else 0.0
    res_count = counts.get("R1-M3", 0)
    res_pct = (res_count / total_n * 100.0) if total_n > 0 else 0.0

    # Grid definition: Rows (R3, R2, R1) x Cols (M1, M2, M3)
    lbl_high = "Tinggi (R3)" if lang == "id" else "High (R3)"
    lbl_med = "Sedang (R2)" if lang == "id" else "Med (R2)"
    lbl_low = "Rendah (R1)" if lang == "id" else "Low (R1)"
    grid_rows = [
        (lbl_high, [("R3-M1", "#d03838", "Critical Gap"), ("R3-M2", "#c47c5d", "High Risk / Med Mit"), ("R3-M3", "#82537e", "High Risk / High Mit")]),
        (lbl_med,  [("R2-M1", "#d88478", "Med Risk / Low Mit"),   ("R2-M2", "#bfa29b", "Med Risk / Med Mit"),   ("R2-M3", "#7a8ba5", "Med Risk / High Mit")]),
        (lbl_low,  [("R1-M1", "#e2dfd2", "Low Risk / Low Mit"),   ("R1-M2", "#9fb8be", "Low Risk / Med Mit"),   ("R1-M3", "#3b82a6", "Resilient Prepared")]),
    ]

    cells_html = ""
    for row_label, cols in grid_rows:
        cells_html += "<div style='display: flex; align-items: center; gap: 4px; margin-bottom: 4px;'>"
        cells_html += f"<span style='width: 58px; font-size: 10px; color: {text_muted}; text-align: right; margin-right: 4px;'>{row_label}</span>"
        for code, color, label in cols:
            cnt = counts.get(code, 0)
            cell_border = f"border: 2px solid {accent};" if code == "R3-M1" else f"border: 1px solid {border_subtle};"
            txt_color = "#0f172a" if (is_bright or code in ["R1-M1", "R1-M2", "R2-M2"]) else "#ffffff"
            cells_html += (
                f"<div title='{code}: {label} ({cnt} units)' style='width: 32px; height: 30px; background-color: {color}; "
                f"{cell_border} border-radius: 2px; display: flex; align-items: center; justify-content: center; "
                f"font-size: 11px; font-weight: 700; color: {txt_color}; cursor: default;'>{cnt}</div>"
            )
        cells_html += "</div>"

    m1_title = "M1: < 30% Fasilitas (Defisit Parah)" if lang == "id" else "M1: < 30% Facility Coverage (Severe Deficit)"
    m2_title = "M2: 30%–60% Fasilitas (Siap Parsial)" if lang == "id" else "M2: 30%–60% Facility Coverage (Partial)"
    m3_title = "M3: > 60% Fasilitas (Kapasitas Unggul)" if lang == "id" else "M3: > 60% Facility Coverage (High Capacity)"
    mit_dir_label = "Kesiapsiagaan Mitigasi →" if lang == "id" else "Mitigation Preparedness →"
    footer_col_labels = (
        f"<div style='display: flex; justify-content: flex-start; margin-left: 66px; gap: 4px; font-size: 10px; color: {text_muted};'>"
        f"<span style='width: 32px; text-align: center;' title='{m1_title}'>M1</span>"
        f"<span style='width: 32px; text-align: center;' title='{m2_title}'>M2</span>"
        f"<span style='width: 32px; text-align: center;' title='{m3_title}'>M3</span>"
        "</div>"
        f"<div style='margin-left: 66px; font-size: 9px; color: {text_muted}; margin-top: 2px;'>{mit_dir_label}</div>"
    )

    # Sub-column Right metrics
    src_df = bivariate_df if bivariate_df is not None and len(bivariate_df) > 0 else gdf
    mean_gap = float(src_df["gap_score"].mean()) if src_df is not None and "gap_score" in src_df.columns else 0.275
    mean_signal = float(src_df["SINYAL_BURUK"].mean()) if src_df is not None and "SINYAL_BURUK" in src_df.columns else 21.4
    mean_gotong = float(src_df["GOTONG_ROYONG"].mean()) if src_df is not None and "GOTONG_ROYONG" in src_df.columns else 84.5
    mean_flood = float(src_df["ADA_BENCANA"].mean()) if src_df is not None and "ADA_BENCANA" in src_df.columns else 39.4
    mean_tsunami = float(src_df["EWS_TSUNAMI"].mean()) if src_df is not None and "EWS_TSUNAMI" in src_df.columns else 1.4
    mean_slope = float(src_df["LERENG"].mean()) if src_df is not None and "LERENG" in src_df.columns else 37.8

    gap_color = ("#dc2626" if is_bright else "#f87171") if mean_gap > 0.3 else (accent if mean_gap > 0 else ("#16a34a" if is_bright else "#34d399"))
    crit_badge_style = (
        "background-color: rgba(220, 38, 38, 0.1); border: 1px solid #dc2626; border-radius: 3px; padding: 3px 6px; font-size: 10px; color: #b91c1c; font-weight: 700;"
        if is_bright
        else "background-color: rgba(208, 56, 56, 0.15); border: 1px solid #d03838; border-radius: 3px; padding: 3px 6px; font-size: 10px; color: #f87171; font-weight: 700;"
    )
    res_badge_style = (
        "background-color: rgba(2, 132, 199, 0.1); border: 1px solid #0284c7; border-radius: 3px; padding: 3px 6px; font-size: 10px; color: #0284c7; font-weight: 700;"
        if is_bright
        else f"background-color: rgba(59, 130, 166, 0.15); border: 1px solid #3b82a6; border-radius: 3px; padding: 3px 6px; font-size: 10px; color: {accent}; font-weight: 700;"
    )
    bar_track_bg = "#e2e8f0" if is_bright else border_subtle

    # Multi-hazard bar width bounds
    flood_bar = min(max(mean_flood, 2.0), 100.0)
    tsunami_bar = min(max(mean_tsunami * 5.0, 2.0), 100.0)  # visually scale low tsunami base
    slope_bar = min(max(mean_slope, 2.0), 100.0)

    title_left = "3×3 Risiko vs. Mitigasi" if lang == "id" else "3×3 Risk vs. Mitigation"
    badge_crit_txt = f"Kesenjangan Kritis (R3-M1): {crit_count} ({crit_pct:.1f}%)" if lang == "id" else f"Critical Gap (R3-M1): {crit_count} ({crit_pct:.1f}%)"
    badge_res_txt = f"Tangguh (R1-M3): {res_count} ({res_pct:.1f}%)" if lang == "id" else f"Resilient (R1-M3): {res_count} ({res_pct:.1f}%)"
    title_right = "Ketahanan Teritorial & Pemicu Bahaya" if lang == "id" else "Territorial Resilience & Hazard Drivers"
    pill_gap = "Rata-rata Gap" if lang == "id" else "Avg Gap Score"
    pill_signal = "Sinyal Buruk" if lang == "id" else "Bad Telecom"
    pill_gotong = "Modal Sosial" if lang == "id" else "Social Capital"
    haz_flood = "Bahaya Banjir / Genangan Air" if lang == "id" else "Floods / Water Hazard"
    haz_tsunami = "Telemetri EWS Tsunami" if lang == "id" else "Tsunami EWS Telemetry"
    haz_slope = "Risiko Longsor / Kemiringan" if lang == "id" else "Landslides / Slopes Risk"

    markup = f"""<div class='idr-viewport-footer' style='display: flex; gap: 16px; background: {bg_surface}; border: 1px solid {border_subtle}; border-radius: 4px; padding: 12px 14px; font-family: monospace;'>
<div class='idr-footer-left' style='flex: 0 0 45%; min-width: 240px;'>
<div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;'>
<span style='font-size: 11px; font-weight: 700; color: {text_primary}; text-transform: uppercase; letter-spacing: 0.5px;'>{title_left}</span>
<span style='font-size: 10px; color: {text_muted};'>N={total_n}</span>
</div>
{cells_html}
{footer_col_labels}
<div style='display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; padding-top: 8px; border-top: 1px solid {border_subtle};'>
<div style='{crit_badge_style}'>{badge_crit_txt}</div>
<div style='{res_badge_style}'>{badge_res_txt}</div>
</div>
</div>
<div class='idr-footer-right' style='flex: 1 1 55%; min-width: 260px; border-left: 1px solid {border_subtle}; padding-left: 14px;'>
<div style='font-size: 11px; font-weight: 700; color: {text_primary}; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>{title_right}</div>
<div style='display: flex; gap: 8px; margin-bottom: 12px;'>
<div style='flex: 1; background-color: {bg_elevated}; border: 1px solid {border_subtle}; border-radius: 3px; padding: 6px 8px; text-align: center;'>
<div style='font-size: 9px; color: {text_muted}; text-transform: uppercase;'>{pill_gap}</div>
<div style='font-size: 14px; font-weight: 700; color: {gap_color};'>{mean_gap:+.3f}</div>
</div>
<div style='flex: 1; background-color: {bg_elevated}; border: 1px solid {border_subtle}; border-radius: 3px; padding: 6px 8px; text-align: center;'>
<div style='font-size: 9px; color: {text_muted}; text-transform: uppercase;'>{pill_signal}</div>
<div style='font-size: 14px; font-weight: 700; color: {text_primary};'>{mean_signal:.1f}%</div>
</div>
<div style='flex: 1; background-color: {bg_elevated}; border: 1px solid {border_subtle}; border-radius: 3px; padding: 6px 8px; text-align: center;'>
<div style='font-size: 9px; color: {text_muted}; text-transform: uppercase;'>{pill_gotong}</div>
<div style='font-size: 14px; font-weight: 700; color: {text_primary};'>{mean_gotong:.1f}%</div>
</div>
</div>
<div style='display: flex; flex-direction: column; gap: 6px;'>
<div>
<div style='display: flex; justify-content: space-between; font-size: 10px; color: {text_muted}; margin-bottom: 2px;'>
<span>{haz_flood}</span>
<b style='color: {text_primary};'>{mean_flood:.1f}%</b>
</div>
<div style='background-color: {bar_track_bg}; height: 6px; border-radius: 2px; overflow: hidden;'>
<div style='width: {flood_bar:.1f}%; background-color: #38bdf8; height: 100%;'></div>
</div>
</div>
<div>
<div style='display: flex; justify-content: space-between; font-size: 10px; color: {text_muted}; margin-bottom: 2px;'>
<span>{haz_tsunami}</span>
<b style='color: {text_primary};'>{mean_tsunami:.1f}%</b>
</div>
<div style='background-color: {bar_track_bg}; height: 6px; border-radius: 2px; overflow: hidden;'>
<div style='width: {tsunami_bar:.1f}%; background-color: #06b6d4; height: 100%;'></div>
</div>
</div>
<div>
<div style='display: flex; justify-content: space-between; font-size: 10px; color: {text_muted}; margin-bottom: 2px;'>
<span>{haz_slope}</span>
<b style='color: {text_primary};'>{mean_slope:.1f}%</b>
</div>
<div style='background-color: {bar_track_bg}; height: 6px; border-radius: 2px; overflow: hidden;'>
<div style='width: {slope_bar:.1f}%; background-color: #f97316; height: 100%;'></div>
</div>
</div>
</div>
</div>
</div>"""
    return markup.strip()


def render_provincial_hotspots_strip_html(
    bivariate_df: Optional[pd.DataFrame] = None,
    gdf: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,
    is_bright: bool = False,
    theme_tokens: Optional[Dict[str, str]] = None,
    top_n: int = 4,
    lang: str = "en",
) -> str:
    """Generate compact horizontal analytical strip showing Top Provincial Priority Hotspots.

    Aggregates Critical Gap (R3-M1) units per province within the currently active corridor.
    """
    if theme_tokens is not None:
        if not is_bright and theme_tokens.get("bg-primary") == "#f8fafc":
            is_bright = True

    df = bivariate_df if bivariate_df is not None and len(bivariate_df) > 0 else gdf
    if df is None or len(df) == 0:
        return ""

    # Ensure is_critical_gap exists
    crit_col = "is_critical_gap"
    if crit_col not in df.columns and "bivariate_category" in df.columns:
        df = df.copy()
        df[crit_col] = df["bivariate_category"] == "R3-M1"

    prov_col = "PROVINSI" if "PROVINSI" in df.columns else None
    if prov_col is None:
        return ""

    # Aggregate by province
    agg = (
        df.groupby(prov_col)
        .agg(
            critical_units=(crit_col, "sum"),
            total_units=("KODE_WIL", "count") if "KODE_WIL" in df.columns else (prov_col, "count"),
        )
        .reset_index()
    )
    top_provs = agg.sort_values(by=["critical_units", "total_units"], ascending=[False, False]).head(top_n)

    container_bg = "#ffffff" if is_bright else "#161e2a"
    border_subtle = "#cbd5e1" if is_bright else "#2a384c"
    card_bg = "#f8fafc" if is_bright else "#0f141c"
    text_primary = "#0f172a" if is_bright else "#f1f5f9"
    text_muted = "#64748b" if is_bright else "#94a3b8"

    badge_crit_style = (
        "background-color: rgba(220, 38, 38, 0.1); border: 1px solid #dc2626; color: #b91c1c; font-weight: 700;"
        if is_bright
        else "background-color: rgba(208, 56, 56, 0.18); border: 1px solid #d03838; color: #f87171; font-weight: 700;"
    )
    badge_norm_style = (
        "background-color: #e2e8f0; border: 1px solid #cbd5e1; color: #334155;"
        if is_bright
        else "background-color: #1e293b; border: 1px solid #2a384c; color: #94a3b8;"
    )

    strip_title = "Titik Prioritas Provinsi Tertinggi" if lang == "id" else "Top Provincial Priority Hotspots"
    strip_sub = "TARGET KONSENTRASI KORIDOR" if lang == "id" else "CORRIDOR CONCENTRATION TARGETS"

    cards_html = []
    for _, r in top_provs.iterrows():
        p_name = str(r[prov_col])
        c_count = int(r["critical_units"])
        t_count = int(r["total_units"])
        unit_lbl = "Wilayah" if lang == "id" else "Units"
        crit_lbl = "Kritis" if lang == "id" else "Critical"
        stable_lbl = "Stabil" if lang == "id" else "Stable"
        if c_count > 0:
            badge_markup = f"<span style='{badge_crit_style} padding: 2px 6px; border-radius: 3px; font-size: 10px;'>{c_count} {unit_lbl} ({crit_lbl})</span>"
        else:
            badge_markup = f"<span style='{badge_norm_style} padding: 2px 6px; border-radius: 3px; font-size: 10px;'>0 {unit_lbl} ({stable_lbl})</span>"

        cards_html.append(
            f"<div style='flex: 1 1 22%; min-width: 130px; background: {card_bg}; border: 1px solid {border_subtle}; border-radius: 3px; padding: 6px 10px; display: flex; justify-content: space-between; align-items: center;'>"
            f"<div><div style='font-size: 11px; font-weight: 700; color: {text_primary};'>{p_name}</div><div style='font-size: 9px; color: {text_muted};'>Total: {t_count} {unit_lbl}</div></div>"
            f"<div>{badge_markup}</div>"
            f"</div>"
        )

    cards_joined = "".join(cards_html)

    markup = f"""<div style='background: {container_bg}; border: 1px solid {border_subtle}; border-radius: 4px; padding: 10px 12px; margin-top: 10px; font-family: monospace;'>
<div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;'>
<span style='font-size: 10px; font-weight: 700; color: {text_primary}; text-transform: uppercase; letter-spacing: 0.5px;'>{strip_title}</span>
<span style='font-size: 9px; color: {text_muted};'>{strip_sub}</span>
</div>
<div style='display: flex; flex-wrap: wrap; gap: 8px;'>
{cards_joined}
</div>
</div>"""
    return markup.strip()


def render_bivariate_legend_html(
    bivariate_df: Optional[pd.DataFrame] = None,
    gdf: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None,
    theme_tokens: Optional[Dict[str, str]] = None,
) -> str:
    """Generate crisp, high-density HTML markup for the 3x3 Bivariate Matrix legend."""
    return render_viewport_footer_html(
        bivariate_df=bivariate_df,
        gdf=gdf,
        theme_tokens=theme_tokens,
    )


def render_lisa_legend_html(theme_tokens: Optional[Dict[str, str]] = None) -> str:
    """Generate crisp, high-density HTML markup for LISA Spatial Clusters legend."""
    tokens = theme_tokens or COLOR_TOKENS
    bg_surface = tokens.get("bg-surface", "#161e2a")
    border_subtle = tokens.get("border-subtle", "#2a384c")
    text_primary = tokens.get("text-primary", "#f1f5f9")
    text_muted = tokens.get("text-muted", "#94a3b8")

    categories = [
        ("High-High (Hotspot)", "#ef4444", "High risk clustered with high risk"),
        ("Low-Low (Coldspot)", "#0284c7", "Low risk clustered with low risk"),
        ("High-Low (Outlier)", "#f97316", "High risk surrounded by low risk"),
        ("Low-High (Outlier)", "#06b6d4", "Low risk surrounded by high risk"),
        ("Not Significant", "#334155", "p > alpha threshold"),
    ]

    rows_html = ""
    for label, color, desc in categories:
        rows_html += (
            f"<div style='display: flex; align-items: center; gap: 8px; margin-bottom: 6px;'>"
            f"<div style='width: 14px; height: 14px; background-color: {color}; border: 1px solid {border_subtle}; border-radius: 2px; flex-shrink: 0;'></div>"
            f"<div style='font-size: 11px;'>"
            f"<b style='color: {text_primary};'>{label}</b>"
            f"<span style='color: {text_muted}; font-size: 10px; margin-left: 4px;'>— {desc}</span>"
            f"</div>"
            f"</div>"
        )

    markup = f"""<div style='background-color: {bg_surface}; border: 1px solid {border_subtle}; border-radius: 4px; padding: 10px 14px; font-family: monospace;'>
<div style='font-size: 11px; font-weight: 700; color: {text_primary}; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>
LISA Spatial Clusters
</div>
{rows_html}
</div>"""
    return markup.strip()


def render_balance_legend_html(
    theme_tokens: Optional[Dict[str, str]] = None,
    is_bright: bool = False,
    lang: str = "en",
) -> str:
    """Generate crisp, high-density HTML markup for Readiness Balance legend."""
    tokens = theme_tokens or COLOR_TOKENS
    bg_surface = "#ffffff" if is_bright else tokens.get("bg-surface", "#161e2a")
    border_subtle = "#cbd5e1" if is_bright else tokens.get("border-subtle", "#2a384c")
    text_primary = "#0f172a" if is_bright else tokens.get("text-primary", "#f1f5f9")
    text_muted = "#64748b" if is_bright else tokens.get("text-muted", "#94a3b8")

    title = "Neraca Kesiapsiagaan (Mitigasi - Risiko)" if lang == "id" else "Readiness Balance (Mitigation - Risk)"

    categories = [
        ("< -0.40", "#b91c1c", "Defisit Ekstrem" if lang == "id" else "Severe Deficit"),
        ("-0.40 s/d -0.10", "#f59e0b", "Defisit Moderat" if lang == "id" else "Moderate Deficit"),
        ("-0.10 s/d +0.10", "#94a3b8", "Seimbang" if lang == "id" else "Balanced / Neutral"),
        ("> +0.10", "#0284c7", "Surplus Mitigasi" if lang == "id" else "Mitigation Surplus"),
    ]

    rows_html = ""
    for rng, color, desc in categories:
        rows_html += (
            f"<div style='display: flex; align-items: center; gap: 8px; margin-bottom: 6px;'>"
            f"<div style='width: 14px; height: 14px; background-color: {color}; border: 1px solid {border_subtle}; border-radius: 2px; flex-shrink: 0;'></div>"
            f"<div style='font-size: 11px;'>"
            f"<b style='color: {text_primary};'>{rng}</b>"
            f"<span style='color: {text_muted}; font-size: 10px; margin-left: 4px;'>— {desc}</span>"
            f"</div>"
            f"</div>"
        )

    markup = f"""<div style='background-color: {bg_surface}; border: 1px solid {border_subtle}; border-radius: 4px; padding: 10px 14px; font-family: monospace;'>
<div style='font-size: 11px; font-weight: 700; color: {text_primary}; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>
{title}
</div>
{rows_html}
</div>"""
    return markup.strip()
