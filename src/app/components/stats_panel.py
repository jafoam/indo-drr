"""Statistical analytics panel component displaying Moran scatterplot, metrics, and critical gap tables."""
from typing import Any, Dict, List, Optional, Union
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import libpysal
from libpysal.weights import W

from src.config import COLOR_TOKENS, LISA_PALETTE
from src.analytics.moran import MoranResult, compute_morans_i, compute_lisa
from src.app.components.choropleth import get_balance_rgba


def create_moran_scatterplot(
    gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
    variable: str = "IRBI_2024",
    weights: Optional[W] = None,
    lisa_clusters: Optional[pd.Series] = None,
    corridor_name: Optional[str] = None,
    theme_tokens: Optional[Dict[str, str]] = None,
    is_bright: bool = False,
) -> go.Figure:
    """Generate high-density Moran scatterplot (standardized variable vs spatial lag).

    Args:
        gdf: GeoDataFrame containing target column and geometries.
        variable: Attribute column name to analyze.
        weights: PySAL spatial weights matrix.
        lisa_clusters: Optional Series containing LISA cluster labels for point coloring.
        corridor_name: Optional name of the regional corridor for localized context.
        theme_tokens: Optional dictionary of active theme design tokens.
        is_bright: Flag indicating whether Light Theme is active.

    Returns:
        Plotly Figure styled with dark or light matte anti-slop tokens.
    """
    if theme_tokens is not None:
        tokens = theme_tokens
        if not is_bright and tokens.get("bg-surface") == "#ffffff":
            is_bright = True
    elif is_bright:
        from src.config import LIGHT_COLOR_TOKENS
        tokens = LIGHT_COLOR_TOKENS
    else:
        tokens = COLOR_TOKENS
    if variable not in gdf.columns:
        raise KeyError(f"Variable '{variable}' not found in dataframe.")

    if weights is None:
        from src.analytics.weights import build_spatial_weights
        weights = build_spatial_weights(gdf, k=5)

    y = gdf[variable].to_numpy(dtype=float)

    # Standardize variable z = (y - mean) / std
    y_std = np.nanstd(y)
    if y_std == 0.0 or np.isnan(y_std):
        z = np.zeros_like(y)
    else:
        z = (y - np.nanmean(y)) / y_std

    # Compute spatial lag Wz
    w_z = libpysal.weights.lag_spatial(weights, z)

    # OLS regression slope equals Global Moran's I
    denom = np.sum(z ** 2)
    slope = float(np.sum(z * w_z) / denom) if denom != 0 else 0.0

    # Determine point colors and cluster hover info
    point_colors: List[str] = []
    cluster_labels: List[str] = []

    if lisa_clusters is not None and len(lisa_clusters) == len(gdf):
        for label in lisa_clusters:
            cluster_labels.append(str(label))
            color = LISA_PALETTE.get(label) or LISA_PALETTE.get(str(label).lower().replace(" ", "-"), "#334155")
            point_colors.append(color)
    else:
        point_colors = [tokens.get("accent-focus", "#38bdf8")] * len(gdf)
        cluster_labels = [""] * len(gdf)

    # Regency / Province names for hover
    kab_names = gdf["KAB_KOTA"].tolist() if "KAB_KOTA" in gdf.columns else [f"Unit {i}" for i in range(len(gdf))]
    prov_names = gdf["PROVINSI"].tolist() if "PROVINSI" in gdf.columns else [""] * len(gdf)

    hover_texts = [
        f"<b>{kab}</b> ({prov})<br>z: {zi:+.2f}<br>Lag Wz: {wi:+.2f}" + (f"<br>LISA: {c}" if c else "")
        for kab, prov, zi, wi, c in zip(kab_names, prov_names, z, w_z, cluster_labels)
    ]

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=z,
            y=w_z,
            mode="markers",
            marker=dict(
                size=7,
                color=point_colors,
                opacity=0.85,
                line=dict(width=0.8, color=tokens.get("bg-surface", "#0f141c")),
            ),
            text=hover_texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Fitted OLS regression line: slope = Moran's I
    x_range = np.linspace(float(np.min(z)), float(np.max(z)), 50)
    legend_name = f"{corridor_name} Moran's I = {slope:.3f}" if corridor_name else f"Moran's I = {slope:.3f}"
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=x_range * slope,
            mode="lines",
            line=dict(color=tokens.get("accent-focus", "#38bdf8"), width=2, dash="dash"),
            name=legend_name,
            hoverinfo="name",
            showlegend=True,
        )
    )

    # Calculate symmetrical boundaries for grid axes
    max_val = max(abs(float(np.min(z))), abs(float(np.max(z))), abs(float(np.min(w_z))), abs(float(np.max(w_z))), 2.5) * 1.05

    title_text = f"Moran's I Scatterplot — {corridor_name} Corridor" if corridor_name else "Moran's I Scatterplot (National)"

    axis_font_color = "#0f172a" if is_bright else tokens["text-primary"]
    axis_grid_color = "#e2e8f0" if is_bright else tokens["border-subtle"]
    axis_zero_color = "#cbd5e1" if is_bright else tokens["text-muted"]
    plot_bg = "#ffffff" if is_bright else tokens["bg-surface"]
    paper_bg = "#ffffff" if is_bright else tokens["bg-surface"]
    title_color = "#0f172a" if is_bright else tokens["text-primary"]

    fig.update_layout(
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        template=tokens.get("plotly-template", "plotly_white" if is_bright else "plotly_dark"),
        font=dict(color=axis_font_color, family="monospace", size=11),
        title=dict(
            text=title_text,
            font=dict(size=11, color=title_color, family="monospace"),
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(
            title=dict(
                text=f"Standardized {variable} (z)",
                font=dict(color=axis_font_color, family="monospace", size=11),
            ),
            tickfont=dict(color=axis_font_color, family="monospace", size=10),
            gridcolor=axis_grid_color,
            linecolor=axis_font_color,
            zerolinecolor=axis_zero_color,
            zerolinewidth=1,
            range=[-max_val, max_val],
        ),
        yaxis=dict(
            title=dict(
                text="Spatial Lag (Wz)",
                font=dict(color=axis_font_color, family="monospace", size=11),
            ),
            tickfont=dict(color=axis_font_color, family="monospace", size=10),
            gridcolor=axis_grid_color,
            linecolor=axis_font_color,
            zerolinecolor=axis_zero_color,
            zerolinewidth=1,
            range=[-max_val, max_val],
        ),
        margin=dict(l=40, r=20, t=36, b=40),
        legend=dict(
            x=0.02,
            y=0.88,
            bgcolor="#f1f5f9" if is_bright else tokens.get("bg-surface-elevated", "rgba(22, 30, 42, 0.8)"),
            bordercolor="#cbd5e1" if is_bright else tokens["border-subtle"],
            borderwidth=1,
            font=dict(size=10, color=axis_font_color),
        ),
        height=340,
    )

    return fig


def render_moran_metrics_html(
    moran_result: MoranResult,
    theme_tokens: Optional[Dict[str, str]] = None,
    is_bright: bool = False,
    lang: str = "en",
) -> str:
    """Generate high-density HTML metric block for Global Moran's I."""
    if theme_tokens is not None:
        tokens = theme_tokens
        if not is_bright and tokens.get("bg-surface") == "#ffffff":
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
    accent = "#0284c7" if is_bright else tokens.get("accent-focus", "#38bdf8")
    bg_elevated = "#f1f5f9" if is_bright else tokens.get("bg-surface-elevated", "#0f141c")

    header_label = "Autokorelasi Global" if lang == "id" else "Global Autocorrelation"
    if moran_result.p_value <= 0.05:
        significance_label = "Signifikan Secara Statistik (p ≤ 0.05)" if lang == "id" else "Statistically Significant (p ≤ 0.05)"
    else:
        significance_label = "Tidak Signifikan (p > 0.05)" if lang == "id" else "Not Significant (p > 0.05)"
    sig_color = accent if moran_result.p_value <= 0.05 else text_muted
    mc_label = f"Permutasi Monte Carlo: N = {moran_result.simulations}" if lang == "id" else f"Monte Carlo permutations: N = {moran_result.simulations}"

    markup = f"""<div style='background: {bg_surface}; border: 1px solid {border_subtle}; border-radius: 4px; padding: 12px; font-family: monospace; margin-bottom: 12px; width: 100%; box-sizing: border-box;'>
<div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;'>
<span style='font-size: 11px; text-transform: uppercase; color: {text_muted}; letter-spacing: 0.5px;'>{header_label}</span>
<span style='font-size: 10px; color: {sig_color}; font-weight: 600;'>{significance_label}</span>
</div>
<div style='display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; text-align: left; width: 100%; box-sizing: border-box;'>
<div style='background: {bg_elevated}; padding: 6px 4px; border: 1px solid {border_subtle}; border-radius: 2px; min-width: 0; box-sizing: border-box; text-align: center;'>
<div style='font-size: 9px; color: {text_muted}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>Moran's I</div>
<div style='font-size: 13px; font-weight: 700; color: {text_primary}; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{moran_result.moran_i:+.4f}</div>
</div>
<div style='background: {bg_elevated}; padding: 6px 4px; border: 1px solid {border_subtle}; border-radius: 2px; min-width: 0; box-sizing: border-box; text-align: center;'>
<div style='font-size: 9px; color: {text_muted}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>Expected I</div>
<div style='font-size: 13px; font-weight: 700; color: {text_muted}; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{moran_result.expected_i:+.4f}</div>
</div>
<div style='background: {bg_elevated}; padding: 6px 4px; border: 1px solid {border_subtle}; border-radius: 2px; min-width: 0; box-sizing: border-box; text-align: center;'>
<div style='font-size: 9px; color: {text_muted}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>z-score</div>
<div style='font-size: 13px; font-weight: 700; color: {text_primary}; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{moran_result.z_score:+.2f}</div>
</div>
<div style='background: {bg_elevated}; padding: 6px 4px; border: 1px solid {border_subtle}; border-radius: 2px; min-width: 0; box-sizing: border-box; text-align: center;'>
<div style='font-size: 9px; color: {text_muted}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>p-value (sim)</div>
<div style='font-size: 13px; font-weight: 700; color: {accent}; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{moran_result.p_value:.3f}</div>
</div>
</div>
<div style='margin-top: 8px; font-size: 10px; color: {text_muted}; text-align: right;'>
{mc_label}
</div>
</div>"""
    return markup.strip()


def render_top_critical_gap_table_html(
    bivariate_df: pd.DataFrame,
    n: int = 10,
    is_bright: bool = False,
    lang: str = "en",
) -> str:
    """Render adaptive theme-aware HTML table for Top N Critical Gap Units.

    Styling tokens:
    - Light Theme: container `#ffffff`, table `#ffffff`, header `#f1f5f9`, text `#0f172a`, borders `#e2e8f0`.
    - Dark Theme: container `#1a2332`, table `#1a2332`, header `#0f172a`, text `#f1f5f9`, borders `#2a384c`.
    """
    table_df = get_top_critical_gap_table(bivariate_df, n=n, lang=lang) if "gap_score" in bivariate_df.columns else bivariate_df.head(n)

    container_bg = "#ffffff" if is_bright else "#1a2332"
    header_bg = "#f1f5f9" if is_bright else "#0f172a"
    text_color = "#0f172a" if is_bright else "#f1f5f9"
    border_color = "#cbd5e1" if is_bright else "#2a384c"
    row_border = "#e2e8f0" if is_bright else "#2a384c"

    lines = [
        f'<div style="background: {container_bg}; border: 1px solid {border_color}; border-radius: 4px; overflow-x: auto; width: 100%; box-sizing: border-box; font-family: monospace; font-size: 10.5px;">',
        f'<table style="width: 100%; border-collapse: collapse; text-align: left; background: {container_bg}; color: {text_color}; box-sizing: border-box;">',
        '<thead>',
        f'<tr style="background: {header_bg}; border-bottom: 1px solid {row_border};">',
    ]
    for col in table_df.columns:
        lines.append(f'<th style="padding: 5px 6px; font-weight: 600; color: {text_color}; border-bottom: 1px solid {row_border}; white-space: nowrap;">{col}</th>')
    lines.append('</tr>')
    lines.append('</thead>')
    lines.append('<tbody>')

    for _, row in table_df.iterrows():
        lines.append(f'<tr style="border-bottom: 1px solid {row_border};">')
        for col in table_df.columns:
            val = str(row[col])
            lines.append(f'<td style="padding: 4px 6px; color: {text_color}; white-space: nowrap;">{val}</td>')
        lines.append('</tr>')

    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')
    return "\n".join(line.strip() for line in lines)


def get_top_critical_gap_table(bivariate_df: pd.DataFrame, n: int = 10, lang: str = "en") -> pd.DataFrame:
    """Retrieve top N administrative units with highest positive Gap Score (Risk exceeds Mitigation)."""
    cols = ["KAB_KOTA", "PROVINSI", "risk_score", "mitigation_score", "gap_score", "bivariate_label"]
    available_cols = [c for c in cols if c in bivariate_df.columns]

    sorted_df = bivariate_df.sort_values(by="gap_score", ascending=False).head(n)[available_cols].copy()

    # Format numeric values
    if "risk_score" in sorted_df.columns:
        sorted_df["risk_score"] = sorted_df["risk_score"].round(3)
    if "mitigation_score" in sorted_df.columns:
        sorted_df["mitigation_score"] = sorted_df["mitigation_score"].round(3)
    if "gap_score" in sorted_df.columns:
        sorted_df["gap_score"] = sorted_df["gap_score"].apply(lambda v: f"{v:+.3f}")

    col_map = {
        "KAB_KOTA": "Kabupaten / Kota" if lang == "id" else "Regency / City",
        "PROVINSI": "Provinsi" if lang == "id" else "Province",
        "risk_score": "Risiko (R)" if lang == "id" else "Risk (R)",
        "mitigation_score": "Mitigasi (M)" if lang == "id" else "Mitigation (M)",
        "gap_score": "Kesenjangan (R - M)" if lang == "id" else "Gap (R - M)",
        "bivariate_label": "Kategori" if lang == "id" else "Category",
    }
    sorted_df.columns = [col_map.get(c, c) for c in sorted_df.columns]
    return sorted_df


def create_parity_quadrant_chart(
    df: pd.DataFrame,
    target_kode: Optional[Union[int, str]] = None,
    theme_tokens: Optional[Dict[str, str]] = None,
    is_bright: bool = False,
    lang: str = "en",
) -> go.Figure:
    """Generate 4-Quadrant Parity Diagram (Mitigation M vs Risk R).

    Args:
        df: DataFrame containing risk_score, mitigation_score, and optional readiness_balance.
        target_kode: Optional BPS KODE_WIL to highlight with high-contrast target ring.
        theme_tokens: Active design tokens dictionary.
        is_bright: Whether Light Theme is active.
        lang: Language code ('en' or 'id').

    Returns:
        Plotly Figure object styled for active theme.
    """
    if theme_tokens is not None:
        tokens = theme_tokens
        if not is_bright and tokens.get("bg-surface") == "#ffffff":
            is_bright = True
    elif is_bright:
        from src.config import LIGHT_COLOR_TOKENS
        tokens = LIGHT_COLOR_TOKENS
    else:
        tokens = COLOR_TOKENS

    if is_bright:
        paper_bg = "#ffffff"
        plot_bg = "#f8fafc"
        text_color = "#0f172a"
        muted_color = "#64748b"
        grid_color = "#e2e8f0"
        line_color = "#cbd5e1"
    else:
        paper_bg = tokens.get("bg-surface", "#161e2a")
        plot_bg = tokens.get("bg-surface-elevated", "#0f141c")
        text_color = tokens.get("text-primary", "#f1f5f9")
        muted_color = tokens.get("text-muted", "#94a3b8")
        grid_color = tokens.get("border-subtle", "#2a384c")
        line_color = tokens.get("border-subtle", "#2a384c")

    r_scores = df["risk_score"].to_numpy(dtype=float)
    m_scores = df["mitigation_score"].to_numpy(dtype=float)
    if "readiness_balance" in df.columns:
        balances = df["readiness_balance"].to_numpy(dtype=float)
    else:
        balances = np.clip(m_scores - r_scores, -1.0, 1.0)

    if "adequacy_ratio" in df.columns:
        adequacies = df["adequacy_ratio"].to_numpy(dtype=float)
    else:
        adequacies = np.where(r_scores > 0, np.clip((m_scores / r_scores) * 100.0, 0.0, 200.0), 100.0)

    kab_names = df["KAB_KOTA"].tolist() if "KAB_KOTA" in df.columns else [f"Unit {i}" for i in range(len(df))]
    prov_names = df["PROVINSI"].tolist() if "PROVINSI" in df.columns else [""] * len(df)

    point_colors = []
    for bal in balances:
        rgba = get_balance_rgba(bal)
        point_colors.append(f"rgba({rgba[0]},{rgba[1]},{rgba[2]},0.85)")

    hover_texts = [
        f"<b>{kab}</b> ({prov})<br>"
        f"Risk (R): {r:.3f}<br>"
        f"Mitigation (M): {m:.3f}<br>"
        f"Balance (M-R): {b:+.3f}<br>"
        f"Adequacy: {ad:.1f}%"
        for kab, prov, r, m, b, ad in zip(kab_names, prov_names, r_scores, m_scores, balances, adequacies)
    ]

    fig = go.Figure()

    # 45-degree dashed reference parity line (M = R)
    parity_line_label = "Garis Paritas (M=R)" if lang == "id" else "Parity Line (M=R)"
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
            name=parity_line_label,
            hoverinfo="skip",
        )
    )

    # Scatter points of districts
    districts_label = "Kabupaten/Kota" if lang == "id" else "Districts"
    fig.add_trace(
        go.Scatter(
            x=r_scores,
            y=m_scores,
            mode="markers",
            marker=dict(
                size=7,
                color=point_colors,
                line=dict(width=0.5, color=grid_color),
            ),
            text=hover_texts,
            hoverinfo="text",
            name=districts_label,
            showlegend=False,
        )
    )

    # High-contrast target highlight ring if target_kode is selected
    if target_kode is not None and "KODE_WIL" in df.columns:
        match_mask = df["KODE_WIL"].astype(int) == int(target_kode)
        if match_mask.any():
            target_row = df[match_mask].iloc[0]
            tr_r = float(target_row["risk_score"])
            tr_m = float(target_row["mitigation_score"])
            tr_name = str(target_row.get("KAB_KOTA", f"Unit {target_kode}"))
            tr_bal = float(target_row.get("readiness_balance", tr_m - tr_r))
            tr_ad = float(target_row.get("adequacy_ratio", (tr_m / tr_r * 100.0) if tr_r > 0 else 100.0))

            fig.add_trace(
                go.Scatter(
                    x=[tr_r],
                    y=[tr_m],
                    mode="markers+text",
                    marker=dict(
                        size=14,
                        color="rgba(0,0,0,0)",
                        line=dict(width=3, color="#ef4444" if is_bright else "#f87171"),
                        symbol="circle-open",
                    ),
                    text=[f"  <b>{tr_name}</b>"],
                    textposition="top right",
                    textfont=dict(color=text_color, size=11, family="monospace"),
                    hovertext=[f"<b>TARGET: {tr_name}</b><br>R: {tr_r:.3f} | M: {tr_m:.3f}<br>Balance: {tr_bal:+.3f} ({tr_ad:.1f}%)"],
                    hoverinfo="text",
                    name=f"Target: {tr_name}",
                    showlegend=True,
                )
            )

    # Subtle zone annotations: Top-Left "Surplus Zone (M > R)", Bottom-Right "Critical Deficit Zone (M < R)"
    surplus_zone_txt = "Zona Surplus (M > R)" if lang == "id" else "Surplus Zone (M > R)"
    deficit_zone_txt = "Zona Defisit Kritis (M < R)" if lang == "id" else "Critical Deficit Zone (M < R)"

    fig.add_annotation(
        x=0.18,
        y=0.85,
        text=surplus_zone_txt,
        showarrow=False,
        font=dict(size=10.5, color="#0284c7" if is_bright else "#38bdf8", family="monospace"),
        bgcolor="rgba(2, 132, 199, 0.08)" if is_bright else "rgba(56, 189, 248, 0.08)",
        bordercolor="#0284c7" if is_bright else "#38bdf8",
        borderwidth=1,
        borderpad=4,
        opacity=0.85,
    )
    fig.add_annotation(
        x=0.78,
        y=0.15,
        text=deficit_zone_txt,
        showarrow=False,
        font=dict(size=10.5, color="#b91c1c" if is_bright else "#f87171", family="monospace"),
        bgcolor="rgba(185, 28, 28, 0.08)" if is_bright else "rgba(248, 113, 113, 0.08)",
        bordercolor="#b91c1c" if is_bright else "#f87171",
        borderwidth=1,
        borderpad=4,
        opacity=0.85,
    )

    x_title = "Skor Risiko Bencana (R)" if lang == "id" else "Hazard Risk Score (R)"
    y_title = "Skor Kesiapsiagaan Mitigasi (M)" if lang == "id" else "Mitigation Score (M)"

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        xaxis=dict(
            title=dict(text=x_title, font=dict(color=text_color, family="monospace", size=11)),
            tickfont=dict(color=muted_color, family="monospace", size=10),
            gridcolor=grid_color,
            linecolor=line_color,
            range=[-0.02, 1.02],
            dtick=0.2,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color=text_color, family="monospace", size=11)),
            tickfont=dict(color=muted_color, family="monospace", size=10),
            gridcolor=grid_color,
            linecolor=line_color,
            range=[-0.02, 1.02],
            dtick=0.2,
            zeroline=False,
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="#ffffff" if is_bright else "rgba(22, 30, 42, 0.85)",
            bordercolor=line_color,
            borderwidth=1,
            font=dict(size=10, color=text_color),
        ),
        height=340,
    )
    return fig
