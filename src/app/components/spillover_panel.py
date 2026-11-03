"""Spatial Spillover Analytics Panel Components.

Provides dark matte Plotly horizontal decomposition visualization, high-density
0-indent metric cards, and effects decomposition tables for cross-boundary impacts.
"""
from typing import List, Optional, Union
import pandas as pd
import plotly.graph_objects as go

from src.config import COLOR_TOKENS
from src.analytics.spillover import SpilloverResult


def render_spillover_metric_card_html(
    result: SpilloverResult,
    theme: Optional[str] = None,
    is_bright: Optional[bool] = None,
    lang: str = "en",
) -> str:
    """Generate high-density 0-indent HTML metric card for spatial spillover econometrics.

    Strict 0-indent markup complying with anti-slop guidelines:
    - Starts with '<div'
    - Zero lines with 4+ spaces or tabs
    - Displays model type, rho parameter, spatial multiplier, and spillover ratio.
    - Adapts cleanly between Dark Theme and Light Theme.

    Args:
        result: SpilloverResult from spatial lag / SLX model estimation.
        theme: Optional visual theme string ("Dark Theme" or "Light Theme").
        is_bright: Explicit flag indicating whether Light Theme is active.
        lang: Language code ("en" or "id").

    Returns:
        Flush 0-indent HTML string.
    """
    if is_bright is None:
        is_bright = (theme == "Light Theme")

    model_name = "Spatial Lag (SAR)" if result.model_type.lower() == "lag" else "Spatial Lag of X (SLX)"

    if result.rho is not None:
        rho_val = f"{result.rho:+.4f}"
        try:
            multiplier = 1.0 / (1.0 - result.rho)
            multiplier_val = f"{multiplier:.2f}x"
        except (ZeroDivisionError, OverflowError):
            multiplier_val = "N/A"
    else:
        rho_val = "None (SLX)" if lang == "en" else "Tidak Ada (SLX)"
        multiplier_val = "1.00x"

    avg_spillover = (
        sum(e.spillover_ratio for e in result.effects) / len(result.effects)
        if result.effects
        else 0.0
    )
    spillover_val = f"{avg_spillover * 100:.1f}%"

    if is_bright:
        card_bg = "#ffffff"
        card_border = "#cbd5e1"
        text_pri = "#0f172a"
        text_mut = "#64748b"
        accent_color = "#0284c7"
        grid_bg = "#f8fafc"
    else:
        card_bg = "#161e2a"
        card_border = "#2a384c"
        text_pri = "#f1f5f9"
        text_mut = "#94a3b8"
        accent_color = "#38bdf8"
        grid_bg = "#0f141c"

    header_title = f"Dekomposisi Ekonometrika Spasial ({model_name})" if lang == "id" else f"Spatial Econometric Decomposition ({model_name})"
    lbl_model = "TIPE MODEL" if lang == "id" else "MODEL TYPE"
    lbl_rho = "SPATIAL RHO (ρ)"
    lbl_mult = "PENGGANDA SPASIAL" if lang == "id" else "SPATIAL MULTIPLIER"
    lbl_ratio = "RASIO LIMPAHAN" if lang == "id" else "SPILLOVER RATIO"

    lines = [
        f'<div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 4px; padding: 12px; font-family: monospace; margin-bottom: 12px;">',
        '<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">',
        f'<span style="font-size: 11px; text-transform: uppercase; color: {text_mut}; letter-spacing: 0.5px;">{header_title}</span>',
        f'<span style="font-size: 10px; color: {accent_color}; font-weight: 600;">N = {result.n_obs} | Pseudo-R²: {result.r2:.3f}</span>',
        '</div>',
        '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; text-align: left;">',
        f'<div style="background: {grid_bg}; padding: 8px 10px; border: 1px solid {card_border}; border-radius: 2px;">',
        f'<div style="font-size: 10px; color: {text_mut};">{lbl_model}</div>',
        f'<div style="font-size: 14px; font-weight: 700; color: {text_pri}; margin-top: 2px;">{result.model_type.upper()}</div>',
        '</div>',
        f'<div style="background: {grid_bg}; padding: 8px 10px; border: 1px solid {card_border}; border-radius: 2px;">',
        f'<div style="font-size: 10px; color: {text_mut};">{lbl_rho}</div>',
        f'<div style="font-size: 14px; font-weight: 700; color: {text_pri}; margin-top: 2px;">{rho_val}</div>',
        '</div>',
        f'<div style="background: {grid_bg}; padding: 8px 10px; border: 1px solid {card_border}; border-radius: 2px;">',
        f'<div style="font-size: 10px; color: {text_mut};">{lbl_mult}</div>',
        f'<div style="font-size: 14px; font-weight: 700; color: {accent_color}; margin-top: 2px;">{multiplier_val}</div>',
        '</div>',
        f'<div style="background: {grid_bg}; padding: 8px 10px; border: 1px solid {card_border}; border-radius: 2px;">',
        f'<div style="font-size: 10px; color: {text_mut};">{lbl_ratio}</div>',
        f'<div style="font-size: 14px; font-weight: 700; color: #f59e0b; margin-top: 2px;">{spillover_val}</div>',
        '</div>',
        '</div>',
        '</div>',
    ]
    return "\n".join(lines)


def render_spillover_waterfall_chart(
    result: SpilloverResult,
    theme: Optional[str] = None,
    is_bright: Optional[bool] = None,
) -> go.Figure:
    """Render horizontal grouped bar chart decomposing Direct vs Indirect (Spillover) effects.

    Adapts cleanly between Dark Theme and Light Theme.

    Args:
        result: SpilloverResult containing decomposed effects.
        theme: Optional visual theme string ("Dark Theme" or "Light Theme").
        is_bright: Explicit flag indicating whether Light Theme is active.

    Returns:
        Plotly Figure configured with dark matte or light carto tokens.
    """
    if is_bright is None:
        is_bright = (theme == "Light Theme")

    var_names = [e.variable for e in result.effects]
    direct_vals = [e.direct_effect for e in result.effects]
    indirect_vals = [e.indirect_effect for e in result.effects]

    if is_bright:
        paper_bg = "#ffffff"
        plot_bg = "#f8fafc"
        text_pri = "#0f172a"
        text_mut = "#64748b"
        border_col = "#cbd5e1"
        direct_color = "#0284c7"
        indirect_color = "#82537e"
        bar_edge = "#ffffff"
        legend_bg = "rgba(255, 255, 255, 0.85)"
    else:
        paper_bg = COLOR_TOKENS["bg-surface"]
        plot_bg = COLOR_TOKENS["bg-primary"]
        text_pri = COLOR_TOKENS["text-primary"]
        text_mut = COLOR_TOKENS["text-muted"]
        border_col = COLOR_TOKENS["border-subtle"]
        direct_color = "#38bdf8"
        indirect_color = "#82537e"
        bar_edge = "#161e2a"
        legend_bg = "rgba(22, 30, 42, 0.8)"

    fig = go.Figure()

    # Trace 1: Direct Effect (Local)
    fig.add_trace(
        go.Bar(
            y=var_names,
            x=direct_vals,
            name="Direct Effect (Local)",
            orientation="h",
            marker=dict(
                color=direct_color,
                line=dict(color=bar_edge, width=1),
            ),
            hovertemplate="<b>%{y}</b> (Direct)<br>Impact: %{x:+.4f}<extra></extra>",
        )
    )

    # Trace 2: Indirect Effect (Cross-Boundary Spillover)
    fig.add_trace(
        go.Bar(
            y=var_names,
            x=indirect_vals,
            name="Indirect (Spillover)",
            orientation="h",
            marker=dict(
                color=indirect_color,
                line=dict(color=bar_edge, width=1),
            ),
            hovertemplate="<b>%{y}</b> (Spillover)<br>Impact: %{x:+.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_pri, family="monospace", size=11),
        title=dict(
            text="Spatial Spillover Decomposition (Direct vs Indirect)",
            font=dict(size=11, color=text_pri, family="monospace"),
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(
            title="Impact Magnitude (dE[Y]/dX)",
            gridcolor=border_col,
            zerolinecolor=text_mut,
            zerolinewidth=1,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            gridcolor=border_col,
            tickfont=dict(color=text_pri, size=10, family="monospace"),
        ),
        margin=dict(l=40, r=20, t=36, b=40),
        legend=dict(
            x=0.02,
            y=0.92,
            bgcolor=legend_bg,
            bordercolor=border_col,
            borderwidth=1,
            font=dict(size=10, color=text_pri),
        ),
        height=320,
    )

    return fig


def render_spillover_effects_table(result: SpilloverResult, lang: str = "en") -> pd.DataFrame:
    """Format spatial spillover decomposition effects into a presentation DataFrame.

    Args:
        result: SpilloverResult containing decomposed effects.
        lang: Language code ("en" or "id").

    Returns:
        Formatted pandas DataFrame with Direct, Indirect, Total effects and p-values.
    """
    col_var = "Variabel" if lang == "id" else "Variable"
    col_dir = "Dampak Langsung" if lang == "id" else "Direct Effect"
    col_ind = "Tidak Langsung (Limpahan)" if lang == "id" else "Indirect (Spillover)"
    col_tot = "Dampak Total" if lang == "id" else "Total Effect"
    col_share = "Porsi Limpahan" if lang == "id" else "Spillover Share"
    col_p = "p-value"
    col_sig = "Signifikansi" if lang == "id" else "Significance"

    rows = []
    for e in result.effects:
        sig_text = ("p ≤ 0.05" if e.p_value <= 0.05 else ("Tidak Sig" if lang == "id" else "Not Sig"))
        rows.append(
            {
                col_var: e.variable,
                col_dir: round(e.direct_effect, 4),
                col_ind: round(e.indirect_effect, 4),
                col_tot: round(e.total_effect, 4),
                col_share: f"{e.spillover_ratio * 100:.1f}%",
                col_p: round(e.p_value, 4),
                col_sig: sig_text,
            }
        )
    return pd.DataFrame(rows)


def render_spillover_effects_table_html(
    result_or_df: Union[SpilloverResult, pd.DataFrame],
    is_bright: bool = False,
    lang: str = "en",
) -> str:
    """Render adaptive theme-aware HTML table for spatial spillover effects decomposition.

    Styling tokens:
    - Light Theme: container `#ffffff`, table `#ffffff`, header `#f1f5f9`, text `#0f172a !important`, borders `#e2e8f0`.
    - Dark Theme: container `#161e2a`, table `#161e2a`, header `#0f172a`, text `#f1f5f9`, borders `#2a384c`.
    """
    table_df = (
        render_spillover_effects_table(result_or_df, lang=lang)
        if isinstance(result_or_df, SpilloverResult)
        else result_or_df
    )

    container_bg = "#ffffff" if is_bright else "#161e2a"
    header_bg = "#f1f5f9" if is_bright else "#0f172a"
    text_color = "#0f172a !important" if is_bright else "#f1f5f9"
    border_color = "#cbd5e1" if is_bright else "#2a384c"
    row_border = "#e2e8f0" if is_bright else "#2a384c"

    lines = [
        f'<div style="background: {container_bg}; border: 1px solid {border_color}; border-radius: 4px; overflow: hidden; font-family: monospace; font-size: 11px;">',
        f'<table style="width: 100%; border-collapse: collapse; text-align: left; background: {container_bg}; color: {text_color};">',
        '<thead>',
        f'<tr style="background: {header_bg}; border-bottom: 1px solid {row_border};">',
    ]
    for col in table_df.columns:
        lines.append(f'<th style="padding: 6px 10px; font-weight: 600; color: {text_color}; border-bottom: 1px solid {row_border};">{col}</th>')
    lines.append('</tr>')
    lines.append('</thead>')
    lines.append('<tbody>')

    for _, row in table_df.iterrows():
        lines.append(f'<tr style="border-bottom: 1px solid {row_border};">')
        for col in table_df.columns:
            val = str(row[col])
            lines.append(f'<td style="padding: 5px 10px; color: {text_color};">{val}</td>')
        lines.append('</tr>')

    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')
    return "\n".join(line.strip() for line in lines)
