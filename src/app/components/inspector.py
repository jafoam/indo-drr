"""District Micro-Profile Inspector Components.

Provides dark matte Plotly polar radar visualization and high-density monospace
inspector card HTML for district-level disaster resilience diagnostics.
"""
from typing import List, Optional
import plotly.graph_objects as go

from src.config import COLOR_TOKENS
from src.analytics.profile import RegencyProfile, PROFILE_INDICATORS, NEGATIVE_INDICATORS
from src.app.i18n import get_indicator_label, t


def render_radar_chart(
    profile: RegencyProfile,
    indicators: Optional[List[str]] = None,
    is_bright: bool = False,
    theme: Optional[str] = None,
) -> go.Figure:
    """Render an adaptive Plotly polar radar chart comparing district indicators against benchmarks.

    Compares the target district's indicators against both corridor-level and national-level
    median benchmarks with 3 overlapping polar traces. Adapts cleanly between Dark and Light Themes.

    Args:
        profile: Computed RegencyProfile.
        indicators: Subset or custom ordering of indicators (defaults to PROFILE_INDICATORS).
        is_bright: Explicit flag indicating whether Light Theme is active.
        theme: Optional visual theme string ("Dark Theme" or "Light Theme").

    Returns:
        Plotly Figure configured with adaptive tokens and polar coordinates.
    """
    if isinstance(is_bright, str):
        is_bright = (is_bright == "Light Theme")
    elif theme is not None:
        is_bright = (theme == "Light Theme")
    else:
        is_bright = bool(is_bright)

    inds = indicators or PROFILE_INDICATORS

    # Theme palette tokens
    if is_bright:
        paper_bg = "#ffffff"
        plot_bg = "#ffffff"
        font_color = "#0f172a"
        polar_bg = "#ffffff"
        radial_grid = "#cbd5e1"
        radial_line = "#cbd5e1"
        radial_tick = "#334155"
        angular_grid = "#cbd5e1"
        angular_line = "#0f172a"
        angular_tick = "#0f172a"
        accent_color = "#0284c7"
        accent_fill = "rgba(2, 132, 199, 0.20)"
        nat_line_color = "#64748b"
        legend_font_color = "#0f172a"
    else:
        paper_bg = COLOR_TOKENS["bg-surface"]
        plot_bg = COLOR_TOKENS["bg-surface"]
        font_color = COLOR_TOKENS["text-primary"]
        polar_bg = COLOR_TOKENS["bg-primary"]
        radial_grid = COLOR_TOKENS["border-subtle"]
        radial_line = COLOR_TOKENS["border-subtle"]
        radial_tick = COLOR_TOKENS["text-muted"]
        angular_grid = COLOR_TOKENS["border-subtle"]
        angular_line = COLOR_TOKENS["border-subtle"]
        angular_tick = COLOR_TOKENS["text-primary"]
        accent_color = COLOR_TOKENS["accent-focus"]
        accent_fill = "rgba(56, 189, 248, 0.20)"
        nat_line_color = COLOR_TOKENS["text-muted"]
        legend_font_color = COLOR_TOKENS["text-muted"]

    # Extract series
    district_vals = [profile.indicators[k].value for k in inds]
    corr_medians = [profile.indicators[k].corridor_median for k in inds]
    nat_medians = [profile.indicators[k].national_median for k in inds]

    fig = go.Figure()

    # Trace 1: Target District
    fig.add_trace(
        go.Scatterpolar(
            r=district_vals,
            theta=inds,
            fill="toself",
            name=profile.kab_kota,
            line=dict(color=accent_color, width=2.5),
            fillcolor=accent_fill,
            hovertemplate="<b>%{theta}</b><br>Value: %{r:.2f}<extra></extra>",
        )
    )

    # Trace 2: Corridor Median Benchmark
    fig.add_trace(
        go.Scatterpolar(
            r=corr_medians,
            theta=inds,
            fill="toself",
            name=f"{profile.corridor} Median",
            line=dict(color="#f59e0b", width=1.8, dash="dot"),
            fillcolor="rgba(245, 158, 11, 0.05)",
            hovertemplate="<b>%{theta}</b><br>Corridor Med: %{r:.2f}<extra></extra>",
        )
    )

    # Trace 3: National Median Benchmark
    fig.add_trace(
        go.Scatterpolar(
            r=nat_medians,
            theta=inds,
            fill="none",
            name="National Median",
            line=dict(color=nat_line_color, width=1.5, dash="dash"),
            hovertemplate="<b>%{theta}</b><br>Nat Med: %{r:.2f}<extra></extra>",
        )
    )

    # Adaptive Layout adhering to Dark / Light Theme tokens
    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family="monospace", color=font_color),
        margin=dict(l=40, r=40, t=35, b=35),
        polar=dict(
            bgcolor=polar_bg,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=radial_grid,
                linecolor=radial_line,
                tickfont=dict(color=radial_tick, size=9),
            ),
            angularaxis=dict(
                gridcolor=angular_grid,
                linecolor=angular_line,
                tickfont=dict(color=angular_tick, size=10, family="monospace"),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color=legend_font_color, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )

    return fig


def render_inspector_card_html(
    profile: RegencyProfile,
    is_bright: bool = False,
    theme: Optional[str] = None,
    lang: str = "en",
) -> str:
    """Generate high-density monospace summary card HTML for a district profile.

    Complies strictly with antislop-ui rules:
    - Zero lines indented with 4 spaces or tabs.
    - No generic glow or purple gradients.
    - Clean 1px borders and semantic badge indicators.
    - Adapts cleanly between Dark Theme and Light Theme.

    Args:
        profile: Computed RegencyProfile.
        is_bright: Explicit flag indicating whether Light Theme is active.
        theme: Optional visual theme string ("Dark Theme" or "Light Theme").
        lang: Language code ("en" or "id").

    Returns:
        Flush 0-indent HTML string starting with <div.
    """
    if isinstance(is_bright, str):
        is_bright = (is_bright == "Light Theme")
    elif theme is not None:
        is_bright = (theme == "Light Theme")
    else:
        is_bright = bool(is_bright)

    if is_bright:
        container_bg = "#ffffff"
        container_border = "#cbd5e1"
        text_pri = "#0f172a"
        text_val = "#334155"
        text_mut = "#64748b"
        accent_color = "#0284c7"
        grid_bg = "#f8fafc"
        divider = "#cbd5e1"
        sub_border = "rgba(203, 213, 225, 0.7)"
        badge_style = f"background-color: {profile.bivariate_color}22; color: #0f172a; border: 1px solid {profile.bivariate_color};"
    else:
        container_bg = "#1a2332"
        container_border = "#2a384c"
        text_pri = "#f1f5f9"
        text_val = "#f1f5f9"
        text_mut = "#94a3b8"
        accent_color = "#38bdf8"
        grid_bg = "#0f141c"
        divider = "#2a384c"
        sub_border = "rgba(42, 56, 76, 0.4)"
        badge_style = f"background-color: {profile.bivariate_color}; color: #ffffff;"

    lbl_prov = "PROVINSI" if lang == "id" else "PROVINCE"
    lbl_corr = "KORIDOR" if lang == "id" else "CORRIDOR"
    lbl_rank = "PERINGKAT GAP" if lang == "id" else "GAP RANK"
    lbl_risk = "SKOR RISIKO" if lang == "id" else "RISK SCORE"
    lbl_mit = "MITIGASI" if lang == "id" else "MITIGATION"
    lbl_bench = "TOLOK UKUR MIKRO-INDIKATOR" if lang == "id" else "MICRO-INDICATOR BENCHMARKS"

    # Clarify positive gap as capacity deficit (unmitigated risk)
    if profile.gap_score > 0:
        lbl_gap = t("lbl_unmitigated_risk", lang)
        gap_val_color = "#b91c1c" if is_bright else "#f87171"
    else:
        lbl_gap = t("lbl_net_surplus", lang)
        gap_val_color = "#15803d" if is_bright else "#4ade80"

    has_severe_deficit = (
        profile.bivariate_category in ("R3-M3", "R3-M2")
        and (profile.gap_score >= 0.25 or profile.risk_score >= 1.5 * profile.mitigation_score)
    )
    warn_text = t("warn_preparedness_deficit", lang)

    lines = [
        f'<div style="background: {container_bg}; border: 1px solid {container_border}; border-radius: 4px; padding: 12px; font-family: monospace; color: {text_pri};">',
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; border-bottom: 1px solid {divider}; padding-bottom: 6px;">',
        f'<div><span style="font-size: 14px; font-weight: bold; color: {text_pri};">{profile.kab_kota}</span> <span style="font-size: 11px; color: {text_mut};">({profile.kode_wil})</span></div>',
        '<div style="text-align: right;">',
        f'<div style="{badge_style} padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: bold; text-transform: uppercase; display: inline-block;">{profile.bivariate_label} ({profile.bivariate_category})</div>',
    ]

    if has_severe_deficit:
        lines.append(
            f'<div style="border: 1px solid #f59e0b; color: #f59e0b; background: rgba(245, 158, 11, 0.1); border-radius: 3px; padding: 2px 6px; font-size: 9.5px; font-weight: bold; margin-top: 3px; text-align: right;">{warn_text}</div>'
        )

    lines.extend([
        '</div>',
        '</div>',
        f'<div style="display: flex; justify-content: space-between; font-size: 11px; color: {text_mut}; margin-bottom: 10px;">',
        f'<span>{lbl_prov}: <strong style="color: {text_pri};">{profile.provinsi}</strong></span>',
        f'<span>{lbl_corr}: <strong style="color: {accent_color};">{profile.corridor}</strong></span>',
        f'<span>{lbl_rank}: <strong style="color: {text_val};">#{profile.gap_rank}</strong> / 514</span>',
        '</div>',
        f'<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 10px; background-color: {grid_bg}; padding: 8px; border: 1px solid {divider}; border-radius: 3px;">',
        f'<div style="text-align: center;"><div style="font-size: 10px; color: {text_mut};">{lbl_gap}</div><div style="font-size: 13px; font-weight: bold; color: {gap_val_color};">{profile.gap_score:+.3f}</div></div>',
        f'<div style="text-align: center;"><div style="font-size: 10px; color: {text_mut};">{lbl_risk}</div><div style="font-size: 13px; font-weight: bold; color: {text_val};">{profile.risk_score:.3f}</div></div>',
        f'<div style="text-align: center;"><div style="font-size: 10px; color: {text_mut};">{lbl_mit}</div><div style="font-size: 13px; font-weight: bold; color: {text_val};">{profile.mitigation_score:.3f}</div></div>',
        '</div>',
    ])

    # Adequacy Ratio KPI Card styling
    adequacy_val = getattr(profile, "adequacy_ratio", (profile.mitigation_score / profile.risk_score * 100.0) if profile.risk_score > 0 else 100.0)
    deficit_val = max(0.0, 100.0 - adequacy_val)
    if adequacy_val < 50.0:
        kpi_border = "#fca5a5" if is_bright else "#ef4444"
        kpi_bg = "#fef2f2" if is_bright else "rgba(239, 68, 68, 0.15)"
        kpi_color = "#b91c1c" if is_bright else "#f87171"
    elif adequacy_val < 80.0:
        kpi_border = "#fcd34d" if is_bright else "#f59e0b"
        kpi_bg = "#fffbeb" if is_bright else "rgba(245, 158, 11, 0.15)"
        kpi_color = "#b45309" if is_bright else "#fbbf24"
    else:
        kpi_border = "#86efac" if is_bright else "#22c55e"
        kpi_bg = "#f0fdf4" if is_bright else "rgba(34, 197, 94, 0.15)"
        kpi_color = "#15803d" if is_bright else "#4ade80"

    kpi_title = "Rasio Kecukupan Kesiapsiagaan" if lang == "id" else "Readiness Adequacy Ratio"
    kpi_subtext = (
        f"Kapasitas mitigasi memenuhi {adequacy_val:.1f}% kebutuhan risiko (Defisit: {deficit_val:.1f}%)"
        if lang == "id"
        else f"Mitigation capacity meets {adequacy_val:.1f}% of hazard risk demand (Deficit: {deficit_val:.1f}%)"
    )

    lines.extend([
        f'<div style="background: {kpi_bg}; border: 1px solid {kpi_border}; border-radius: 4px; padding: 7px 10px; margin-bottom: 8px; box-sizing: border-box;">',
        f'<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2px;">',
        f'<span style="font-size: 10px; font-weight: bold; color: {text_mut}; text-transform: uppercase; letter-spacing: 0.5px;">{kpi_title}</span>',
        f'<span style="font-size: 14px; font-weight: 800; color: {kpi_color};">{adequacy_val:.1f}%</span>',
        '</div>',
        f'<div style="font-size: 10px; color: {text_mut}; line-height: 1.35;">{kpi_subtext}</div>',
        '</div>',
        '<div style="font-size: 11px; margin-top: 8px;">',
        f'<div style="font-size: 10px; font-weight: bold; color: {text_mut}; margin-bottom: 4px; border-bottom: 1px solid {divider}; padding-bottom: 2px;">{lbl_bench}</div>',
    ])

    for k in PROFILE_INDICATORS:
        m = profile.indicators[k]
        ind_title = get_indicator_label(k, lang=lang)

        is_neg = k in NEGATIVE_INDICATORS or getattr(m, "is_negative", False)
        effective_nat_pct = (100.0 - m.national_pct) if is_neg else m.national_pct
        effective_corr_pct = (100.0 - m.corridor_pct) if is_neg else m.corridor_pct

        # Status badge determination based on effective national percentile
        if effective_nat_pct < 25.0:
            badge_text = t("status_lagging", lang)
            badge_css = (
                "background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5;"
                if is_bright
                else "background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444;"
            )
        elif effective_nat_pct <= 50.0:
            badge_text = t("status_below_med", lang)
            badge_css = (
                "background: #fef3c7; color: #b45309; border: 1px solid #fcd34d;"
                if is_bright
                else "background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b;"
            )
        elif effective_nat_pct <= 75.0:
            badge_text = t("status_above_med", lang)
            badge_css = (
                "background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc;"
                if is_bright
                else "background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #0284c7;"
            )
        else:
            badge_text = t("status_leading", lang)
            badge_css = (
                "background: #dcfce7; color: #15803d; border: 1px solid #86efac;"
                if is_bright
                else "background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid #22c55e;"
            )

        val_label = t("bench_value_label", lang)
        med_label = t("bench_median_label", lang)

        if is_neg:
            outperform_nat = t("bench_burden_nat", lang, nat_pct=m.national_pct)
            outperform_corr = t("bench_burden_corr", lang, corr_pct=m.corridor_pct, corridor=profile.corridor)
        else:
            outperform_nat = t("bench_outperforms_nat", lang, nat_pct=m.national_pct)
            outperform_corr = t("bench_outperforms_corr", lang, corr_pct=m.corridor_pct, corridor=profile.corridor)

        lines.append(
            f'<div style="padding: 6px 0; border-bottom: 1px solid {sub_border}; width: 100%; box-sizing: border-box; font-size: 0.82rem; line-height: 1.35;">'
        )
        lines.append(
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">'
            f'<span style="font-weight: 600; color: {text_val};">{ind_title}</span>'
            f'<span style="{badge_css} padding: 1px 7px; border-radius: 9999px; font-size: 10px; font-weight: 600;">{badge_text}</span>'
            f'</div>'
        )
        lines.append(
            f'<div style="font-size: 11px; margin-bottom: 2px;">'
            f'<span style="color: {text_mut};">{val_label}: </span><strong style="color: {text_pri};">{m.value:.1f}</strong> '
            f'<span style="color: {text_mut};">({med_label}: {m.national_median:.1f})</span>'
            f'</div>'
        )
        lines.append(
            f'<div style="font-size: 10.5px; color: {accent_color};">'
            f'<span>{outperform_nat}</span> <span style="color: {text_mut};">|</span> <span>{outperform_corr}</span>'
            f'</div>'
        )
        lines.append('</div>')

    lines.append('</div>')
    lines.append('</div>')

    return "\n".join(line.strip() for line in lines)
