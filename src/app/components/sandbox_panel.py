"""Policy Intervention Sandbox component for interactive resilience simulations.

Provides high-density matte controls and KPI metric strip for simulated digital
and community social capital interventions.
"""
from typing import Dict, Optional, Tuple
import streamlit as st

from src.analytics.simulation import SimulationResult
from src.config import COLOR_TOKENS


def render_simulation_delta_html(
    sim_result: SimulationResult,
    is_bright: bool = False,
    theme_tokens: Optional[Dict[str, str]] = None,
    lang: str = "en",
) -> str:
    """Generate high-density matte KPI strip for policy simulation outcomes.

    Strict 0-indent markup to prevent Streamlit / Markdown code-block escaping.
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

    baseline = int(sim_result.baseline_critical_gaps)
    simulated = int(sim_result.simulated_critical_gaps)
    resolved = int(sim_result.resolved_gaps)
    delta_pct = (resolved / baseline * 100.0) if baseline > 0 else 0.0

    bg_surface = "#ffffff" if is_bright else tokens.get("bg-surface", "#161e2a")
    border_subtle = "#cbd5e1" if is_bright else tokens.get("border-subtle", "#2a384c")
    bg_elevated = "#f8fafc" if is_bright else tokens.get("bg-surface-elevated", "#0f141c")
    text_primary = "#0f172a" if is_bright else tokens.get("text-primary", "#f1f5f9")
    text_muted = "#64748b" if is_bright else tokens.get("text-muted", "#94a3b8")
    accent = "#0284c7" if is_bright else tokens.get("accent-focus", "#38bdf8")
    base_color = "#dc2626" if is_bright else "#f87171"

    resolved_color = accent if resolved > 0 else text_muted
    resolved_sign = f"+{resolved}" if resolved > 0 else f"{resolved}"

    title_text = "Dampak Intervensi Kebijakan" if lang == "id" else "Policy Intervention Impact"
    reduction_text = f"Penurunan {delta_pct:.1f}% Kesenjangan Kritis" if lang == "id" else f"{delta_pct:.1f}% Reduction in Critical Gaps"
    base_label = "Kesenjangan Kritis Dasar" if lang == "id" else "Baseline Critical Gaps"
    sim_label = "Kesenjangan Kritis Simulasi" if lang == "id" else "Simulated Critical Gaps"
    net_label = "Bersih Teratasi" if lang == "id" else "Net Resolved"
    note_text = "Transisi Risiko Tinggi (R3) & Mitigasi Rendah (M1)" if lang == "id" else "High Risk (R3) & Low Mitigation (M1) Transition"

    html = f"""<div style='background: {bg_surface}; border: 1px solid {border_subtle}; border-radius: 4px; padding: 12px; font-family: monospace; margin-bottom: 12px;'>
<div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;'>
<span style='font-size: 11px; text-transform: uppercase; color: {text_muted}; letter-spacing: 0.5px;'>{title_text}</span>
<span style='font-size: 10px; color: {resolved_color}; font-weight: 600;'>{reduction_text}</span>
</div>
<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; text-align: left;'>
<div style='background: {bg_elevated}; padding: 8px 10px; border: 1px solid {border_subtle}; border-radius: 2px;'>
<div style='font-size: 10px; color: {text_muted};'>{base_label}</div>
<div style='font-size: 16px; font-weight: 700; color: {base_color}; margin-top: 2px;'>{baseline}</div>
</div>
<div style='background: {bg_elevated}; padding: 8px 10px; border: 1px solid {border_subtle}; border-radius: 2px;'>
<div style='font-size: 10px; color: {text_muted};'>{sim_label}</div>
<div style='font-size: 16px; font-weight: 700; color: {text_primary}; margin-top: 2px;'>{simulated}</div>
</div>
<div style='background: {bg_elevated}; padding: 8px 10px; border: 1px solid {border_subtle}; border-radius: 2px;'>
<div style='font-size: 10px; color: {text_muted};'>{net_label}</div>
<div style='font-size: 16px; font-weight: 700; color: {resolved_color}; margin-top: 2px;'>{resolved_sign}</div>
</div>
</div>
<div style='margin-top: 8px; font-size: 10px; color: {text_muted}; text-align: right;'>
{note_text}
</div>
</div>"""
    return html.strip()


def render_sandbox_controls(lang: str = "en") -> Tuple[float, float, bool]:
    """Render interactive Streamlit policy sliders for digital & social resilience.

    Returns:
        Tuple containing:
        - signal_reduction (float): Telecom expansion percentage [0, 100].
        - gotong_royong_boost (float): Gotong royong boost percentage [0, 100].
        - is_active (bool): True if at least one intervention is non-zero.
    """
    label_signal = "Perluasan Sinyal Telekomunikasi (+%)" if lang == "id" else "Telecom Signal Expansion (+%)"
    label_gotong = "Mobilisasi Gotong Royong (+%)" if lang == "id" else "Gotong Royong Mobilization (+%)"
    signal_reduction = float(
        st.slider(label_signal, 0, 100, 0, step=5)
    )
    gotong_royong_boost = float(
        st.slider(label_gotong, 0, 100, 0, step=5)
    )
    is_active = bool(signal_reduction > 0.0 or gotong_royong_boost > 0.0)
    return signal_reduction, gotong_royong_boost, is_active
