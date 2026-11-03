"""Internationalization (i18n) module for IDR-GE.

Provides bilingual English / Indonesian UI dictionary and lookup helper.
"""
from typing import Dict

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # App Header
    "app_title": {
        "en": "Indonesian Disaster Readiness & Gap Explorer (IDR-GE)",
        "id": "Eksplorasi Kesiapsiagaan & Kesenjangan Bencana Indonesia (IDR-GE)",
    },
    "app_subtitle_prefix": {
        "en": "Active Viewport",
        "id": "Tampilan Aktif",
    },
    "app_scope_prefix": {
        "en": "Scope",
        "id": "Cakupan",
    },
    "units_label": {
        "en": "units",
        "id": "kab/kota",
    },
    "sim_active_badge": {
        "en": " [SIMULATION ACTIVE]",
        "id": " [SIMULASI AKTIF]",
    },

    # Sidebar & Theme / Lang Controls
    "theme_toggle_label": {
        "en": "🌙 Dark Theme",
        "id": "🌙 Mode Gelap",
    },
    "lang_toggle_label": {
        "en": "🌐 Language / Bahasa",
        "id": "🌐 Bahasa / Language",
    },
    "sidebar_brand_sub": {
        "en": "Disaster Readiness & Gap Explorer",
        "id": "Kesiapsiagaan & Kesenjangan Bencana",
    },
    "map_layer_mode": {
        "en": "Map Layer Mode",
        "id": "Mode Lapisan Peta",
    },
    "layer_bivariate": {
        "en": "Bivariate Risk vs Mitigation (3x3)",
        "id": "Risiko Bivariat vs Mitigasi (3x3)",
    },
    "layer_balance": {
        "en": "Readiness Balance (Deficit-Surplus)",
        "id": "Neraca Kesiapsiagaan (Defisit-Surplus)",
    },
    "layer_lisa": {
        "en": "LISA Spatial Clusters",
        "id": "Klaster Spasial LISA",
    },
    "spatial_indicator": {
        "en": "Spatial Indicator",
        "id": "Indikator Spasial",
    },
    "significance_level": {
        "en": "Significance Level (α)",
        "id": "Tingkat Signifikansi (α)",
    },
    "policy_sandbox": {
        "en": "Policy Intervention Sandbox",
        "id": "Simulasi Intervensi Kebijakan",
    },
    "telecom_signal_expansion": {
        "en": "Telecom Signal Expansion",
        "id": "Perluasan Sinyal Telekomunikasi",
    },
    "gotong_royong_mobilization": {
        "en": "Gotong Royong Mobilization",
        "id": "Mobilisasi Gotong Royong",
    },
    "regional_corridor": {
        "en": "Regional Corridor",
        "id": "Koridor Wilayah / Pulau",
    },
    "national_all_corridors": {
        "en": "National (All 7 Corridors)",
        "id": "Nasional (Semua 7 Koridor)",
    },
    "filter_province": {
        "en": "Filter Province",
        "id": "Filter Provinsi",
    },
    "all_provinces": {
        "en": "All 38 Provinces",
        "id": "Seluruh 38 Provinsi",
    },
    "all_prefix": {
        "en": "All",
        "id": "Seluruh",
    },
    "provinces_suffix": {
        "en": "Provinces",
        "id": "Provinsi",
    },
    "sim_impact_title": {
        "en": "Policy Intervention Impact",
        "id": "Dampak Intervensi Kebijakan",
    },
    "reduction_in_critical": {
        "en": "{pct:.1f}% Reduction in Critical Gaps",
        "id": "Penurunan {pct:.1f}% Kesenjangan Kritis",
    },
    "baseline_critical": {
        "en": "Baseline Critical Gaps",
        "id": "Kesenjangan Kritis Dasar",
    },
    "simulated_critical": {
        "en": "Simulated Critical Gaps",
        "id": "Kesenjangan Kritis Simulasi",
    },
    "net_resolved": {
        "en": "Net Resolved",
        "id": "Bersih Teratasi",
    },
    "sim_footer_note": {
        "en": "High Risk (R3) & Low Mitigation (M1) Transition",
        "id": "Transisi Risiko Tinggi (R3) & Mitigasi Rendah (M1)",
    },
    "units_in_view": {
        "en": "Units in View",
        "id": "Jumlah Wilayah",
    },
    "corridor_meta": {
        "en": "Corridor",
        "id": "Koridor",
    },
    "topology_meta": {
        "en": "Topology",
        "id": "Topologi",
    },
    "export_corridor_brief": {
        "en": "Export {corridor} Brief (MD)",
        "id": "Ekspor Ringkasan {corridor} (MD)",
    },

    # Analytics Panel (Right)
    "analytics_view": {
        "en": "Analytics View",
        "id": "Tampilan Analisis",
    },
    "view_moran": {
        "en": "Spatial Autocorrelation (Moran)",
        "id": "Autokorelasi Spasial (Moran)",
    },
    "view_parity": {
        "en": "4-Quadrant Parity (M vs R)",
        "id": "Paritas 4-Kuadran (M vs R)",
    },
    "view_spillover": {
        "en": "Spatial Spillover & Multipliers",
        "id": "Limpahan Spasial & Efek Pengganda",
    },
    "global_autocorrelation": {
        "en": "Global Autocorrelation",
        "id": "Autokorelasi Global",
    },
    "stat_significant": {
        "en": "Statistically Significant (p ≤ 0.05)",
        "id": "Signifikan Secara Statistik (p ≤ 0.05)",
    },
    "not_significant": {
        "en": "Not Significant (p > 0.05)",
        "id": "Tidak Signifikan (p > 0.05)",
    },
    "moran_i": {
        "en": "Moran's I",
        "id": "Moran's I",
    },
    "expected_i": {
        "en": "Expected I",
        "id": "Expected I",
    },
    "z_score": {
        "en": "z-score",
        "id": "z-score",
    },
    "p_value_sim": {
        "en": "p-value (sim)",
        "id": "p-value (sim)",
    },
    "monte_carlo_perms": {
        "en": "Monte Carlo permutations",
        "id": "Permutasi Monte Carlo",
    },
    "moran_scatterplot": {
        "en": "Moran's I Scatterplot",
        "id": "Diagram Pencar Moran's I",
    },
    "spatial_lag": {
        "en": "Spatial Lag (Wz)",
        "id": "Lag Spasial (Wz)",
    },
    "standardized_var": {
        "en": "Standardized {var} (z)",
        "id": "{var} Terstandarisasi (z)",
    },
    "top_critical_gap_title": {
        "en": "Top 10 Critical Gap Units (Risk > Mitigation)",
        "id": "10 Daerah Kesenjangan Paling Kritis (Risiko > Mitigasi)",
    },
    "top_critical_gap_sim_title": {
        "en": "Top 10 Critical Gap Units (Simulated Risk > Mitigation)",
        "id": "10 Daerah Kesenjangan Paling Kritis (Simulasi Risiko > Mitigasi)",
    },

    # Econometric Spillover Panel
    "econometric_spec": {
        "en": "Econometric Specification",
        "id": "Spesifikasi Ekonometrika",
    },
    "spec_sar": {
        "en": "Spatial Lag (SAR: Y = ρWY + Xβ + ε)",
        "id": "Spatial Lag (SAR: Y = ρWY + Xβ + ε)",
    },
    "spec_slx": {
        "en": "Spatial Lag of X (SLX: Y = Xβ + WXθ + ε)",
        "id": "Spatial Lag of X (SLX: Y = Xβ + WXθ + ε)",
    },
    "cross_boundary_title": {
        "en": "Cross-Boundary Effect Decomposition",
        "id": "Dekomposisi Dampak Lintas Wilayah",
    },
    "spatial_autoreg_rho": {
        "en": "Spatial Autoregressive Rho (ρ)",
        "id": "Koefisien Spasial Autoregresif Rho (ρ)",
    },
    "spatial_multiplier": {
        "en": "Spatial Multiplier",
        "id": "Pengganda Spasial",
    },
    "spillover_ratio": {
        "en": "Spillover Ratio (Indirect/Total)",
        "id": "Rasio Limpahan (Tidak Langsung/Total)",
    },

    # Bottom Panels & Viewport Footer
    "risk_vs_mitigation": {
        "en": "3×3 Risk vs. Mitigation",
        "id": "3×3 Risiko vs. Mitigasi",
    },
    "critical_gap_badge": {
        "en": "Critical Gap",
        "id": "Kesenjangan Kritis",
    },
    "resilient_badge": {
        "en": "Resilient",
        "id": "Tangguh / Resilien",
    },
    "territorial_resilience_title": {
        "en": "TERRITORIAL RESILIENCE & HAZARD DRIVERS",
        "id": "KETAHANAN TERITORIAL & PENDORONG BAHAYA",
    },
    "avg_gap_score": {
        "en": "AVG GAP SCORE",
        "id": "RATA-RATA GAP",
    },
    "bad_telecom": {
        "en": "BAD TELECOM",
        "id": "SINYAL BURUK",
    },
    "social_capital": {
        "en": "SOCIAL CAPITAL",
        "id": "MODAL SOSIAL",
    },
    "hazard_floods": {
        "en": "Floods / Water Hazard",
        "id": "Bahaya Banjir / Genangan Air",
    },
    "hazard_tsunami": {
        "en": "Tsunami EWS Telemetry",
        "id": "Telemetri EWS Tsunami",
    },
    "hazard_landslides": {
        "en": "Landslides / Slopes Risk",
        "id": "Risiko Longsor / Kemiringan",
    },

    # Table Column Headers
    "col_regency": {
        "en": "Regency / City",
        "id": "Kabupaten / Kota",
    },
    "col_province": {
        "en": "Province",
        "id": "Provinsi",
    },
    "col_risk": {
        "en": "Risk (R)",
        "id": "Risiko (R)",
    },
    "col_mitigation": {
        "en": "Mitigation (M)",
        "id": "Mitigasi (M)",
    },
    "col_gap": {
        "en": "Gap (R - M)",
        "id": "Kesenjangan (R - M)",
    },
    "col_category": {
        "en": "Category",
        "id": "Kategori",
    },

    # District Micro-Profile Inspector
    "district_inspector": {
        "en": "District Micro-Profile Inspector",
        "id": "Inspektur Profil Mikro Wilayah",
    },
    "select_unit_inspect": {
        "en": "Select Administrative Unit to Inspect ({count} Regencies)",
        "id": "Pilih Wilayah Administratif untuk Diinspeksi ({count} Kab/Kota)",
    },
    "lbl_unmitigated_risk": {
        "en": "UNMITIGATED RISK",
        "id": "DEFISIT RISIKO",
    },
    "lbl_net_surplus": {
        "en": "NET SURPLUS",
        "id": "SURPLUS MITIGASI",
    },
    "download_brief_btn": {
        "en": "Download Executive Policy Brief (Markdown) — {kab}",
        "id": "Unduh Ringkasan Kebijakan Eksekutif (Markdown) — {kab}",
    },
    "download_pdf_brief_btn": {
        "en": "📄 Download Executive Policy Brief (PDF) — {kab}",
        "id": "📄 Unduh Ringkasan Kebijakan Eksekutif (PDF) — {kab}",
    },
    "download_prov_pdf_btn": {
        "en": "📄 Download Provincial Policy Brief (PDF) — {prov}",
        "id": "📄 Unduh Policy Brief Provinsi (PDF) — {prov}",
    },

    # Micro-indicator readable names
    "ind_ews": {
        "en": "Early Warning System (EWS)",
        "id": "Peringatan Dini (EWS)",
    },
    "ind_rambu": {
        "en": "Evacuation Signage",
        "id": "Rambu Evakuasi",
    },
    "ind_perkap": {
        "en": "Safety Equipment (Boats, Tents, Gear)",
        "id": "Perlengkapan Keselamatan",
    },
    "ind_rawat_sungai": {
        "en": "Water Area Maintenance (Rivers/Canals/Dykes)",
        "id": "Perawatan Wilayah Perairan (Sungai/Kanal/Tanggul)",
    },
    "ind_rawat_air": {
        "en": "Water Area Maintenance (Rivers/Canals/Dykes)",
        "id": "Perawatan Wilayah Perairan (Sungai/Kanal/Tanggul)",
    },
    "ind_ada_bencana": {
        "en": "Disaster Occurrence (2023-2024)",
        "id": "Kejadian Bencana (2023–2024)",
    },
    "ind_gotong_royong": {
        "en": "Social Solidarity",
        "id": "Gotong Royong",
    },
    "ind_sinyal_buruk": {
        "en": "Weak Telecom Coverage",
        "id": "Keterbatasan Sinyal",
    },
    "ind_ipm": {
        "en": "Human Development Index (HDI)",
        "id": "Indeks Pembangunan Manusia (IPM)",
    },
    "ind_tpm": {
        "en": "Temporary Evacuation Shelters",
        "id": "Tempat Penampungan",
    },

    # Humanized benchmark comparisons & badges
    "bench_outperforms_nat": {
        "en": "Outperforms {nat_pct:.0f}% of regencies nationally",
        "id": "Melampaui {nat_pct:.0f}% daerah se-Indonesia",
    },
    "bench_outperforms_corr": {
        "en": "Outperforms {corr_pct:.0f}% across {corridor}",
        "id": "Melampaui {corr_pct:.0f}% daerah di {corridor}",
    },
    "bench_burden_nat": {
        "en": "Higher burden than {nat_pct:.0f}% of regencies nationally",
        "id": "Tingkat keterbatasan/kerentanan lebih tinggi dari {nat_pct:.0f}% daerah se-Indonesia",
    },
    "bench_burden_corr": {
        "en": "Higher burden than {corr_pct:.0f}% across {corridor}",
        "id": "Tingkat keterbatasan/kerentanan lebih tinggi dari {corr_pct:.0f}% daerah di {corridor}",
    },
    "warn_preparedness_deficit": {
        "en": "Mitigation Insufficient for Severe Risk (Preparedness Deficit)",
        "id": "Mitigasi Belum Mengimbangi Risiko Ekstrem (Defisit Kesiapsiagaan)",
    },
    "bench_value_label": {
        "en": "Value",
        "id": "Nilai",
    },
    "bench_median_label": {
        "en": "National Median",
        "id": "Median Nasional",
    },
    "status_lagging": {
        "en": "Lagging",
        "id": "Tertinggal",
    },
    "status_below_med": {
        "en": "Below Median",
        "id": "Di Bawah Median",
    },
    "status_above_med": {
        "en": "Above Median",
        "id": "Cukup Baik",
    },
    "status_leading": {
        "en": "Leading",
        "id": "Unggul",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """Retrieve translated string for the provided key and language code.

    Falls back to English if the key or language translation is not found.
    Accepts string formatting kwargs.
    """
    lang_key = lang.lower() if lang else "en"
    if lang_key not in ("en", "id"):
        lang_key = "en"

    entry = TRANSLATIONS.get(key, {})
    val = entry.get(lang_key) or entry.get("en", key)
    if kwargs:
        try:
            return val.format(**kwargs)
        except Exception:
            return val
    return val


def get_indicator_label(key: str, lang: str = "en") -> str:
    """Return descriptive localized label for an indicator key."""
    normalized = f"ind_{key.lower()}"
    if normalized in TRANSLATIONS:
        return t(normalized, lang=lang)
    return key
