"""Tests for Micro-Profile District Inspector Module & Radar Component (Task 3.1)."""
from html.parser import HTMLParser
import plotly.graph_objects as go
import pytest

from src.api.routes import get_data_manager
from src.config import COLOR_TOKENS
from src.analytics.profile import (
    compute_regency_profile,
    RegencyProfile,
    IndicatorMetric,
)
from src.app.components.inspector import (
    render_radar_chart,
    render_inspector_card_html,
)

INDICATOR_KEYS = [
    "EWS",
    "RAMBU",
    "PERKAP",
    "RAWAT_SUNGAI",
    "GOTONG_ROYONG",
    "SINYAL_BURUK",
    "IPM",
]


class HTMLTagValidator(HTMLParser):
    """Validator for HTML tag closure and nesting."""

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
def data_context():
    """Load cached spatial DataManager."""
    dm = get_data_manager()
    dm.load()
    return dm.gdf, dm.bivariate_df


def test_compute_regency_profile_valid(data_context):
    """Verify compute_regency_profile produces valid profile with 7 indicators."""
    gdf, bivariate_df = data_context
    kode_wil = 1101  # Aceh Selatan

    profile = compute_regency_profile(gdf, bivariate_df, kode_wil)
    assert isinstance(profile, RegencyProfile)
    assert profile.kode_wil == 1101
    assert profile.kab_kota == "Simeulue"
    assert profile.provinsi == "Aceh"
    assert profile.corridor == "Sumatera"

    # All 7 indicators present
    for ind in INDICATOR_KEYS:
        assert ind in profile.indicators
        metric = profile.indicators[ind]
        assert isinstance(metric, IndicatorMetric)
        # Percentiles strictly between 0.0 and 100.0
        assert 0.0 <= metric.national_pct <= 100.0
        assert 0.0 <= metric.corridor_pct <= 100.0
        # Medians are positive floats
        assert isinstance(metric.national_median, float)
        assert metric.national_median > 0.0
        assert isinstance(metric.corridor_median, float)
        assert metric.corridor_median > 0.0

    # National rank on gap_score is an integer between 1 and 514
    assert isinstance(profile.gap_rank, int)
    assert 1 <= profile.gap_rank <= 514


def test_compute_regency_profile_invalid_kode(data_context):
    """Verify invalid kode_wil raises KeyError."""
    gdf, bivariate_df = data_context
    with pytest.raises(KeyError):
        compute_regency_profile(gdf, bivariate_df, 9999)


def test_render_radar_chart(data_context):
    """Verify radar chart is a polar go.Figure matching COLOR_TOKENS and indicators."""
    gdf, bivariate_df = data_context
    profile = compute_regency_profile(gdf, bivariate_df, 1101)

    fig = render_radar_chart(profile)
    assert isinstance(fig, go.Figure)

    # Polar coordinates
    assert len(fig.data) >= 3  # District, Corridor Median, National Median
    for trace in fig.data:
        assert trace.type == "scatterpolar"
        # Polar categories match the 7 indicators
        assert set(trace.theta) == set(INDICATOR_KEYS)

    # Anti-slop check: no generic blue-purple glow
    fig_json = fig.to_json()
    assert "#667eea" not in fig_json
    assert "linear-gradient" not in fig_json


def test_render_inspector_card_html(data_context):
    """Verify inspector card HTML complies with anti-slop rules and tag balance."""
    gdf, bivariate_df = data_context
    profile = compute_regency_profile(gdf, bivariate_df, 1101)

    # 1. Default / Dark Theme Card
    html = render_inspector_card_html(profile, is_bright=False)
    assert isinstance(html, str)
    assert html.startswith("<div")
    assert "#1a2332" in html
    assert "#2a384c" in html
    assert "#f1f5f9" in html

    # Anti-slop rule: Zero lines indented with 4 spaces or tabs
    bad_lines = [
        line for line in html.splitlines()
        if line.startswith("    ") or line.startswith("\t")
    ]
    assert len(bad_lines) == 0, f"Found lines with forbidden leading whitespace: {bad_lines}"

    # Required content: regency name, province, corridor, gap score badge
    assert profile.kab_kota in html
    assert profile.provinsi in html
    assert profile.corridor in html
    assert "gap" in html.lower() or "score" in html.lower()

    # Tag balance valid
    validator = HTMLTagValidator()
    validator.feed(html)
    assert validator.is_valid(), f"HTML tag errors: {validator.errors}"

    # 2. Light Theme Card: white container, slate border, dark title/label, slate value text
    html_light = render_inspector_card_html(profile, is_bright=True)
    assert "#ffffff" in html_light
    assert "#cbd5e1" in html_light
    assert "#0f172a" in html_light
    assert "#334155" in html_light
    assert "#1a2332" not in html_light
    assert "#0b0f19" not in html_light
    assert "color: #ffffff" not in html_light
    validator_light = HTMLTagValidator()
    validator_light.feed(html_light)
    assert validator_light.is_valid(), f"HTML tag errors in light theme: {validator_light.errors}"


def test_profile_serialization_and_properties(data_context):
    """Verify RegencyProfile and IndicatorMetric property aliases and serialization."""
    gdf, bivariate_df = data_context
    profile = compute_regency_profile(gdf, bivariate_df, 1101)

    assert profile.regency_name == profile.kab_kota
    assert profile.province == profile.provinsi
    assert profile.national_rank == profile.gap_rank
    assert len(profile.metrics) == 7
    assert profile["EWS"] == profile.indicators["EWS"]

    d = profile.to_dict()
    assert d["kode_wil"] == 1101
    assert "indicators" in d
    assert "EWS" in d["indicators"]
    assert d["indicators"]["EWS"]["value"] == profile.indicators["EWS"].value

    ews = profile.indicators["EWS"]
    assert ews.indicator == "EWS"
    assert ews.raw_value == ews.value
    ews_dict = ews.to_dict()
    assert ews_dict["name"] == "EWS"


@pytest.mark.parametrize("kw", [3171, 7371, 6471, 5171, 9471])
def test_compute_regency_profile_diverse_sample(data_context, kw):
    """Verify profile calculation across diverse provinces and corridors."""
    gdf, bivariate_df = data_context
    profile = compute_regency_profile(gdf, bivariate_df, kw)

    assert profile.kode_wil == kw
    assert 1 <= profile.gap_rank <= 514
    for ind in INDICATOR_KEYS:
        m = profile.indicators[ind]
        assert 0.0 <= m.national_pct <= 100.0
        assert 0.0 <= m.corridor_pct <= 100.0
        assert m.national_median > 0.0


def test_negative_indicator_polarity_and_effective_pct(data_context):
    """Verify negative indicator percentiles are inverted and comparison text reflects burden."""
    gdf, bivariate_df = data_context
    profile = compute_regency_profile(gdf, bivariate_df, 1101)
    sinyal = profile.indicators["SINYAL_BURUK"]
    assert sinyal.is_negative is True
    assert sinyal.effective_national_pct == round(100.0 - sinyal.national_pct, 2)
    assert sinyal.effective_corridor_pct == round(100.0 - sinyal.corridor_pct, 2)

    # Check inspector card html in EN and ID
    html_en = render_inspector_card_html(profile, lang="en")
    assert "Higher burden than" in html_en
    validator_en = HTMLTagValidator()
    validator_en.feed(html_en)
    assert validator_en.is_valid()

    html_id = render_inspector_card_html(profile, lang="id")
    assert "Tingkat keterbatasan/kerentanan lebih tinggi dari" in html_id
    validator_id = HTMLTagValidator()
    validator_id.feed(html_id)
    assert validator_id.is_valid()


def test_gap_aware_deficit_warning_badge(data_context):
    """Verify warning qualifier badge appears for R3-M3/R3-M2 with severe gap."""
    from src.analytics.bivariate import is_severe_preparedness_deficit
    gdf, bivariate_df = data_context
    profile_1310 = compute_regency_profile(gdf, bivariate_df, 1310)
    assert profile_1310.has_preparedness_deficit is True
    assert is_severe_preparedness_deficit(
        profile_1310.bivariate_category,
        profile_1310.gap_score,
        profile_1310.risk_score,
        profile_1310.mitigation_score,
    ) is True

    # In HTML, check amber badge
    html_en = render_inspector_card_html(profile_1310, lang="en")
    assert "border: 1px solid #f59e0b" in html_en
    assert "Mitigation Insufficient for Severe Risk (Preparedness Deficit)" in html_en
    validator_en = HTMLTagValidator()
    validator_en.feed(html_en)
    assert validator_en.is_valid()

    html_id = render_inspector_card_html(profile_1310, lang="id")
    assert "Mitigasi Belum Mengimbangi Risiko Ekstrem (Defisit Kesiapsiagaan)" in html_id
    validator_id = HTMLTagValidator()
    validator_id.feed(html_id)
    assert validator_id.is_valid()


def test_composite_mitigation_score_penalty_handling():
    """Verify composite mitigation treats negative indicators as penalties."""
    from src.analytics.bivariate import compute_composite_mitigation_score
    import pandas as pd
    test_df = pd.DataFrame({
        "EWS": [50.0, 50.0],
        "RAMBU": [50.0, 50.0],
        "PERKAP": [50.0, 50.0],
        "RAWAT_SUNGAI": [50.0, 50.0],
        "SINYAL_BURUK": [10.0, 90.0],
    })
    scores = compute_composite_mitigation_score(
        test_df,
        weights={"EWS": 0.5, "SINYAL_BURUK": 0.5},
        negative_indicators=["SINYAL_BURUK"],
    )
    # Unit 0 (10% bad signal) should have higher mitigation score than Unit 1 (90% bad signal)
    assert scores.iloc[0] > scores.iloc[1]


def test_ijed_2026_indicator_definitions():
    """Verify indicator naming and labels strictly match published IJED 2026 paper."""
    from src.app.i18n import get_indicator_label
    from src.analytics.profile import INDICATOR_METADATA

    # 1. PERKAP -> Safety Equipment / Perlengkapan Keselamatan
    assert get_indicator_label("PERKAP", "id") == "Perlengkapan Keselamatan"
    assert "Safety Equipment" in get_indicator_label("PERKAP", "en")
    assert "Regulasi" not in get_indicator_label("PERKAP", "id")
    assert "Regulation" not in get_indicator_label("PERKAP", "en")

    # 2. RAWAT_SUNGAI / RAWAT_AIR -> Perawatan Wilayah Perairan (Sungai/Kanal/Tanggul)
    assert get_indicator_label("RAWAT_SUNGAI", "id") == "Perawatan Wilayah Perairan (Sungai/Kanal/Tanggul)"
    assert get_indicator_label("RAWAT_SUNGAI", "en") == "Water Area Maintenance (Rivers/Canals/Dykes)"
    assert get_indicator_label("RAWAT_AIR", "id") == "Perawatan Wilayah Perairan (Sungai/Kanal/Tanggul)"
    assert get_indicator_label("RAWAT_AIR", "en") == "Water Area Maintenance (Rivers/Canals/Dykes)"

    # 3. ADA_BENCANA -> Kejadian Bencana (2023–2024)
    assert "Bencana" in get_indicator_label("ADA_BENCANA", "id")
    assert "Disaster" in get_indicator_label("ADA_BENCANA", "en")

    # 4. Metadata dictionary integrity
    assert "Safety Equipment" in INDICATOR_METADATA["PERKAP"]["label"]
    assert "Water Area Maintenance" in INDICATOR_METADATA["RAWAT_SUNGAI"]["label"]
    assert "ADA_BENCANA" in INDICATOR_METADATA

