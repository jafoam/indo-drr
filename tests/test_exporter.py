"""Unit tests for Policy Brief Exporter (Milestone 3 Feature C)."""
import io
import pypdf
import pytest
import geopandas as gpd
import pandas as pd

from src.api.routes import get_data_manager
from src.analytics.profile import compute_regency_profile
from src.analytics.spillover import estimate_spatial_spillover
from src.analytics.exporter import (
    generate_district_policy_brief,
    generate_district_policy_brief_pdf,
    generate_provincial_policy_brief_pdf,
    generate_corridor_policy_brief,
)
from src.analytics.provincial import compute_provincial_profile
from src.app.components.inspector import render_radar_chart


@pytest.fixture(scope="module")
def loaded_data():
    """Fixture providing initialized DataManager with 514 units."""
    dm = get_data_manager()
    dm.load()
    return dm


def test_generate_district_policy_brief(loaded_data):
    """Verify district policy brief markdown generation and anti-slop rules."""
    profile = compute_regency_profile(loaded_data.gdf, loaded_data.bivariate_df, kode_wil=1101)
    spillover_res = estimate_spatial_spillover(loaded_data.gdf, weights=loaded_data.weights)

    brief_md = generate_district_policy_brief(profile, spillover_res)
    assert isinstance(brief_md, str)
    assert len(brief_md) > 200

    # Required sections and content
    assert "# Executive Disaster Policy Brief" in brief_md
    assert "Simeulue" in brief_md
    assert "Aceh" in brief_md
    assert "Sumatera" in brief_md
    assert "1101" in brief_md
    assert "Bivariate Quadrant" in brief_md
    assert "PODES Micro-Indicator Diagnostic" in brief_md
    assert "| Indicator |" in brief_md
    assert "Spatial Spillover Context" in brief_md
    assert "Actionable Policy Interventions" in brief_md

    # Check that indicators are included in table
    for ind in ["EWS", "RAMBU", "PERKAP", "RAWAT_SUNGAI", "GOTONG_ROYONG", "SINYAL_BURUK", "IPM"]:
        assert ind in brief_md

    # Anti-slop check: no generic AI filler phrases
    banned_phrases = [
        "delve",
        "tapestry",
        "testament to",
        "in conclusion, it is crucial",
        "it is important to remember",
    ]
    for phrase in banned_phrases:
        assert phrase not in brief_md.lower(), f"Found banned slop phrase: {phrase}"


def test_generate_district_policy_brief_indonesian(loaded_data):
    """Verify Indonesian localization of district policy brief."""
    profile = compute_regency_profile(loaded_data.gdf, loaded_data.bivariate_df, kode_wil=1101)
    spillover_res = estimate_spatial_spillover(loaded_data.gdf, weights=loaded_data.weights)

    brief_id = generate_district_policy_brief(profile, spillover_res, lang="id")
    assert isinstance(brief_id, str)
    assert "# RINGKASAN KEBIJAKAN EKSEKUTIF KESIAPSIAGAAN BENCANA" in brief_id
    assert "Profil Wilayah" in brief_id
    assert "Evaluasi Kesenjangan Risiko vs Mitigasi" in brief_id
    assert "Temuan Tolok Ukur Mikro" in brief_id
    assert "Rekomendasi Intervensi Prioritas" in brief_id
    assert "Sistem Peringatan Dini (EWS)" in brief_id
    assert "Defisit" in brief_id or "Memadai" in brief_id
    assert "| Kode | Indikator | Nilai |" in brief_id


def test_generate_corridor_policy_brief(loaded_data):
    """Verify regional corridor policy brief markdown generation."""
    spillover_res = estimate_spatial_spillover(
        loaded_data.gdf, weights=loaded_data.weights, corridor="Sumatera"
    )
    brief_md = generate_corridor_policy_brief(
        corridor_name="Sumatera",
        gdf=loaded_data.gdf,
        bivariate_df=loaded_data.bivariate_df,
        spillover_res=spillover_res,
    )
    assert isinstance(brief_md, str)
    assert "# Regional Corridor Disaster Policy Brief: Sumatera" in brief_md
    assert "154" in brief_md
    assert "Top Critical Gap Districts" in brief_md
    assert "Cross-Boundary Spillover Dynamics" in brief_md
    assert "| Regency / City |" in brief_md


def test_generate_district_policy_brief_pdf(loaded_data):
    """Verify single-page A4 PDF brief generation for ID and EN locales with radar chart."""
    profile = compute_regency_profile(loaded_data.gdf, loaded_data.bivariate_df, kode_wil=1101)
    spillover_res = estimate_spatial_spillover(loaded_data.gdf, weights=loaded_data.weights)
    radar_fig = render_radar_chart(profile, is_bright=True)

    # 1. Indonesian locale with RegencyProfile dataclass and embedded radar figure
    pdf_id = generate_district_policy_brief_pdf(
        profile, spatial_context=spillover_res, lang="id", radar_fig=radar_fig
    )
    assert isinstance(pdf_id, bytes)
    assert pdf_id.startswith(b"%PDF-")
    reader_id = pypdf.PdfReader(io.BytesIO(pdf_id))
    assert len(reader_id.pages) == 1, f"Expected strictly 1 page, got {len(reader_id.pages)}"

    # Verify extracted text contains localized status and indicators
    extracted_text_id = reader_id.pages[0].extract_text()
    assert "Simeulue" in extracted_text_id
    assert any(term in extracted_text_id for term in ["Tertinggal", "Di Bawah Median", "Cukup Baik"])

    # 2. English locale with dictionary representation and embedded radar figure
    pdf_en = generate_district_policy_brief_pdf(
        profile.to_dict(), spatial_context=spillover_res, lang="en", radar_fig=radar_fig
    )
    assert isinstance(pdf_en, bytes)
    assert pdf_en.startswith(b"%PDF-")
    reader_en = pypdf.PdfReader(io.BytesIO(pdf_en))
    assert len(reader_en.pages) == 1, f"Expected strictly 1 page, got {len(reader_en.pages)}"

    extracted_text_en = reader_en.pages[0].extract_text()
    assert "Simeulue" in extracted_text_en
    assert any(term in extracted_text_en for term in ["Deficit", "Below Median", "Adequate"])


def test_compute_provincial_profile(loaded_data):
    """Verify provincial aggregation engine calculations."""
    prof = compute_provincial_profile(loaded_data.gdf, "Aceh")
    assert prof["provinsi"] == "Aceh"
    assert prof["corridor"] == "Sumatera"
    assert prof["total_units"] == 23
    assert 0.0 <= prof["risk_score"] <= 1.0
    assert 0.0 <= prof["mitigation_score"] <= 1.0
    assert len(prof["top_bottlenecks"]) <= 5
    assert prof["top_bottlenecks"][0]["gap_score"] >= prof["top_bottlenecks"][-1]["gap_score"]
    assert "EWS" in prof["indicators"]
    assert "IPM" in prof["indicators"]


def test_generate_provincial_policy_brief_pdf(loaded_data):
    """Verify single-page A4 provincial PDF brief generation for ID and EN locales."""
    prof = compute_provincial_profile(loaded_data.gdf, "Aceh")

    # 1. Indonesian locale
    pdf_id = generate_provincial_policy_brief_pdf(prof, lang="id")
    assert isinstance(pdf_id, bytes)
    assert pdf_id.startswith(b"%PDF-")
    reader_id = pypdf.PdfReader(io.BytesIO(pdf_id))
    assert len(reader_id.pages) == 1, f"Expected strictly 1 page, got {len(reader_id.pages)}"

    extracted_id = reader_id.pages[0].extract_text()
    assert "Aceh" in extracted_id
    assert "PROVINSI" in extracted_id
    assert "GUBERNUR" in extracted_id.upper() or "BPBD" in extracted_id.upper()

    # 2. English locale
    pdf_en = generate_provincial_policy_brief_pdf(prof, lang="en")
    assert isinstance(pdf_en, bytes)
    assert pdf_en.startswith(b"%PDF-")
    reader_en = pypdf.PdfReader(io.BytesIO(pdf_en))
    assert len(reader_en.pages) == 1, f"Expected strictly 1 page, got {len(reader_en.pages)}"

    extracted_en = reader_en.pages[0].extract_text()
    assert "Aceh" in extracted_en
    assert "PROVINCE" in extracted_en
    assert "GOVERNOR" in extracted_en.upper() or "BPBD" in extracted_en.upper()

