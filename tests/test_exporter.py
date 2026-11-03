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
    generate_corridor_policy_brief,
)


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
    """Verify single-page A4 PDF brief generation for ID and EN locales."""
    profile = compute_regency_profile(loaded_data.gdf, loaded_data.bivariate_df, kode_wil=1101)
    spillover_res = estimate_spatial_spillover(loaded_data.gdf, weights=loaded_data.weights)

    # 1. Indonesian locale with RegencyProfile dataclass
    pdf_id = generate_district_policy_brief_pdf(profile, spatial_context=spillover_res, lang="id")
    assert isinstance(pdf_id, bytes)
    assert pdf_id.startswith(b"%PDF-")
    reader_id = pypdf.PdfReader(io.BytesIO(pdf_id))
    assert len(reader_id.pages) == 1, f"Expected strictly 1 page, got {len(reader_id.pages)}"

    # 2. English locale with dictionary representation
    pdf_en = generate_district_policy_brief_pdf(profile.to_dict(), spatial_context=spillover_res, lang="en")
    assert isinstance(pdf_en, bytes)
    assert pdf_en.startswith(b"%PDF-")
    reader_en = pypdf.PdfReader(io.BytesIO(pdf_en))
    assert len(reader_en.pages) == 1, f"Expected strictly 1 page, got {len(reader_en.pages)}"
