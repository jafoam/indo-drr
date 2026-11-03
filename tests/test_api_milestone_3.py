"""FastAPI integration tests for Milestone 3 (District Profile & Spillover Analysis)."""
import pytest
from starlette.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a TestClient instance for the FastAPI application."""
    return TestClient(app)


def test_regency_profile_simeulue(client):
    """GET /api/v1/spatial/regencies/1101/profile returns 200 with valid profile and 7 indicators."""
    response = client.get("/api/v1/spatial/regencies/1101/profile")
    assert response.status_code == 200

    data = response.json()
    assert data["kode_wil"] == 1101
    assert data["kab_kota"] == "Simeulue"
    assert data["provinsi"] == "Aceh"
    assert data["corridor"] == "Sumatera"
    assert "risk_score" in data
    assert "mitigation_score" in data
    assert "gap_score" in data
    assert "gap_rank" in data
    assert "bivariate_category" in data
    assert "indicators" in data

    indicators = data["indicators"]
    expected_indicators = {
        "EWS",
        "RAMBU",
        "PERKAP",
        "RAWAT_SUNGAI",
        "GOTONG_ROYONG",
        "SINYAL_BURUK",
        "IPM",
    }
    assert set(indicators.keys()) == expected_indicators

    for ind in expected_indicators:
        item = indicators[ind]
        assert "name" in item
        assert "label" in item
        assert "value" in item
        assert "national_pct" in item
        assert "corridor_pct" in item
        assert "national_median" in item
        assert "corridor_median" in item
        assert 0.0 <= item["national_pct"] <= 100.0
        assert 0.0 <= item["corridor_pct"] <= 100.0


def test_regency_profile_jakarta_selatan(client):
    """GET /api/v1/spatial/regencies/3171/profile returns 200 with valid Jakarta Selatan profile."""
    response = client.get("/api/v1/spatial/regencies/3171/profile")
    assert response.status_code == 200

    data = response.json()
    assert data["kode_wil"] == 3171
    assert data["provinsi"] == "DKI Jakarta"
    assert data["corridor"] == "Jawa-Bali"
    assert "Jakarta" in data["kab_kota"]
    assert len(data["indicators"]) == 7


def test_regency_profile_not_found(client):
    """GET /api/v1/spatial/regencies/9999/profile returns 404 Not Found with error message."""
    response = client.get("/api/v1/spatial/regencies/9999/profile")
    assert response.status_code == 404
    detail = response.json().get("detail", "")
    assert "9999" in str(detail) or "not found" in str(detail).lower()


def test_spillover_default_query(client):
    """GET /api/v1/analytics/spillover default query returns 200 with national spatial lag model."""
    response = client.get("/api/v1/analytics/spillover")
    assert response.status_code == 200

    data = response.json()
    assert data["model_type"] == "lag"
    assert data["n_obs"] == 514
    assert "rho" in data
    assert isinstance(data["rho"], float)
    assert 0.2 < data["rho"] < 0.5  # float ~0.36
    assert "effects" in data
    effects = data["effects"]
    assert len(effects) == 3
    var_names = [e["variable"] for e in effects]
    assert set(var_names) == {"SINYAL_BURUK", "GOTONG_ROYONG", "IPM"}

    for effect in effects:
        assert "direct_effect" in effect
        assert "indirect_effect" in effect
        assert "total_effect" in effect
        assert "spillover_ratio" in effect
        assert "p_value" in effect
        assert "interpretation" in effect
        assert isinstance(effect["interpretation"], str)
        assert len(effect["interpretation"]) > 10


def test_spillover_corridor_sumatera(client):
    """GET /api/v1/analytics/spillover?corridor=Sumatera returns 200 with n_obs=154."""
    response = client.get("/api/v1/analytics/spillover?corridor=Sumatera")
    assert response.status_code == 200

    data = response.json()
    assert data["corridor"] == "Sumatera"
    assert data["n_obs"] == 154
    assert len(data["effects"]) == 3


def test_spillover_invalid_corridor(client):
    """GET /api/v1/analytics/spillover?corridor=InvalidCorridor returns 404 Not Found."""
    response = client.get("/api/v1/analytics/spillover?corridor=InvalidCorridor")
    assert response.status_code == 404
    assert "InvalidCorridor" in response.json()["detail"]


def test_spillover_model_type_slx(client):
    """GET /api/v1/analytics/spillover?model_type=slx returns 200 with model_type='slx'."""
    response = client.get("/api/v1/analytics/spillover?model_type=slx")
    assert response.status_code == 200

    data = response.json()
    assert data["model_type"] == "slx"
    assert data["n_obs"] == 514
    assert data["rho"] is None
    assert len(data["effects"]) == 3


def test_spillover_corridor_national(client):
    """GET /api/v1/analytics/spillover?corridor=National returns 200 with national 514 units."""
    response = client.get("/api/v1/analytics/spillover?corridor=National")
    assert response.status_code == 200
    data = response.json()
    assert data["n_obs"] == 514
    assert data["model_type"] == "lag"


def test_spillover_invalid_model_type(client):
    """GET /api/v1/analytics/spillover?model_type=ols returns 400 Bad Request."""
    response = client.get("/api/v1/analytics/spillover?model_type=ols")
    assert response.status_code == 400
    assert "Invalid model_type" in response.json()["detail"]


def test_spillover_invalid_y_var(client):
    """GET /api/v1/analytics/spillover?y_var=non_existent returns 400 Bad Request."""
    response = client.get("/api/v1/analytics/spillover?y_var=non_existent")
    assert response.status_code == 400


def test_regency_profile_invalid_type(client):
    """GET /api/v1/spatial/regencies/abc/profile returns 422 Unprocessable Entity."""
    response = client.get("/api/v1/spatial/regencies/abc/profile")
    assert response.status_code == 422

