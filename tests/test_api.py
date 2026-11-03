"""FastAPI integration tests for IDR-GE backend."""
import pytest
from starlette.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a TestClient instance for the FastAPI application."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test GET /health returns status 200 and expected schema."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "units": 514}


def test_spatial_regencies_endpoint(client):
    """Test GET /api/v1/spatial/regencies returns status 200 with 514 features."""
    response = client.get("/api/v1/spatial/regencies")
    assert response.status_code == 200
    data = response.json()
    assert data.get("type") == "FeatureCollection"
    features = data.get("features", [])
    assert len(features) == 514

    # Verify first feature structure
    first = features[0]
    assert "properties" in first
    assert "geometry" in first
    assert "KODE_WIL" in first["properties"]
    assert "KAB_KOTA" in first["properties"]


def test_moran_analytics_endpoint(client):
    """Test GET /api/v1/analytics/moran?variable=IRBI_2024 returns status 200 with Moran metrics."""
    response = client.get("/api/v1/analytics/moran?variable=IRBI_2024")
    assert response.status_code == 200
    data = response.json()
    assert "moran_i" in data
    assert "p_value" in data
    assert "z_score" in data
    assert "expected_i" in data
    assert -1.0 <= data["moran_i"] <= 1.0
    assert 0.0 <= data["p_value"] <= 1.0


def test_moran_analytics_invalid_variable(client):
    """Test GET /api/v1/analytics/moran with invalid variable returns 400 or 422."""
    response = client.get("/api/v1/analytics/moran?variable=NON_EXISTENT_VAR")
    assert response.status_code in (400, 422)


def test_bivariate_analytics_endpoint(client):
    """Test GET /api/v1/analytics/bivariate returns status 200 with 514 items."""
    response = client.get("/api/v1/analytics/bivariate")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 514

    for item in data[:10]:
        assert "KODE_WIL" in item
        assert "risk_score" in item
        assert "mitigation_score" in item
        assert "gap_score" in item
        assert "bivariate_category" in item
        assert "bivariate_color" in item
        assert item["bivariate_color"].startswith("#")
