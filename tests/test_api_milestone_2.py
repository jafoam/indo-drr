"""FastAPI integration tests for Milestone 2 (Simulation & Corridor Analytics)."""
import pytest
from starlette.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a TestClient instance for the FastAPI application."""
    return TestClient(app)


def test_simulate_endpoint_active(client):
    """POST /api/v1/analytics/simulate with active parameters returns 200 OK and expected schema."""
    payload = {"signal_reduction": 30.0, "gotong_royong_boost": 20.0}
    response = client.post("/api/v1/analytics/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "baseline_critical_gaps" in data
    assert "simulated_critical_gaps" in data
    assert "resolved_gaps" in data
    assert "units" in data

    assert data["baseline_critical_gaps"] == 148
    assert data["simulated_critical_gaps"] < 148
    assert data["resolved_gaps"] > 0
    assert data["resolved_gaps"] == data["baseline_critical_gaps"] - data["simulated_critical_gaps"]

    units = data["units"]
    assert len(units) == 514

    first = units[0]
    required_keys = {
        "KODE_WIL",
        "simulated_gap_score",
        "simulated_bivariate_category",
        "simulated_bivariate_color",
        "is_critical_gap_simulated",
    }
    assert required_keys.issubset(set(first.keys()))
    assert isinstance(first["KODE_WIL"], int)
    assert -1.0 <= first["simulated_gap_score"] <= 1.0
    assert first["simulated_bivariate_color"].startswith("#")
    assert isinstance(first["is_critical_gap_simulated"], bool)


def test_simulate_endpoint_defaults(client):
    """POST /api/v1/analytics/simulate with empty payload {} returns 200 OK with resolved_gaps == 0."""
    response = client.post("/api/v1/analytics/simulate", json={})
    assert response.status_code == 200

    data = response.json()
    assert data["baseline_critical_gaps"] == 148
    assert data["simulated_critical_gaps"] == 148
    assert data["resolved_gaps"] == 0
    assert len(data["units"]) == 514


def test_simulate_endpoint_invalid_payload(client):
    """POST /api/v1/analytics/simulate with invalid bounds returns 422 Unprocessable Entity."""
    # Negative signal reduction
    res = client.post("/api/v1/analytics/simulate", json={"signal_reduction": -5.0})
    assert res.status_code == 422

    # Signal reduction > 100
    res = client.post("/api/v1/analytics/simulate", json={"signal_reduction": 105.0})
    assert res.status_code == 422

    # Negative gotong royong boost
    res = client.post("/api/v1/analytics/simulate", json={"gotong_royong_boost": -1.0})
    assert res.status_code == 422

    # Gotong royong boost > 100
    res = client.post("/api/v1/analytics/simulate", json={"gotong_royong_boost": 150.0})
    assert res.status_code == 422


def test_corridors_endpoint(client):
    """GET /api/v1/spatial/corridors returns 200 OK with 7 regional corridors summing to 514 units."""
    response = client.get("/api/v1/spatial/corridors")
    assert response.status_code == 200

    corridors = response.json()
    assert isinstance(corridors, list)
    assert len(corridors) == 7

    expected_names = {
        "Sumatera",
        "Jawa-Bali",
        "Kalimantan",
        "Sulawesi",
        "Nusa Tenggara",
        "Maluku",
        "Papua",
    }
    actual_names = {c["name"] for c in corridors}
    assert actual_names == expected_names

    total_units = 0
    for corridor in corridors:
        assert "name" in corridor
        assert "unit_count" in corridor
        assert "provinces" in corridor
        assert isinstance(corridor["unit_count"], int)
        assert corridor["unit_count"] > 0
        assert isinstance(corridor["provinces"], list)
        assert len(corridor["provinces"]) > 0
        total_units += corridor["unit_count"]

    assert total_units == 514


def test_corridor_moran_endpoint(client):
    """GET /api/v1/analytics/corridor/moran returns 200 OK with localized Moran stats for Sumatera."""
    response = client.get("/api/v1/analytics/corridor/moran?corridor=Sumatera&variable=IRBI_2024")
    assert response.status_code == 200

    data = response.json()
    assert data["corridor"] == "Sumatera"
    assert data["variable"] == "IRBI_2024"
    assert data["unit_count"] == 154
    assert "moran_i" in data
    assert "p_value" in data
    assert "z_score" in data
    assert -1.0 <= data["moran_i"] <= 1.0
    assert 0.0 <= data["p_value"] <= 1.0
    assert isinstance(data["z_score"], float)


def test_corridor_moran_not_found(client):
    """GET /api/v1/analytics/corridor/moran for non-existent corridor returns 404 Not Found."""
    response = client.get("/api/v1/analytics/corridor/moran?corridor=Atlantis&variable=IRBI_2024")
    assert response.status_code == 404
    assert "Atlantis" in response.json()["detail"]
