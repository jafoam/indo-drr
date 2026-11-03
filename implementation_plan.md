# Indonesian Disaster Readiness & Gap Explorer (IDR-GE) — Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify Milestone 1 of the Indonesian Disaster Readiness & Gap Explorer: repository scaffold, 514-regency data ingestion pipeline, PySAL spatial Moran's I analytics module, and a clean Streamlit + FastAPI interactive viewport.

**Architecture:** A FastAPI backend exposing spatial statistics, LISA clusters, and geo-data endpoints; a decoupled pure-data analytics library (GeoPandas + PySAL); and a high-density Streamlit spatial dashboard using Deck.gl/Plotly for no-slop visualization.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Streamlit, GeoPandas, Shapely, PySAL (`libpysal`, `esda`), Plotly, PyDeck, Pytest.

**Spec:** [DESIGN.md](file:///Users/rifizzu/disaster-resilience-nz-prep/DESIGN.md)

---

## Global Constraints

- **Python Runtime**: Use Python 3.11 (`.venv` based on `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11`).
- **Administrative Unit Count**: Exactly 514 regencies/cities (Kabupaten/Kota), matched on 4-digit BPS `KODE_WIL`. Zero unmatched or dropped units.
- **Coordinate Reference System**: EPSG:4326 (WGS84).
- **Anti-Slop Directives**: No generic purple/blue gradients, no decorative KPI cards, no floating glassmorphic panels. High data-density layout with 70% spatial viewport.
- **Tone/Style**: Terse, precise, zero ceremonial filler.

---

## Review Focus

1. **Disconnected Archipelago Islands**: Queen contiguity on archipelagic Indonesia yields zero-neighbor islands (islands with no land border), causing PySAL division-by-zero or singular spatial weight errors. Expected behavior: Island fallback to K-Nearest Neighbors ($k=5$) ensuring every regency has valid spatial connectivity.
2. **Missing or Inconsistent District Codes**: BPS codes may be formatted as floats (`1101.0`), strings (`"1101"`), or hierarchical (`"11.01"`). Expected behavior: Standardized integer `KODE_WIL` with strict validation against 514 known regencies.
3. **GeoJSON Payload Bloat**: Raw shapefile is 226MB, crashing browser webgl viewports. Expected behavior: Topologically simplified GeoJSON under 5MB retaining district boundaries and coastal fidelity.
4. **Zero-Variance / Edge Cases in Moran Permutations**: Indicators with identical values across multiple regions must not crash Monte Carlo LISA significance simulations. Expected behavior: Robust error trapping and p-value computation.
5. **Streamlit / FastAPI Headless Port Collision**: Local verification must bind to designated non-conflicting test ports (FastAPI on 8000, Streamlit on 8501) with clean background process teardown.

---

## Tasks

### Task 1.1: Directory Scaffolding & Virtualenv Dependency Setup
**Goal:** Scaffold clean repository structure and establish an isolated Python 3.11 virtual environment with pinned dependencies.

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Steps:**
- [x] Create `.gitignore` ignoring `.venv`, `__pycache__`, `.pytest_cache`, `.DS_Store`, and temporary test artifacts.
- [x] Write `requirements.txt` pinning `fastapi`, `uvicorn`, `streamlit`, `geopandas`, `shapely`, `libpysal`, `esda`, `plotly`, `pydeck`, `pytest`, `httpx`.
- [x] Initialize Python 3.11 virtualenv at `.venv` and install dependencies.
- [x] Create `src/config.py` defining central directories (`DATA_RAW_DIR`, `DATA_PROCESSED_DIR`), color tokens, and coordinate settings.
- [x] Write unit test `tests/test_env.py` asserting all critical packages import correctly and `src.config` exposes expected paths.
- [x] Verify test suite passes with `.venv/bin/pytest tests/test_env.py`.

---

### Task 1.2: Ingestion & Validation Pipeline (PODES CSV + Geometry)
**Goal:** Parse `Data/DATA_V#.csv` and shapefile `Data/GEODA/MITIGASI BENCANA.shp`, validate all 514 administrative units, simplify geometry to lightweight GeoJSON (<5MB), and persist clean parquet + geojson.

**Files:**
- Create: `src/ingestion/__init__.py`
- Create: `src/ingestion/parser.py`
- Create: `src/ingestion/validator.py`
- Create: `scripts/build_data.py`
- Create: `tests/test_ingestion.py`
- Target outputs: `data/processed/indonesia_regencies.geojson`, `data/processed/podes_irbi_clean.parquet`

**Steps:**
- [x] Write failing test `tests/test_ingestion.py` asserting:
  - Dataset contains exactly 514 rows.
  - No missing values in key columns (`KODE_WIL`, `IRBI_2024`, `EWS`, `RAMBU`, `PERKAP`).
  - Output GeoJSON CRS is EPSG:4326.
  - GeoJSON feature count equals 514 and file size is < 5MB.
- [x] Run `pytest tests/test_ingestion.py` to confirm failure before implementation.
- [x] Implement `src/ingestion/parser.py`:
  - Load `Data/DATA_V#.csv`.
  - Extract district names (`KAB_KOTA`), province names (`PROVINSI`), and codes (`KODE_WIL`) from DBF/shapefile metadata.
  - Read shapefile geometry, repair invalid topologies with `.buffer(0)` / `make_valid`, and simplify geometry (`tolerance=0.005`).
- [x] Implement `src/ingestion/validator.py`:
  - Enforce 514-unit completeness check.
  - Verify boundary coordinates fall within Indonesia bounds: Lat $[-11.5, 6.5]$, Lon $[95.0, 141.5]$.
- [x] Implement CLI builder script `scripts/build_data.py` generating `data/processed/indonesia_regencies.geojson` and `data/processed/podes_irbi_clean.parquet`.
- [x] Run `scripts/build_data.py` and verify `pytest tests/test_ingestion.py` passes with green results.

---

### Task 1.3: Core Spatial Analytics Module (Spatial Weights & Moran's I)
**Goal:** Compute spatial connectivity matrix for Indonesia's archipelagic geography, global univariate Moran's I, local LISA clusters, and composite risk vs mitigation gap scores.

**Files:**
- Create: `src/analytics/__init__.py`
- Create: `src/analytics/weights.py`
- Create: `src/analytics/moran.py`
- Create: `src/analytics/bivariate.py`
- Create: `tests/test_analytics.py`

**Steps:**
- [x] Write failing tests in `tests/test_analytics.py`:
  - Spatial weight matrix is symmetric, non-empty, and has no islands ($W$ has degree $> 0$ for all 514 units).
  - Global Moran's I computation returns valid statistic $I \in [-1, 1]$ and $p$-value $\in [0, 1]$ for `IRBI_2024` and `EWS`.
  - LISA calculation yields 514 cluster labels categorized into High-High, Low-Low, High-Low, Low-High, or Not-Significant.
  - Bivariate matrix assigns each regency into a $3 \times 3$ risk-mitigation cell.
- [x] Run `pytest tests/test_analytics.py` to confirm failure.
- [x] Implement `src/analytics/weights.py`:
  - Build KNN ($k=5$) or Queen-KNN hybrid spatial weight matrix from centroid geometries.
  - Apply row-standardization (`transform='R'`).
- [x] Implement `src/analytics/moran.py`:
  - `compute_morans_i(gdf, column, weights)` returning Moran's I, expected I, z-score, and p-value.
  - `compute_lisa(gdf, column, weights)` computing local spatial autocorrelation and mapping quadrants to labels.
- [x] Implement `src/analytics/bivariate.py`:
  - Calculate normalized Composite Risk Score ($R_i$), Mitigation Score ($M_i$), and Gap Score ($G_i$).
  - Segment $R_i$ and $M_i$ into terciles to produce the $3 \times 3$ category matrix and assign hex color codes from `DESIGN.md`.
- [x] Run `pytest tests/test_analytics.py` and verify all tests pass.

---

### Task 1.4: Minimal Streamlit Viewport & FastAPI Integration
**Goal:** Implement FastAPI endpoints for spatial datasets & Moran metrics, and build a streamlined, antislop-compliant Streamlit viewport verifying choropleth rendering and spatial stats.

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/main.py`
- Create: `src/api/routes.py`
- Create: `src/app/__init__.py`
- Create: `src/app/app.py`
- Create: `src/app/components/choropleth.py`
- Create: `src/app/components/stats_panel.py`
- Create: `tests/test_api.py`

**Steps:**
- [x] Write failing tests in `tests/test_api.py`:
  - `GET /health` returns status `200 ok`.
  - `GET /api/v1/spatial/regencies` returns 514-feature GeoJSON or summary.
  - `GET /api/v1/analytics/moran?variable=IRBI_2024` returns Moran's I and p-value.
  - `GET /api/v1/analytics/bivariate` returns bivariate matrix classification for 514 units.
- [x] Run `pytest tests/test_api.py` to confirm failure.
- [x] Implement `src/api/routes.py` and `src/api/main.py` with FastAPI, loading cached parquet and spatial weight matrix at startup.
- [x] Verify `pytest tests/test_api.py` passes.
- [x] Implement `src/app/components/choropleth.py`:
  - PyDeck / Plotly Choropleth map with bivariate 3x3 color mapping and LISA cluster layers.
  - Clean tooltip displaying: Regency name, Province, Risk Score, Mitigation Score, and Gap Classification.
  - Strictly adhering to visual tokens in `DESIGN.md` (no generic blue-purple glow).
- [x] Implement `src/app/components/stats_panel.py`:
  - Compact Moran scatterplot (lagged variable vs standardized variable).
  - Global Moran's I readout ($I$, $z$-score, $p$-value).
  - Critical Gap district ranking table (Top 10 High Risk / Low Mitigation regencies).
- [x] Implement `src/app/app.py` Streamlit entrypoint connecting to FastAPI or local analytics engine with instant interactive filtering.

---

### Task 1.5: End-to-End Delivery Verification Gate
**Goal:** Execute full test suite, verify clean startup of backend & frontend, validate memory/render performance, and ensure zero antislop violations.

**Files:**
- Create: `scripts/verify_milestone_1.py`

**Steps:**
- [x] Run full test suite: `.venv/bin/pytest tests/ -v`.
- [x] Run `scripts/verify_milestone_1.py`:
  - Confirm 514 regencies intact with non-null geometry and indices.
  - Validate Moran's I computation speed (< 2.0s for 514 units).
  - Confirm GeoJSON file size < 5MB.
  - Launch FastAPI test instance, perform curl checks, teardown cleanly.
  - Perform headless Streamlit dry-run script test.
- [x] Confirm compliance with `antislop` guidelines:
  - Check absence of generic AI aesthetic tokens (no gradient borders, no glass cards, no ungrounded metric counters).
- [x] Output Milestone 1 completion receipt.
