# Indonesian Disaster Readiness & Gap Explorer (IDR-GE) — Milestone 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Milestone 2 features for IDR-GE: (A) Interactive Policy Intervention Sandbox for digital & social resilience, (B) Interactive LISA Spatial Cluster Map Toggle with dynamic symbology, and (C) Dynamic Regional Corridor Filter with localized spatial autocorrelation recalculation.

**Architecture:** Extend FastAPI backend with simulation and corridor-sliced spatial analytics; extend Streamlit frontend with a high-density policy parameter sandbox, zero-latency layer switching, and localized spatial weight recalculation for archipelagic regional corridors.

**Tech Stack:** Python 3.11, FastAPI, Streamlit, GeoPandas, PySAL (`libpysal`, `esda`), Plotly, PyDeck, Pytest.

**Spec:** [DESIGN.md](file:///Users/rifizzu/disaster-resilience-nz-prep/DESIGN.md)

---

## Global Constraints

- **Execution Runtime**: Python 3.11 (`.venv`).
- **Data Integrity**: 514 administrative units remain fully accounted for across all corridor filters and simulations.
- **Archipelagic Connectivity**: Regional corridor sub-graphs must dynamically bound $k$-NN ($k = \min(5, N_{\text{sub}} - 1)$) to guarantee zero isolated island nodes.
- **Anti-Slop Directives**: No generic glowing card dashboards, no bloated rounded slider skins, matte dark GIS styling matching `COLOR_TOKENS` from `src.config`.
- **Latency Budget**: Policy simulation & localized Moran recalculation must execute in $< 100\text{ ms}$ for real-time slider responsiveness.

---

## Review Focus

1. **Sub-regional Island Singularity**: Corridors like Nusa Tenggara or Maluku with fewer regencies and archipelagic separation must not produce singular or zero-neighbor weights matrices.
2. **Boundary Inflation in Simulation**: Interventions in `SINYAL_BURUK` (-%) and `GOTONG_ROYONG` (+%) must be normalized and bounded strictly within $[0, 1]$ mitigation score scales without score overflow.
3. **Dual-Layer Tooltip Synchronization**: Switching between 3x3 Bivariate and LISA cluster layers must cleanly update PyDeck tooltip keys without JavaScript/HTML attribute lookup errors.
4. **Dynamic Plotly Axis Re-anchoring**: Localized corridor scatterplots must dynamically rescale standardized $z$ and spatial lag $Wz$ to the filtered sub-population, not the national mean.
5. **State Bleed Between Simulation & Baseline**: Resetting sliders to zero must reproduce baseline statistics (54 Critical Gap units) with exact numerical identity ($0.0000$ variance).

---

## Regional Corridor Mapping

Regional units grouped into 7 analytical corridors across the 38 provinces:
- **Sumatera**: 154 regencies/cities (10 provinces)
- **Jawa-Bali**: 128 regencies/cities (7 provinces)
- **Kalimantan**: 56 regencies/cities (5 provinces)
- **Sulawesi**: 81 regencies/cities (6 provinces)
- **Nusa Tenggara**: 32 regencies/cities (2 provinces: NTB, NTT)
- **Maluku**: 21 regencies/cities (2 provinces: Maluku, Maluku Utara)
- **Papua**: 42 regencies/cities (6 provinces)
*Total: 514 administrative units.*

---

## Tasks

### Task 2.1: Analytics Engine Extensions (Policy Simulation & Corridor Sub-Weights)
**Goal:** Implement policy simulation math and sub-regional spatial weights matrix generator with localized Moran's I.

**Files:**
- Create: `src/analytics/simulation.py`
- Create: `src/analytics/corridors.py`
- Modify: `src/analytics/__init__.py`
- Create: `tests/test_simulation.py`

**Steps:**
- [x] Write failing test `tests/test_simulation.py`:
  - `simulate_policy_intervention(bivariate_df, signal_reduction_pct, gotong_royong_boost_pct)`:
    * Asserts simulated mitigation scores remain in $[0, 1]$.
    * Asserts zero intervention reproduces exactly baseline 54 Critical Gap units.
    * Asserts 50% signal reduction and 25% gotong royong boost reduces Critical Gap units monotonically.
  - `get_corridor_subgraph(gdf, corridor_name)`:
    * Asserts each of 7 corridors returns valid GeoDataFrame and row counts sum to 514.
    * Generates valid $W_{sub}$ with zero islands for all 7 corridors.
    * Calculates localized Moran's I and p-value for `IRBI_2024` on corridor subset.
- [x] Run `.venv/bin/pytest tests/test_simulation.py` to confirm failure (RED).
- [x] Implement `src/analytics/simulation.py`:
  - `simulate_policy_intervention(df, signal_reduction_pct, gotong_royong_boost_pct)`:
    * Adjusts `SINYAL_BURUK` downwards and `GOTONG_ROYONG` upwards.
    * Recalculates $M_i^{adj} = M_i + 0.15 \cdot \text{Norm}(\Delta_{\text{signal}}) + 0.15 \cdot \text{Norm}(\Delta_{\text{gotong\_royong}})$.
    * Recalculates $G_i^{adj} = R_i - M_i^{adj}$ and assigns simulated 3x3 category.
    * Returns summary dictionary and augmented DataFrame.
- [x] Implement `src/analytics/corridors.py`:
  - Province-to-corridor registry covering all 38 provinces.
  - `get_corridor_weights(corridor_gdf)` building localized $k$-NN weights.
  - `compute_corridor_moran(corridor_gdf, variable, weights)` computing localized statistics.
- [x] Export new functions in `src/analytics/__init__.py`.
- [x] Run `.venv/bin/pytest tests/test_simulation.py` and verify GREEN.

---

### Task 2.2: FastAPI Endpoints for Simulation & Corridor Queries
**Goal:** Add backend REST API endpoints for policy intervention simulation and corridor-sliced spatial analytics.

**Files:**
- Modify: `src/api/routes.py`
- Modify: `src/api/main.py`
- Create: `tests/test_api_milestone_2.py`

**Steps:**
- [x] Write failing test `tests/test_api_milestone_2.py`:
  - `POST /api/v1/analytics/simulate` with payload `{"signal_reduction": 30.0, "gotong_royong_boost": 20.0}` returns 200 with baseline count, simulated count, resolved count, and 514 simulated units.
  - `GET /api/v1/spatial/corridors` returns list of 7 corridors with member provinces and unit counts.
  - `GET /api/v1/analytics/corridor/moran?corridor=Sumatera&variable=IRBI_2024` returns localized Moran's I and p-value for Sumatera.
- [x] Run `.venv/bin/pytest tests/test_api_milestone_2.py` to confirm failure (RED).
- [x] Implement endpoints in `src/api/routes.py`:
  - Pydantic models for simulation request and response schemas.
  - Corridor list and corridor Moran computation endpoints.
- [x] Run `.venv/bin/pytest tests/test_api_milestone_2.py` and verify GREEN.

---

### Task 2.3: Streamlit Policy Sandbox & Dual-Layer Viewport Components
**Goal:** Build the interactive policy slider panel and enhance the choropleth component with dual-layer toggle and simulation overlays.

**Files:**
- Create: `src/app/components/sandbox_panel.py`
- Modify: `src/app/components/choropleth.py`
- Modify: `src/app/components/stats_panel.py`
- Create: `tests/test_sandbox_components.py`

**Steps:**
- [x] Write failing test `tests/test_sandbox_components.py`:
  - `render_simulation_delta_html(summary)` produces clean 0-indent markup with baseline, simulated, and net resolved metrics.
  - `prepare_choropleth_data` accepts optional `simulated_df` and attaches `simulated_gap_score` and `simulated_category`.
  - PyDeck tooltip reflects active layer mode and simulation delta.
- [x] Run `.venv/bin/pytest tests/test_sandbox_components.py` to confirm failure (RED).
- [x] Implement `src/app/components/sandbox_panel.py`:
  - Compact slider controls for Telecom Improvement % (`0% - 100%`) and Gotong Royong Mobilization % (`0% - 100%`).
  - Monospace readout card displaying: `Baseline Critical Gaps: 54`, `Simulated: X`, `Resolved: Y`.
- [x] Enhance `src/app/components/choropleth.py`:
  - Dynamic tooltip showing baseline vs simulated gap when policy simulation active.
  - Smooth toggle between Bivariate 3x3 and LISA layers with zero raw HTML leaks.
- [x] Enhance `src/app/components/stats_panel.py`:
  - Add corridor-aware OLS regression slope comparison.
- [x] Run `.venv/bin/pytest tests/test_sandbox_components.py` and verify GREEN.

---

### Task 2.4: Viewport Integration & Milestone 2 Verification Gate
**Goal:** Wire all components into `src/app/app.py`, execute end-to-end smoke tests, verify anti-slop compliance, and build Milestone 2 verification script.

**Files:**
- Modify: `src/app/app.py`
- Create: `scripts/verify_milestone_2.py`
- Create: `tests/test_milestone_2_gate.py`

**Steps:**
- [x] Write failing test `tests/test_milestone_2_gate.py` asserting `scripts/verify_milestone_2.py` runs and returns 0.
- [x] Run `.venv/bin/pytest tests/test_milestone_2_gate.py` to confirm failure (RED).
- [x] Update `src/app/app.py`:
  - Add Sidebar expandable "Policy Intervention Sandbox" with real-time sliders.
  - Add Regional Corridor dropdown alongside Province selector.
  - When corridor selected: dynamically zoom map to corridor centroid, filter choropleth, and compute localized Moran's I on the right panel.
- [x] Implement `scripts/verify_milestone_2.py`:
  - Validates all 7 corridors sum to 514 units and have non-singular weights.
  - Benchmarks simulation latency ($< 50\text{ ms}$).
  - Tests live FastAPI endpoints for simulation and corridor analytics.
  - Verifies headless Streamlit app instantiation with policy sliders active.
  - Performs static anti-slop audit across all modified files.
- [x] Run `.venv/bin/pytest tests/test_milestone_2_gate.py` and verify GREEN.
- [x] Run full test suite: `.venv/bin/pytest tests/ -v`.
- [x] Run `scripts/verify_milestone_2.py` and output Milestone 2 completion receipt.
