# Indonesian Disaster Readiness & Gap Explorer (IDR-GE) — Milestone 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Milestone 3 for IDR-GE: (A) Micro-Profile District Inspector with 7-indicator radar breakdown and national/corridor percentile benchmarking, and (B) Spatial Spillover Analytics Module estimating direct, indirect, and total cross-boundary effects via PySAL `spreg`.

**Architecture:** Extend FastAPI backend with district micro-profile and spatial spillover regression endpoints; extend Streamlit viewport with a dark matte radar inspection drawer and an interactive cross-boundary spillover analysis panel.

**Tech Stack:** Python 3.11, FastAPI, Streamlit, GeoPandas, PySAL (`libpysal`, `esda`, `spreg`), Plotly, PyDeck, Pytest.

**Spec:** [DESIGN.md](file:///Users/rifizzu/disaster-resilience-nz-prep/DESIGN.md)

---

## Global Constraints

- **Execution Runtime**: Python 3.11 (`.venv`).
- **Data Integrity**: All 514 administrative units must support micro-profiling and 100% valid percentile computations.
- **Spatial Econometrics Robustness**: Spatial lag regressions must use row-standardized weights ($W_{\text{std}}$) ensuring matrix eigenvalues satisfy $|\lambda| \le 1$ and power expansions converge.
- **Anti-Slop Directives**: Matte dark GIS palette (`COLOR_TOKENS`), zero glowing borders or generic progress rings, strict 0-indent markup in all HTML snippets, high data density.
- **Latency Budget**: Micro-profile generation $< 10\text{ ms}$; Spatial lag regression $< 500\text{ ms}$.

---

## Review Focus

1. **Percentile Edge Ties**: Regencies with identical indicator values (e.g. 0.0% tsunami EWS) must use consistent rank averaging (`method='average'`) without producing out-of-bound percentiles ($[0.0, 100.0]$).
2. **Spatial Multiplier Convergence**: When estimating SAR/SLX models, ensure spatial autoregressive parameter $\rho \in (-1, 1)$ to guarantee finite spillover multipliers.
3. **Multicollinearity in Regressors**: Correlation between `SINYAL_BURUK`, `GOTONG_ROYONG`, and `IPM` must be monitored via variance inflation factors or condition numbers in `spreg`.
4. **Interactive District Selection State**: Streamlit selectbox and map click synchronization must not trigger infinite reruns or desynchronize active corridor filters.
5. **No HTML Code Block Leaks**: Radar annotations, district drawer header, and spillover summary cards must be strictly validated against leading 4-space indentations.

---

## Tasks

### Task 3.1: Micro-Profile District Inspector Module & Radar Component
**Goal:** Implement district micro-profiling analytics (percentiles & benchmarks) and build the dark matte Plotly polar radar component.

**Files:**
- Create: `src/analytics/profile.py`
- Create: `src/app/components/inspector.py`
- Modify: `src/analytics/__init__.py`
- Create: `tests/test_profile.py`

**Steps:**
- [x] Write failing test `tests/test_profile.py`:
  - `compute_regency_profile(gdf, bivariate_df, kode_wil)`:
    * Asserts returned profile contains all 7 micro-indicators: `EWS`, `RAMBU`, `PERKAP`, `RAWAT_SUNGAI`, `GOTONG_ROYONG`, `SINYAL_BURUK`, `IPM`.
    * Asserts National Percentiles and Corridor Percentiles fall strictly in $[0.0, 100.0]$.
    * Asserts comparison medians (National Median, Corridor Median) are accurate.
  - `render_radar_chart(profile)`:
    * Asserts returned Plotly figure is `go.Figure` with polar coordinates.
    * Uses `COLOR_TOKENS` palette (no generic purple/blue glow).
  - `render_inspector_card_html(profile)`:
    * Asserts string starts with `<div` and has zero lines indented with 4 spaces or tabs (`len(bad_lines) == 0`).
- [x] Run `.venv/bin/pytest tests/test_profile.py` to confirm failure (RED).
- [x] Implement `src/analytics/profile.py`:
  - Extract regency attributes and corridor membership.
  - Compute empirical cumulative distribution ranking (`scipy.stats.percentileofscore` or `pandas.Series.rank(pct=True)`).
  - Compute corridor-level and national-level benchmark medians for the 7 indicators.
  - Return `RegencyProfile` dataclass.
- [x] Implement `src/app/components/inspector.py`:
  - `render_radar_chart(profile: RegencyProfile) -> go.Figure`: 3 overlapping polar traces (District, Corridor Median, National Median) with matte fill and clear labels.
  - `render_inspector_card_html(profile: RegencyProfile) -> str`: compact, 0-indent HTML metric badges.
- [x] Export in `src/analytics/__init__.py`.
- [x] Run `.venv/bin/pytest tests/test_profile.py` and verify GREEN.

---

### Task 3.2: Spatial Spillover Econometric Module (SLX / Spatial Lag)
**Goal:** Build spatial econometrics engine calculating direct, indirect (spillover), and total cross-boundary effects using `spreg`.

**Files:**
- Create: `src/analytics/spillover.py`
- Modify: `src/analytics/__init__.py`
- Create: `tests/test_spillover.py`

**Steps:**
- [x] Write failing test `tests/test_spillover.py`:
  - `estimate_spatial_spillover(gdf, weights, y_var, x_vars)`:
    * Estimates spatial lag model ($Y = \rho WY + X\beta + \epsilon$) and SLX model ($Y = X\beta + WX\theta + \epsilon$).
    * Asserts spatial lag parameter $\rho$ is computed and statistically tested.
    * Asserts Direct Effects, Indirect (Spillover) Effects, and Total Effects are computed for each $X_k$.
    * Returns `SpilloverResult` dataclass with coefficients, standard errors, z-statistics, p-values, pseudo-$R^2$, and summary table.
- [x] Run `.venv/bin/pytest tests/test_spillover.py` to confirm failure (RED).
- [x] Implement `src/analytics/spillover.py`:
  - Model formulation with `spreg.GM_Lag` (or `spreg.ML_Lag` / `spreg.OLS` with spatial lag variables).
  - Calculate direct effect: $S_{\text{direct}, k} = \frac{\partial E[y_i]}{\partial x_{ik}}$.
  - Calculate indirect (spillover) effect: $S_{\text{indirect}, k} = \sum_{j \neq i} \frac{\partial E[y_i]}{\partial x_{jk}}$.
  - Calculate total effect: $S_{\text{total}, k} = S_{\text{direct}, k} + S_{\text{indirect}, k}$.
  - Support national scope or corridor sub-graph scope.
- [x] Export in `src/analytics/__init__.py`.
- [x] Run `.venv/bin/pytest tests/test_spillover.py` and verify GREEN.

---

### Task 3.3: FastAPI Endpoints for District Profile & Spillover Analysis
**Goal:** Implement backend REST API endpoints for regency micro-profiles and spatial spillover estimation.

**Files:**
- Modify: `src/api/routes.py`
- Create: `tests/test_api_milestone_3.py`

**Steps:**
- [x] Write failing test `tests/test_api_milestone_3.py`:
  - `GET /api/v1/spatial/regencies/{kode_wil}/profile`:
    * Valid `kode_wil` (e.g. 1101, 3171) returns 200 with 7 radar metrics, national percentiles, corridor percentiles.
    * Invalid `kode_wil` (e.g. 9999) returns 404 Not Found.
  - `GET /api/v1/analytics/spillover?corridor=National&y_var=gap_score`:
    * Returns 200 with model type, rho parameter, and list of effects (direct, indirect, total) for `SINYAL_BURUK`, `GOTONG_ROYONG`, `IPM`.
    * Invalid corridor returns 404 Not Found.
- [x] Run `.venv/bin/pytest tests/test_api_milestone_3.py` to confirm failure (RED).
- [x] Implement endpoints in `src/api/routes.py`:
  - Define Pydantic models: `RegencyProfileResponse`, `SpilloverEffectItem`, `SpilloverResponse`.
  - Wire route handlers using cached `DataManager`.
- [x] Run `.venv/bin/pytest tests/test_api_milestone_3.py` and verify GREEN.

---

### Task 3.4: Viewport Integration & Milestone 3 Verification Gate
**Goal:** Wire District Inspector drawer and Spillover Analysis panel into `src/app/app.py`, execute end-to-end delivery verification, and verify anti-slop compliance.

**Files:**
- Create: `src/app/components/spillover_panel.py`
- Modify: `src/app/app.py`
- Create: `scripts/verify_milestone_3.py`
- Create: `tests/test_milestone_3_gate.py`

**Steps:**
- [x] Write failing test `tests/test_milestone_3_gate.py` asserting `scripts/verify_milestone_3.py` exists, runs cleanly, and returns 0.
- [x] Run `.venv/bin/pytest tests/test_milestone_3_gate.py` to confirm failure (RED).
- [x] Implement `src/app/components/spillover_panel.py`:
  - `render_spillover_effects_table(result: SpilloverResult) -> pd.DataFrame`: Direct, Indirect, and Total effects formatted table.
  - `render_spillover_waterfall_chart(result: SpilloverResult) -> go.Figure`: Horizontal bar decomposition of direct vs cross-boundary spillover.
- [x] Update `src/app/app.py`:
  - Add District Inspector Drawer in sidebar / main area: select any of the 514 regencies to view the 7-indicator radar and percentile ranking.
  - Add "Spillover Analysis" toggle in right analytics panel to switch between Moran Scatterplot and Cross-Boundary Spillover Decomposition.
  - Ensure zero raw HTML leaks and strict anti-slop CSS compliance.
- [x] Implement `scripts/verify_milestone_3.py`:
  - Gate 1: Profile integrity for all 514 units (0 missing, percentiles in $[0, 100]$).
  - Gate 2: Spatial regression latency benchmark ($< 500\text{ ms}$) and parameter boundedness ($|\rho| < 1$).
  - Gate 3: Live FastAPI Milestone 3 endpoints check (`/profile`, `/spillover`).
  - Gate 4: Headless Streamlit dry-run with Inspector radar and Spillover panel rendered.
  - Gate 5: Static anti-slop code audit across `src/` (0 violations).
- [x] Run `.venv/bin/pytest tests/test_milestone_3_gate.py` and verify GREEN.
- [x] Run full test suite: `.venv/bin/pytest tests/ -v`.
- [x] Run `scripts/verify_milestone_3.py` and output Milestone 3 completion receipt.
