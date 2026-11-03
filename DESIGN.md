# Indonesian Disaster Readiness & Gap Explorer (IDR-GE)
## System Design & Technical Specification — Milestone 1

### 1. Overview & Objective
Interactive spatial analytics platform combining:
- **PODES 2024** (Potensi Desa / Village Potential Census) disaster mitigation & infrastructure indicators.
- **BNPB IRBI 2024** (Indeks Risiko Bencana Indonesia / Disaster Risk Index).
- **Indonesian Regency/City Administrative Boundaries** (514 spatial units).
- **Spatial Econometrics** (Spatial weights, Moran's I autocorrelation, LISA cluster mapping, bivariate readiness-gap matrix).

Architecture: FastAPI backend service providing spatial statistics and geo-queries, paired with a high-density Streamlit viewport powered by PyDeck/Plotly spatial choropleths.

---

### 2. Data Architecture & Schema

#### 2.1 Administrative Spatial Units
- **Total Units**: 514 regencies/cities (Kabupaten/Kota) spanning 38 provinces.
- **Primary Spatial Key**: `KODE_WIL` (BPS numeric 4-digit code, e.g., `1101`, `3171`, `7371`).
- **Secondary Key**: `KODE_KK` (formatted code, e.g., `11.01`).
- **Coordinate Reference System (CRS)**: WGS84 (EPSG:4326) for GeoJSON storage and PyDeck viewport.
- **Geometry Source**: Extracted from `Data/GEODA/MITIGASI BENCANA.shp`, topologically validated, and simplified (<5MB) to `data/processed/indonesia_regencies.geojson`.

#### 2.2 Variables Schema
Extracted from `Data/DATA_V#.csv` and verified 1:1 against the 514 administrative spatial records:

| Field Name | Type | Domain | Description | Category |
| :--- | :--- | :--- | :--- | :--- |
| `KODE_WIL` | Integer | [1101, 9471] | Official BPS 4-digit district code | Key |
| `KAB_KOTA_NAME`| String | Text | Regency/City Name (from DBF/Shapefile) | Metadata |
| `PROVINSI` | String | Text | Province Name | Metadata |
| `IS_KOTA` | Integer | {0, 1} | 1 = Kota (City), 0 = Kabupaten (Regency) | Metadata |
| `IRBI_2024` | Float | [0.0, 300.0] | Disaster Risk Index (BNPB 2024) | Risk |
| `KEJADIAN_BENCANA`| Integer | $\ge 0$ | Total disaster incidents reported (PODES 2024) | Risk |
| `TERDAMPAK` | Integer | $\ge 0$ | Total population impacted by disaster | Risk |
| `ADA_BENCANA` | Float | [0.0, 100.0] | % of villages in regency experiencing disaster | Risk |
| `EWS` | Float | [0.0, 100.0] | % villages with Early Warning System | Mitigation |
| `EWS_TSUNAMI`| Float | [0.0, 100.0] | % coastal villages with tsunami warning equipment| Mitigation |
| `PERKAP` | Float | [0.0, 100.0] | % villages with disaster mitigation equipment | Mitigation |
| `RAMBU` | Float | [0.0, 100.0] | % villages with evacuation route signage | Mitigation |
| `RAWAT_SUNGAI`| Float | [0.0, 100.0] | % villages actively maintaining riverbeds | Mitigation |
| `MANGROVE` | Float | [0.0, 100.0] | % coastal villages with mangrove preservation | Mitigation |
| `KEMISKINAN` | Float | [0.0, 100.0] | Poverty headcount rate (%) | Vulnerability |
| `IPM` | Float | [0.0, 100.0] | Human Development Index (HDI) | Vulnerability |
| `LANSIA` | Float | [0.0, 100.0] | Elderly population share (%) | Vulnerability |
| `LOW_EDU` | Float | [0.0, 100.0] | Population with low education attainment (%) | Vulnerability |
| `AKSES_AIR` | Float | [0.0, 100.0] | % households with access to clean water | Vulnerability |
| `SINYAL_BURUK`| Float | [0.0, 100.0] | % villages with weak/no telecommunication signal | Vulnerability |
| `GOTONG_ROYONG`| Float | [0.0, 100.0] | Community mutual aid participation index | Social Capital |

#### 2.3 Derived Composite Indices
For bivariate spatial analysis, indicators are normalized to $[0, 1]$ via min-max scaling:
- **Composite Risk Score ($R_i$)**:
  $$R_i = 0.5 \cdot \text{Norm}(\text{IRBI\_2024}) + 0.3 \cdot \text{Norm}(\text{ADA\_BENCANA}) + 0.2 \cdot \text{Norm}(\ln(\text{TERDAMPAK} + 1))$$
- **Composite Mitigation Preparedness Score ($M_i$)**:
  $$M_i = 0.30 \cdot \text{Norm}(\text{EWS}) + 0.25 \cdot \text{Norm}(\text{RAMBU}) + 0.25 \cdot \text{Norm}(\text{PERKAP}) + 0.20 \cdot \text{Norm}(\text{RAWAT\_SUNGAI})$$
- **Disaster Gap Score ($G_i$)**:
  $$G_i = R_i - M_i \in [-1, 1]$$
  (Positive values represent acute vulnerability where risk exceeds mitigation infrastructure).

---

### 3. Spatial Weights & Econometrics Architecture

#### 3.1 Spatial Contiguity Matrix for Archipelagic Geography
Because Indonesia consists of over 17,000 islands, standard Queen/Rook contiguity produces disconnected islands (islands with zero neighbors, causing singular spatial weight matrices).
- **Strategy**: Hybrid K-Nearest Neighbors ($k=5$) or Queen contiguity augmented with $k$-NN minimum distance links for insular units (`libpysal.weights.KNN.from_dataframe` or `libpysal.weights.Queen` + island linkage).
- **Transformation**: Row-standardized spatial weight matrix ($W_{std}$).

#### 3.2 Global & Local Spatial Autocorrelation
- **Global Moran's I**:
  $$I = \frac{n}{\sum_{i=1}^n \sum_{j=1}^n w_{ij}} \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_{i=1}^n (x_i - \bar{x})^2}$$
- **Local Indicators of Spatial Association (LISA)**:
  Categorizes regions into 5 statistical clusters ($p \le 0.05$ via 999 Monte Carlo permutations):
  1. **High-High (Hotspot)**: High risk regency surrounded by high risk regencies.
  2. **Low-Low (Coldspot)**: Low risk regency surrounded by low risk regencies.
  3. **High-Low (Spatial Outlier)**: High risk regency surrounded by low risk regencies.
  4. **Low-High (Spatial Outlier)**: Low risk regency surrounded by high risk regencies.
  5. **Not Significant**: $p > 0.05$.

---

### 4. Anti-Slop Visual Tokens & UI Specification

Strict adherence to `antislop-ui` rules:
- **No generic purple/blue gradients** (R-01).
- **No glassmorphism or floating soft card shadows** (R-10, R-12).
- **No gratuitous circular progress gauges or decorative KPI tiles** (R-14).
- **Data-First Layout**: Spatial choropleth takes 70% of viewport width; right 30% panel houses Moran scatterplot and bivariate gap distribution.

#### 4.1 Color Tokens
| Token | Hex Value | Semantic Role |
| :--- | :--- | :--- |
| `bg-primary` | `#0f141c` | Dark analytical viewport background |
| `bg-surface` | `#161e2a` | Panel, sidebar, and control surface |
| `bg-surface-elevated`| `#1e293b` | Interactive controls, inputs, hover states |
| `border-subtle` | `#2a384c` | Structural divider lines (1px crisp) |
| `text-primary` | `#f1f5f9` | Primary labels, statistics, tooltips |
| `text-muted` | `#94a3b8` | Supporting metadata, unit labels |
| `accent-focus` | `#38bdf8` | Interactive active border / selection highlight |

#### 4.2 Bivariate 3x3 Risk vs. Mitigation Color Palette
A 9-class choropleth palette mapping Terza/Stevens bivariate color ramps optimized for cartographic contrast and color-blind friendliness:

| Risk \ Mitigation | Low Mitigation ($M_1$) | Med Mitigation ($M_2$) | High Mitigation ($M_3$) |
| :--- | :--- | :--- | :--- |
| **High Risk ($R_3$)** | `#d03838` (Critical Gap) | `#c47c5d` | `#82537e` (High Risk / High Mitigation)|
| **Med Risk ($R_2$)** | `#d88478` | `#bfa29b` | `#7a8ba5` |
| **Low Risk ($R_1$)** | `#e2dfd2` (Low Risk / Low Mit) | `#9fb8be` | `#3b82a6` (Resilient Prepared) |

#### 4.3 LISA Cluster Palette
- **High-High (Hotspot)**: `#ef4444` (Vivid Crimson)
- **Low-Low (Coldspot)**: `#0284c7` (Cobalt Blue)
- **High-Low (Outlier)**: `#f97316` (Amber Orange)
- **Low-High (Outlier)**: `#06b6d4` (Teal Cyan)
- **Not Significant**: `#334155` (Subdued Muted Slate)

---

### 5. Repository File Structure

```
.
├── DESIGN.md                                 # Technical architecture & visual spec
├── implementation_plan.md                    # Milestone 1 execution plan
├── pyproject.toml / requirements.txt         # Pinned virtualenv dependencies
├── data/
│   ├── raw/
│   │   ├── DATA_V#.csv                       # BPS & PODES 514 regency dataset
│   │   └── geoda_shapefile/                  # Original raw boundary shapefiles
│   └── processed/
│       ├── indonesia_regencies.geojson       # Simplified, validated 514 GeoJSON (<5MB)
│       └── podes_irbi_clean.parquet          # Normalized, typed analytical dataset
├── src/
│   ├── __init__.py
│   ├── config.py                             # Settings, filepaths, color constants
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py                         # CSV & Shapefile parsing
│   │   └── validator.py                      # 514-unit code validation & integrity gates
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── weights.py                        # Spatial weights (KNN / Queen archipelagic)
│   │   ├── moran.py                          # Global Moran's I & local LISA computation
│   │   └── bivariate.py                      # Composite index calculation & 3x3 categorization
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                           # FastAPI application entrypoint
│   │   └── routes.py                         # Spatial data, Moran stats, GeoJSON endpoints
│   └── app/
│       ├── __init__.py
│       ├── app.py                            # Streamlit entrypoint
│       └── components/
│           ├── __init__.py
│           ├── choropleth.py                 # PyDeck / Plotly map renderer
│           └── stats_panel.py                # Moran scatterplot & density histograms
└── tests/
    ├── __init__.py
    ├── test_ingestion.py                     # 514 codes & CRS unit test
    ├── test_spatial_weights.py               # Weights matrix non-singularity test
    ├── test_moran.py                         # Moran's I statistical verification
    └── test_api.py                           # FastAPI endpoints integration test
```
