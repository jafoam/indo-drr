"""API routes and endpoints for IDR-GE spatial analytics service."""
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional
import geopandas as gpd
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import GEOJSON_FILE, PARQUET_FILE
from src.analytics.weights import build_spatial_weights
from src.analytics.moran import compute_morans_i, compute_lisa
from src.analytics.bivariate import compute_bivariate_scores
from src.analytics.simulation import simulate_policy_intervention
from src.analytics.corridors import (
    CORRIDORS,
    assign_corridors,
    get_corridor_subgraph,
    compute_corridor_moran,
)
from src.analytics.profile import compute_regency_profile
from src.analytics.spillover import estimate_spatial_spillover

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field("ok", description="Service status indicator")
    units: int = Field(514, description="Number of active spatial administrative units")


class MoranResponse(BaseModel):
    """Global Moran's I autocorrelation response schema."""
    variable: str = Field(..., description="Target variable evaluated")
    moran_i: float = Field(..., description="Observed Global Moran's I statistic")
    p_value: float = Field(..., description="Simulated pseudo p-value")
    z_score: float = Field(..., description="Standardized z-score")
    expected_i: float = Field(..., description="Expected I under spatial randomness")
    simulations: int = Field(999, description="Number of Monte Carlo permutations")


class BivariateItem(BaseModel):
    """Bivariate risk vs mitigation item for single administrative unit."""
    KODE_WIL: int
    KAB_KOTA: str
    PROVINSI: str
    risk_score: float
    mitigation_score: float
    gap_score: float
    risk_tercile: str
    mitigation_tercile: str
    bivariate_category: str
    bivariate_color: str
    bivariate_label: str
    is_critical_gap: bool
    is_resilient: bool


class LisaItem(BaseModel):
    """Local Moran LISA result for single administrative unit."""
    KODE_WIL: int
    KAB_KOTA: str
    PROVINSI: str
    lisa_i: float
    p_value: float
    z_score: float
    lisa_q: int
    lisa_cluster: str
    color: str


class VariableMetadata(BaseModel):
    """Metadata for an analytical variable."""
    column: str
    category: str
    description: str


class SimulationRequest(BaseModel):
    """Request schema for policy intervention simulation."""
    signal_reduction: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Percentage reduction in weak/no-signal villages [0.0, 100.0]",
    )
    gotong_royong_boost: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Percentage increase in community mutual aid [0.0, 100.0]",
    )


class SimulationUnitResult(BaseModel):
    """Unit-level simulated bivariate outcomes."""
    KODE_WIL: int
    KAB_KOTA: Optional[str] = None
    PROVINSI: Optional[str] = None
    simulated_mitigation_score: Optional[float] = None
    simulated_gap_score: float
    simulated_bivariate_category: str
    simulated_bivariate_color: str
    simulated_bivariate_label: Optional[str] = None
    is_critical_gap_simulated: bool
    is_resilient_simulated: Optional[bool] = None


class SimulationResponse(BaseModel):
    """Aggregated policy intervention simulation outcome."""
    baseline_critical_gaps: int
    simulated_critical_gaps: int
    resolved_gaps: int
    units: List[SimulationUnitResult]


class CorridorSummary(BaseModel):
    """Regional corridor metadata and coverage summary."""
    name: str
    unit_count: int
    provinces: List[str]


class CorridorMoranResponse(BaseModel):
    """Localized Global Moran's I spatial autocorrelation for a regional corridor."""
    corridor: str
    variable: str
    unit_count: int
    moran_i: float
    p_value: float
    z_score: float
    expected_i: Optional[float] = None
    simulations: Optional[int] = None


class IndicatorMetricModel(BaseModel):
    """Benchmark metric for an individual district indicator."""
    name: str = Field(..., description="Indicator column identifier (e.g. EWS, IPM)")
    label: str = Field(..., description="Human-readable indicator label")
    value: float = Field(..., description="Observed raw value for this administrative unit")
    national_pct: float = Field(..., description="Empirical national percentile [0.0, 100.0]")
    corridor_pct: float = Field(..., description="Empirical regional corridor percentile [0.0, 100.0]")
    national_median: float = Field(..., description="National distribution median value")
    corridor_median: float = Field(..., description="Corridor distribution median value")
    description: str = Field("", description="Detailed narrative description")


class RegencyProfileResponse(BaseModel):
    """Complete diagnostic micro-profile response for a regency / city."""
    kode_wil: int = Field(..., description="BPS 4-digit administrative unit code")
    kab_kota: str = Field(..., description="Regency or city name")
    provinsi: str = Field(..., description="Province name")
    corridor: str = Field(..., description="Regional archipelagic corridor")
    is_kota: int = Field(..., description="1 if Kota (urban), 0 if Kabupaten (rural)")
    risk_score: float = Field(..., description="Standardized disaster risk index [0, 1]")
    mitigation_score: float = Field(..., description="Standardized disaster mitigation capacity index [0, 1]")
    gap_score: float = Field(..., description="Disaster risk-mitigation gap score [-1, 1]")
    gap_rank: int = Field(..., description="National vulnerability ranking (1 = highest gap)")
    bivariate_category: str = Field(..., description="3x3 bivariate grid class (e.g. R3-M1, R1-M3)")
    bivariate_label: str = Field(..., description="Bivariate classification narrative label")
    bivariate_color: str = Field(..., description="Hex color token for bivariate grid cell")
    indicators: Dict[str, IndicatorMetricModel] = Field(..., description="7 diagnostic micro-indicators")


class SpilloverEffectModel(BaseModel):
    """Estimated direct, indirect (spillover), and total impacts for an analytical regressor."""
    variable: str = Field(..., description="Regressor column identifier (e.g. SINYAL_BURUK, GOTONG_ROYONG, IPM)")
    direct_effect: float = Field(..., description="Direct localized impact estimate")
    indirect_effect: float = Field(..., description="Cross-boundary spatial spillover estimate onto neighbors")
    total_effect: float = Field(..., description="Combined regional total impact")
    spillover_ratio: float = Field(..., description="Proportion of total effect attributable to spatial spillover [0.0, 1.0]")
    coef: float = Field(..., description="Raw model regression coefficient (beta)")
    std_err: float = Field(..., description="Standard error of the estimate")
    z_stat: float = Field(..., description="z-statistic or t-statistic")
    p_value: float = Field(..., description="Statistical significance p-value")
    interpretation: str = Field(..., description="Domain-specific impact interpretation narrative")
    indirect_p_value: Optional[float] = Field(None, description="p-value for indirect effect (SLX only)")
    total_p_value: Optional[float] = Field(None, description="p-value for total effect (SLX only)")


class SpilloverResponse(BaseModel):
    """Complete spatial spillover econometric regression response schema."""
    model_type: str = Field(..., description="Spatial model type: 'lag' (SAR) or 'slx'")
    y_var: str = Field(..., description="Dependent target variable (e.g. gap_score)")
    x_vars: List[str] = Field(..., description="List of regressor variables")
    n_obs: int = Field(..., description="Number of administrative observations included")
    rho: Optional[float] = Field(None, description="Spatial autoregressive lag parameter (SAR only)")
    rho_p_value: Optional[float] = Field(None, description="Spatial lag parameter p-value")
    rho_z_stat: Optional[float] = Field(None, description="Spatial lag parameter z-statistic")
    r2: float = Field(..., description="Pseudo-R2 (lag) or R2 (slx)")
    effects: List[SpilloverEffectModel] = Field(..., description="Decomposed direct, indirect, and total impacts")
    corridor: Optional[str] = Field(None, description="Active regional corridor or None for National")
    model_summary: Optional[str] = Field(None, description="Textual regression output summary")


# ---------------------------------------------------------------------------
# Data State & In-Memory Cache
# ---------------------------------------------------------------------------

class DataManager:
    """Manages cached spatial data and weights in memory."""

    def __init__(self):
        self._gdf: Optional[gpd.GeoDataFrame] = None
        self._df: Optional[pd.DataFrame] = None
        self._weights = None
        self._raw_geojson: Optional[Dict[str, Any]] = None
        self._bivariate_df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Load and cache spatial datasets."""
        if self._gdf is not None:
            return

        if not GEOJSON_FILE.exists():
            raise FileNotFoundError(f"GeoJSON file not found at {GEOJSON_FILE}")
        if not PARQUET_FILE.exists():
            raise FileNotFoundError(f"Parquet dataset not found at {PARQUET_FILE}")

        # Load GeoJSON geometry and base metadata
        with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
            self._raw_geojson = json.load(f)

        gdf = gpd.read_file(GEOJSON_FILE)
        df = pd.read_parquet(PARQUET_FILE)

        # Ensure matching key data types
        gdf["KODE_WIL"] = gdf["KODE_WIL"].astype(int)
        df["KODE_WIL"] = df["KODE_WIL"].astype(int)

        # Merge tabular attributes into GeoDataFrame
        attr_cols = [c for c in df.columns if c not in gdf.columns or c == "KODE_WIL"]
        merged_gdf = gdf.merge(df[attr_cols], on="KODE_WIL", how="inner")

        # Assign corridors
        df = assign_corridors(df)
        merged_gdf = assign_corridors(merged_gdf)

        # Compute bivariate scores
        self._bivariate_df = compute_bivariate_scores(df)
        self._gdf = compute_bivariate_scores(merged_gdf)
        self._df = df
        self._weights = build_spatial_weights(self._gdf, k=5)

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        if self._gdf is None:
            self.load()
        return self._gdf

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df

    @property
    def weights(self):
        if self._weights is None:
            self.load()
        return self._weights

    @property
    def raw_geojson(self) -> Dict[str, Any]:
        if self._raw_geojson is None:
            self.load()
        return self._raw_geojson

    @property
    def bivariate_df(self) -> pd.DataFrame:
        if self._bivariate_df is None:
            self.load()
        return self._bivariate_df


_data_manager = DataManager()


def get_data_manager() -> DataManager:
    """Return singleton DataManager instance."""
    return _data_manager


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """System health check and unit count verification."""
    dm = get_data_manager()
    dm.load()
    return HealthResponse(status="ok", units=len(dm.df))


@router.get("/api/v1/spatial/regencies", tags=["Spatial"])
def get_regencies(include_indicators: bool = False) -> Dict[str, Any]:
    """Retrieve 514 Indonesian regency geometries as GeoJSON FeatureCollection."""
    dm = get_data_manager()
    dm.load()

    if not include_indicators:
        return dm.raw_geojson

    # Return full GeoJSON with all analytical columns
    return json.loads(dm.gdf.to_json())


@router.get(
    "/api/v1/analytics/moran",
    response_model=MoranResponse,
    tags=["Analytics"],
)
def get_global_moran(
    variable: str = Query("IRBI_2024", description="Variable column to calculate Global Moran's I"),
    permutations: int = Query(999, ge=99, le=9999, description="Number of Monte Carlo permutations"),
) -> MoranResponse:
    """Calculate Global Moran's I spatial autocorrelation for target variable."""
    dm = get_data_manager()
    dm.load()

    if variable not in dm.gdf.columns:
        valid_cols = [c for c in dm.gdf.columns if c not in ("geometry", "KODE_KK")]
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{variable}' not found. Available variables include: {valid_cols[:10]}...",
        )

    res = compute_morans_i(
        dm.gdf,
        column=variable,
        weights=dm.weights,
        permutations=permutations,
    )

    return MoranResponse(
        variable=res.variable,
        moran_i=res.moran_i,
        p_value=res.p_value,
        z_score=res.z_score,
        expected_i=res.expected_i,
        simulations=res.simulations,
    )


@router.get(
    "/api/v1/analytics/lisa",
    response_model=List[LisaItem],
    tags=["Analytics"],
)
def get_lisa_clusters(
    variable: str = Query("IRBI_2024", description="Target variable for LISA clusters"),
    p_threshold: float = Query(0.05, gt=0.0, lt=1.0, description="Significance p-value threshold"),
) -> List[LisaItem]:
    """Compute Local Indicators of Spatial Association (LISA) clusters."""
    dm = get_data_manager()
    dm.load()

    if variable not in dm.gdf.columns:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found in dataset.")

    lisa_gdf = compute_lisa(
        dm.gdf,
        column=variable,
        weights=dm.weights,
        p_threshold=p_threshold,
    )

    items: List[LisaItem] = []
    for _, row in lisa_gdf.iterrows():
        items.append(
            LisaItem(
                KODE_WIL=int(row["KODE_WIL"]),
                KAB_KOTA=str(row["KAB_KOTA"]),
                PROVINSI=str(row["PROVINSI"]),
                lisa_i=float(row["lisa_i"]),
                p_value=float(row["p_value"]),
                z_score=float(row["z_score"]),
                lisa_q=int(row["lisa_q"]),
                lisa_cluster=str(row["lisa_cluster"]),
                color=str(row["color"]),
            )
        )
    return items


@router.get(
    "/api/v1/analytics/bivariate",
    response_model=List[BivariateItem],
    tags=["Analytics"],
)
def get_bivariate_analysis() -> List[BivariateItem]:
    """Compute and retrieve 3x3 Risk vs Mitigation bivariate classification for all 514 units."""
    dm = get_data_manager()
    dm.load()

    biv_df = dm.bivariate_df
    items: List[BivariateItem] = []
    for _, row in biv_df.iterrows():
        items.append(
            BivariateItem(
                KODE_WIL=int(row["KODE_WIL"]),
                KAB_KOTA=str(row["KAB_KOTA"]),
                PROVINSI=str(row["PROVINSI"]),
                risk_score=round(float(row["risk_score"]), 4),
                mitigation_score=round(float(row["mitigation_score"]), 4),
                gap_score=round(float(row["gap_score"]), 4),
                risk_tercile=str(row["risk_tercile"]),
                mitigation_tercile=str(row["mitigation_tercile"]),
                bivariate_category=str(row["bivariate_category"]),
                bivariate_color=str(row["bivariate_color"]),
                bivariate_label=str(row["bivariate_label"]),
                is_critical_gap=bool(row["is_critical_gap"]),
                is_resilient=bool(row["is_resilient"]),
            )
        )
    return items


@router.post(
    "/api/v1/analytics/simulate",
    response_model=SimulationResponse,
    tags=["Analytics"],
)
def post_simulate_intervention(
    req: SimulationRequest = SimulationRequest(),
) -> SimulationResponse:
    """Simulate policy interventions on digital infrastructure and social capital."""
    dm = get_data_manager()
    dm.load()

    result = simulate_policy_intervention(
        dm.bivariate_df,
        signal_reduction_pct=req.signal_reduction,
        gotong_royong_boost_pct=req.gotong_royong_boost,
    )

    sim_df = result.df
    units: List[SimulationUnitResult] = []
    for _, row in sim_df.iterrows():
        units.append(
            SimulationUnitResult(
                KODE_WIL=int(row["KODE_WIL"]),
                KAB_KOTA=str(row["KAB_KOTA"]) if "KAB_KOTA" in row and pd.notna(row["KAB_KOTA"]) else None,
                PROVINSI=str(row["PROVINSI"]) if "PROVINSI" in row and pd.notna(row["PROVINSI"]) else None,
                simulated_mitigation_score=round(float(row["simulated_mitigation_score"]), 4)
                if "simulated_mitigation_score" in row and pd.notna(row["simulated_mitigation_score"])
                else None,
                simulated_gap_score=round(float(row["simulated_gap_score"]), 4),
                simulated_bivariate_category=str(row["simulated_bivariate_category"]),
                simulated_bivariate_color=str(row["simulated_bivariate_color"]),
                simulated_bivariate_label=str(row.get("simulated_bivariate_label", ""))
                if "simulated_bivariate_label" in row and pd.notna(row["simulated_bivariate_label"])
                else None,
                is_critical_gap_simulated=bool(row["is_critical_gap_simulated"]),
                is_resilient_simulated=bool(row.get("is_resilient_simulated", False)),
            )
        )

    return SimulationResponse(
        baseline_critical_gaps=int(result.baseline_critical_gaps),
        simulated_critical_gaps=int(result.simulated_critical_gaps),
        resolved_gaps=int(result.resolved_gaps),
        units=units,
    )


@router.get(
    "/api/v1/spatial/corridors",
    response_model=List[CorridorSummary],
    tags=["Spatial"],
)
def get_spatial_corridors() -> List[CorridorSummary]:
    """Retrieve 7 archipelagic regional corridors, their unit counts, and member provinces."""
    dm = get_data_manager()
    dm.load()

    df = dm.df
    if "corridor" not in df.columns:
        df = assign_corridors(df)

    summaries: List[CorridorSummary] = []
    for corridor_name in CORRIDORS:
        sub = df[df["corridor"] == corridor_name]
        provinces = sorted(sub["PROVINSI"].dropna().unique().tolist())
        summaries.append(
            CorridorSummary(
                name=corridor_name,
                unit_count=len(sub),
                provinces=provinces,
            )
        )
    return summaries


@router.get(
    "/api/v1/analytics/corridor/moran",
    response_model=CorridorMoranResponse,
    tags=["Analytics"],
)
def get_corridor_moran(
    corridor: str = Query(..., description="Corridor name (e.g., Sumatera, Jawa-Bali)"),
    variable: str = Query("IRBI_2024", description="Variable column to calculate localized Global Moran's I"),
    permutations: int = Query(999, ge=99, le=9999, description="Number of Monte Carlo permutations"),
) -> CorridorMoranResponse:
    """Calculate localized Global Moran's I autocorrelation for a specific regional corridor."""
    if corridor not in CORRIDORS:
        raise HTTPException(
            status_code=404,
            detail=f"Corridor '{corridor}' not found. Available corridors: {CORRIDORS}",
        )

    dm = get_data_manager()
    dm.load()

    if variable not in dm.gdf.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{variable}' not found in dataset.",
        )

    corridor_gdf = get_corridor_subgraph(dm.gdf, corridor)
    res = compute_corridor_moran(
        corridor_gdf,
        variable=variable,
        permutations=permutations,
    )

    return CorridorMoranResponse(
        corridor=corridor,
        variable=res.variable,
        unit_count=len(corridor_gdf),
        moran_i=res.moran_i,
        p_value=res.p_value,
        z_score=res.z_score,
        expected_i=res.expected_i,
        simulations=res.simulations,
    )


@router.get(
    "/api/v1/spatial/regencies/{kode_wil}/profile",
    response_model=RegencyProfileResponse,
    tags=["Spatial"],
)
def get_regency_profile(kode_wil: int) -> RegencyProfileResponse:
    """Retrieve micro-profile district diagnostics and 7-indicator benchmarking percentiles."""
    dm = get_data_manager()
    dm.load()

    try:
        profile = compute_regency_profile(dm.gdf, dm.bivariate_df, int(kode_wil))
    except (KeyError, ValueError) as err:
        raise HTTPException(
            status_code=404,
            detail=f"Regency with KODE_WIL {kode_wil} not found: {err}",
        )

    return RegencyProfileResponse(**profile.to_dict())


@router.get(
    "/api/v1/analytics/spillover",
    response_model=SpilloverResponse,
    tags=["Analytics"],
)
def get_spillover_analysis(
    corridor: Optional[str] = Query(None, description="Regional corridor name ('Sumatera', 'Jawa-Bali', etc.) or 'National' / None"),
    y_var: str = Query("gap_score", description="Target dependent variable (default 'gap_score')"),
    model_type: str = Query("lag", description="Spatial econometric model type ('lag' or 'slx')"),
) -> SpilloverResponse:
    """Estimate spatial spillover regression models and decompose cross-boundary impacts."""
    clean_corridor: Optional[str] = None
    if corridor is not None and corridor.strip() != "":
        val = corridor.strip()
        if val.lower() != "national":
            if val not in CORRIDORS:
                raise HTTPException(
                    status_code=404,
                    detail=f"Corridor '{corridor}' not found. Available corridors: {CORRIDORS}",
                )
            clean_corridor = val

    dm = get_data_manager()
    dm.load()

    effective_weights = dm.weights if clean_corridor is None else None

    try:
        res = estimate_spatial_spillover(
            dm.gdf,
            weights=effective_weights,
            y_var=y_var,
            model_type=model_type,
            corridor=clean_corridor,
        )
    except (ValueError, KeyError) as err:
        raise HTTPException(status_code=400, detail=str(err))

    effects_models = [
        SpilloverEffectModel(
            variable=e.variable,
            direct_effect=round(float(e.direct_effect), 6),
            indirect_effect=round(float(e.indirect_effect), 6),
            total_effect=round(float(e.total_effect), 6),
            spillover_ratio=round(float(e.spillover_ratio), 6),
            coef=round(float(e.coef), 6),
            std_err=round(float(e.std_err), 6),
            z_stat=round(float(e.z_stat), 6),
            p_value=round(float(e.p_value), 6),
            interpretation=e.interpretation,
            indirect_p_value=round(float(e.indirect_p_value), 6) if e.indirect_p_value is not None else None,
            total_p_value=round(float(e.total_p_value), 6) if e.total_p_value is not None else None,
        )
        for e in res.effects
    ]

    return SpilloverResponse(
        model_type=res.model_type,
        y_var=res.y_var,
        x_vars=res.x_vars,
        n_obs=res.n_obs,
        rho=round(float(res.rho), 6) if res.rho is not None else None,
        rho_p_value=round(float(res.rho_p_value), 6) if res.rho_p_value is not None else None,
        rho_z_stat=round(float(res.rho_z_stat), 6) if res.rho_z_stat is not None else None,
        r2=round(float(res.r2), 6),
        effects=effects_models,
        corridor=res.corridor,
        model_summary=res.model_summary,
    )

