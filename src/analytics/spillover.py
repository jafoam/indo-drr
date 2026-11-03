"""Spatial Spillover Econometric Module for Cross-Boundary Impact Analysis.

Estimates spatial lag (SAR: Y = rho*W*Y + X*beta + eps) and spatial lag of X
(SLX: Y = X*beta + W*X*theta + eps) models to quantify direct, indirect (cross-boundary
spillover), and total effects using LeSage & Pace (2009) spatial multiplier decomposition.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import W
import spreg
from spreg.sputils import spmultiplier

from src.analytics.weights import build_spatial_weights
from src.analytics.bivariate import compute_bivariate_scores
from src.analytics.corridors import (
    CORRIDORS,
    get_corridor_subgraph,
    get_corridor_weights,
)

DEFAULT_X_VARS: List[str] = ["SINYAL_BURUK", "GOTONG_ROYONG", "IPM"]


@dataclass
class SpilloverEffect:
    """Estimated direct, indirect (spillover), and total impacts for an analytical regressor."""

    variable: str
    direct_effect: float
    indirect_effect: float
    total_effect: float
    spillover_ratio: float
    coef: float
    std_err: float
    z_stat: float
    p_value: float
    interpretation: str
    indirect_p_value: Optional[float] = None
    total_p_value: Optional[float] = None


@dataclass
class SpilloverResult:
    """Complete spatial spillover econometric regression result."""

    model_type: str  # 'lag' or 'slx'
    y_var: str
    x_vars: List[str]
    n_obs: int
    rho: Optional[float]  # Spatial autoregressive parameter (float for 'lag', None for 'slx')
    rho_p_value: Optional[float]
    rho_z_stat: Optional[float]
    r2: float  # Pseudo-R2 (lag) or R2 (slx)
    effects: List[SpilloverEffect]
    summary_df: pd.DataFrame
    corridor: Optional[str] = None
    model_summary: Optional[str] = None


def _generate_interpretation(
    var: str,
    direct: float,
    indirect: float,
    total: float,
    p_val: float,
) -> str:
    """Generate human-readable, domain-specific interpretation for a regressor impact.

    Args:
        var: Regressor indicator name.
        direct: Direct localized impact estimate.
        indirect: Cross-boundary spatial spillover estimate.
        total: Combined total regional impact.
        p_val: Statistical significance p-value.

    Returns:
        Formatted narrative string describing direct and spillover impacts.
    """
    sig_note = "statistically significant" if p_val < 0.05 else "not statistically significant"

    if var == "SINYAL_BURUK":
        return (
            f"A 1% increase in weak/no-signal villages directly changes the disaster gap by {direct:+.4f}, "
            f"with an additional cross-boundary spillover of {indirect:+.4f} onto neighboring districts "
            f"(total impact: {total:+.4f}, p={p_val:.4f}, {sig_note})."
        )
    if var == "GOTONG_ROYONG":
        return (
            f"A 1-unit increase in community mutual aid directly changes the disaster gap by {direct:+.4f}, "
            f"generating a protective cross-boundary spillover of {indirect:+.4f} for surrounding regencies "
            f"(total impact: {total:+.4f}, p={p_val:.4f}, {sig_note})."
        )
    if var == "IPM":
        return (
            f"A 1-point increase in Human Development Index directly changes the disaster gap by {direct:+.4f}, "
            f"generating regional positive externalities of {indirect:+.4f} across neighboring regencies "
            f"(total impact: {total:+.4f}, p={p_val:.4f}, {sig_note})."
        )

    direction = "increases" if total > 0 else "reduces" if total < 0 else "leaves unchanged"
    return (
        f"A 1-unit increase in {var} directly changes the target variable by {direct:+.4f}, "
        f"with a spatial spillover of {indirect:+.4f} onto neighbors, which {direction} regional impact "
        f"(total: {total:+.4f}, p={p_val:.4f}, {sig_note})."
    )


def estimate_spatial_spillover(
    gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
    weights: Optional[W] = None,
    y_var: str = "gap_score",
    x_vars: Optional[List[str]] = None,
    model_type: str = "lag",
    corridor: Optional[str] = None,
) -> SpilloverResult:
    """Estimate spatial spillover regression models and decompose cross-boundary impacts.

    Supports:
        1. Spatial Lag (SAR): Y = rho*W*Y + X*beta + eps via Generalized Moments (spreg.GM_Lag).
           Impacts decomposed using LeSage & Pace (2009) spatial multiplier matrix (I - rho*W)^(-1).
        2. Spatial Lag of X (SLX): Y = X*beta + W*X*theta + eps via OLS (spreg.OLS).
           Direct effect = beta, Indirect (spillover) = theta, Total = beta + theta.

    Args:
        gdf: GeoDataFrame or DataFrame containing unit records and geometries.
        weights: Optional precomputed row-standardized libpysal.weights.W matrix.
        y_var: Dependent variable name (default 'gap_score').
        x_vars: Regressor variable names (default ['SINYAL_BURUK', 'GOTONG_ROYONG', 'IPM']).
        model_type: Either 'lag' (SAR model) or 'slx' (Spatial Lag of X).
        corridor: Optional regional corridor name ('Sumatera', 'Jawa-Bali', etc.).

    Returns:
        SpilloverResult with rho parameter, decomposed effects, and formatted summary table.

    Raises:
        ValueError: If model_type or corridor is invalid, or if weights dimension mismatches.
        KeyError: If y_var or any x_vars are missing from the data.
    """
    clean_model_type = model_type.strip().lower()
    if clean_model_type not in ("lag", "slx"):
        raise ValueError(
            f"Invalid model_type: '{model_type}'. Supported models are 'lag' and 'slx'."
        )

    # 1. Filter corridor subgraph if specified
    active_corridor: Optional[str] = None
    if corridor is not None and corridor.strip() != "" and corridor.strip().lower() != "national":
        clean_corridor = corridor.strip()
        if clean_corridor not in CORRIDORS:
            raise ValueError(
                f"Invalid corridor '{corridor}'. Must be one of: {CORRIDORS} or 'National'."
            )
        work_df = get_corridor_subgraph(gdf, clean_corridor)
        active_corridor = clean_corridor
    else:
        work_df = gdf.copy()

    # 2. Resolve dependent variable (compute bivariate scores if needed)
    if y_var not in work_df.columns:
        if y_var in ("gap_score", "risk_score", "mitigation_score"):
            try:
                work_df = compute_bivariate_scores(work_df)
            except Exception as err:
                raise KeyError(
                    f"Target variable '{y_var}' could not be computed: {err}"
                ) from err
        if y_var not in work_df.columns:
            raise KeyError(f"Dependent variable '{y_var}' not found in data.")

    # 3. Resolve independent regressors
    if x_vars is None:
        effective_x_vars = list(DEFAULT_X_VARS)
    else:
        effective_x_vars = list(x_vars)

    missing_x = [c for c in effective_x_vars if c not in work_df.columns]
    if missing_x:
        raise KeyError(f"Independent variables not found in data: {missing_x}")

    # Drop any null records in analysis columns
    cols_to_check = [y_var] + effective_x_vars
    work_df = work_df.dropna(subset=cols_to_check).copy()
    n_obs = len(work_df)
    if n_obs < 3:
        raise ValueError(f"Insufficient observations for spatial regression (N={n_obs}).")

    # 4. Spatial weights matrix setup and row-standardization
    if weights is None or (active_corridor is not None and weights.n != n_obs):
        if active_corridor is not None:
            w = get_corridor_weights(work_df, transform="R")
        else:
            w = build_spatial_weights(work_df, k=5, transform="R")
    else:
        w = weights
        if w.n != n_obs:
            raise ValueError(
                f"Weights dimension ({w.n}) does not match dataset row count ({n_obs})."
            )
        if w.transform.upper() != "R":
            w.transform = "R"

    # Prepare endogenous and exogenous arrays
    y = work_df[[y_var]].values.astype(float)
    x = work_df[effective_x_vars].values.astype(float)

    effects: List[SpilloverEffect] = []

    # 5. Model estimation
    if clean_model_type == "lag":
        # Estimate Spatial Autoregressive Lag Model via 2SLS / Instrumental Variables
        model = spreg.GM_Lag(
            y,
            x,
            w=w,
            name_y=y_var,
            name_x=effective_x_vars,
            spat_diag=True,
            spat_impacts="simple",
        )

        rho = float(model.rho[0] if hasattr(model.rho, "__len__") else model.rho)
        rho_z = float(model.z_stat[-1][0])
        rho_p = float(model.z_stat[-1][1])
        r2 = float(getattr(model, "pr2", getattr(model, "r2", 0.0)))

        # LeSage & Pace (2009) Multiplier Decomposition
        spmult = spmultiplier(w, rho, method="full")
        adi = float(spmult["adi"])  # Average Direct Impact multiplier
        ati = float(spmult["ati"])  # Average Total Impact multiplier
        aii = float(spmult["aii"])  # Average Indirect Impact multiplier

        for k, var in enumerate(effective_x_vars):
            beta = float(model.betas[1 + k][0])
            se = float(model.std_err[1 + k])
            z = float(model.z_stat[1 + k][0])
            p_val = float(model.z_stat[1 + k][1])

            direct = beta * adi
            indirect = beta * aii
            total = beta * ati

            # Spillover ratio: proportion of impact attributable to cross-boundary spillover
            denom = abs(direct) + abs(indirect)
            ratio = float(abs(indirect) / denom) if denom > 1e-9 else 0.0

            interp = _generate_interpretation(var, direct, indirect, total, p_val)

            effects.append(
                SpilloverEffect(
                    variable=var,
                    direct_effect=direct,
                    indirect_effect=indirect,
                    total_effect=total,
                    spillover_ratio=ratio,
                    coef=beta,
                    std_err=se,
                    z_stat=z,
                    p_value=p_val,
                    interpretation=interp,
                )
            )

        model_summary = getattr(model, "summary", str(model))

        summary_df = pd.DataFrame([
            {
                "Variable": e.variable,
                "Direct Effect": e.direct_effect,
                "Indirect (Spillover)": e.indirect_effect,
                "Total Effect": e.total_effect,
                "Spillover Ratio": e.spillover_ratio,
                "p-value": e.p_value,
                "Interpretation": e.interpretation,
            }
            for e in effects
        ])

        return SpilloverResult(
            model_type="lag",
            y_var=y_var,
            x_vars=effective_x_vars,
            n_obs=n_obs,
            rho=rho,
            rho_p_value=rho_p,
            rho_z_stat=rho_z,
            r2=r2,
            effects=effects,
            summary_df=summary_df,
            corridor=active_corridor,
            model_summary=model_summary,
        )

    else:
        # Spatial Lag of X (SLX) Model: Y = X*beta + W*X*theta + eps
        wx = spreg.lag_spatial(w, x)
        x_slx = np.hstack([x, wx])
        slx_names = effective_x_vars + ["W_" + v for v in effective_x_vars]

        model = spreg.OLS(
            y,
            x_slx,
            name_y=y_var,
            name_x=slx_names,
        )

        r2 = float(getattr(model, "r2", 0.0))
        k_vars = len(effective_x_vars)

        for k, var in enumerate(effective_x_vars):
            beta = float(model.betas[1 + k][0])
            theta = float(model.betas[1 + k_vars + k][0])

            direct = beta
            indirect = theta
            total = direct + indirect

            se = float(model.std_err[1 + k])
            z = float(model.t_stat[1 + k][0])
            p_val = float(model.t_stat[1 + k][1])
            ind_p_val = float(model.t_stat[1 + k_vars + k][1])

            denom = abs(direct) + abs(indirect)
            ratio = float(abs(indirect) / denom) if denom > 1e-9 else 0.0

            interp = _generate_interpretation(var, direct, indirect, total, p_val)

            effects.append(
                SpilloverEffect(
                    variable=var,
                    direct_effect=direct,
                    indirect_effect=indirect,
                    total_effect=total,
                    spillover_ratio=ratio,
                    coef=beta,
                    std_err=se,
                    z_stat=z,
                    p_value=p_val,
                    interpretation=interp,
                    indirect_p_value=ind_p_val,
                )
            )

        model_summary = getattr(model, "summary", str(model))

        summary_df = pd.DataFrame([
            {
                "Variable": e.variable,
                "Direct Effect": e.direct_effect,
                "Indirect (Spillover)": e.indirect_effect,
                "Total Effect": e.total_effect,
                "Spillover Ratio": e.spillover_ratio,
                "p-value": e.p_value,
                "Interpretation": e.interpretation,
            }
            for e in effects
        ])

        return SpilloverResult(
            model_type="slx",
            y_var=y_var,
            x_vars=effective_x_vars,
            n_obs=n_obs,
            rho=None,
            rho_p_value=None,
            rho_z_stat=None,
            r2=r2,
            effects=effects,
            summary_df=summary_df,
            corridor=active_corridor,
            model_summary=model_summary,
        )
