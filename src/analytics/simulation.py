"""Interactive Policy Intervention Sandbox analytics engine.

Simulates the impact of targeted infrastructure and social capital interventions
on disaster mitigation readiness, recalculating composite mitigation scores,
disaster gap scores, and bivariate 3x3 risk-preparedness classifications.
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union
import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import BIVARIATE_MATRIX_HEX
from src.analytics.bivariate import (
    _min_max_scale,
    compute_bivariate_scores,
    BIVARIATE_CATEGORY_LABELS,
    MITIGATION_THRESHOLDS,
)


@dataclass
class SimulationResult:
    """Dataclass encapsulating policy simulation outcomes and adjusted dataset."""
    baseline_critical_gaps: int
    simulated_critical_gaps: int
    resolved_gaps: int
    df: Union[gpd.GeoDataFrame, pd.DataFrame]

    @property
    def data(self) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """Alias for df for API compatibility."""
        return self.df

    @property
    def simulated_df(self) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """Alias for df for API compatibility."""
        return self.df

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style key indexing."""
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        """Summary dictionary without raw tabular payload."""
        return {
            "baseline_critical_gaps": int(self.baseline_critical_gaps),
            "simulated_critical_gaps": int(self.simulated_critical_gaps),
            "resolved_gaps": int(self.resolved_gaps),
        }


def simulate_policy_intervention(
    df: Union[gpd.GeoDataFrame, pd.DataFrame],
    signal_reduction_pct: float = 0.0,
    gotong_royong_boost_pct: float = 0.0,
) -> SimulationResult:
    """Simulate policy interventions on digital infrastructure and social capital.

    Bounded, grounded policy simulation formula (DESIGN.md Section 2.3 & Milestone 2 Plan):
        Adjusts mitigation score upwards based on signal coverage improvement and
        community mutual aid mobilization:
            M_i^{adj} = clip(
                M_i
                + 0.15 * (Delta_signal / 100) * (1 - Norm(SINYAL_BURUK_i))
                + 0.15 * (Delta_gotong / 100) * Norm(GOTONG_ROYONG_i),
                0.0, 1.0
            )
        Recalculates:
            G_i^{adj} = clip(R_i - M_i^{adj}, -1.0, 1.0)

    Tercile classification for M_i^{adj} is evaluated against baseline tercile cutoffs
    to ensure absolute progress from M1 (Low) to M2 (Moderate) / M3 (High).

    Args:
        df: GeoDataFrame or DataFrame containing baseline indicators or bivariate scores.
        signal_reduction_pct: Percentage reduction in weak/no-signal villages [0, 100].
        gotong_royong_boost_pct: Percentage increase in community mutual aid [0, 100].

    Returns:
        SimulationResult containing baseline and simulated gap counts and augmented DataFrame.
    """
    # Ensure baseline bivariate scores exist
    if "mitigation_score" not in df.columns or "risk_score" not in df.columns:
        work_df = compute_bivariate_scores(df)
    else:
        work_df = df.copy()

    # Clamp input percentage parameters to [0.0, 100.0]
    delta_signal = float(np.clip(signal_reduction_pct, 0.0, 100.0))
    delta_gotong = float(np.clip(gotong_royong_boost_pct, 0.0, 100.0))

    baseline_critical_gaps = int(
        work_df["is_critical_gap"].sum()
        if "is_critical_gap" in work_df.columns
        else (work_df["bivariate_category"] == "R3-M1").sum()
    )

    # Baseline case (0%, 0%)
    if delta_signal == 0.0 and delta_gotong == 0.0:
        work_df["simulated_mitigation_score"] = work_df["mitigation_score"].copy()
        work_df["simulated_gap_score"] = work_df["gap_score"].copy()
        work_df["simulated_mitigation_tercile"] = work_df["mitigation_tercile"].copy()
        work_df["simulated_bivariate_category"] = work_df["bivariate_category"].copy()
        work_df["simulated_bivariate_color"] = work_df["bivariate_color"].copy()
        work_df["simulated_bivariate_label"] = (
            work_df["bivariate_label"].copy()
            if "bivariate_label" in work_df.columns
            else work_df["simulated_bivariate_category"].map(
                lambda c: BIVARIATE_CATEGORY_LABELS.get(c, c)
            )
        )
        work_df["is_critical_gap_simulated"] = work_df["is_critical_gap"].copy()
        work_df["is_resilient_simulated"] = work_df["is_resilient"].copy()

        return SimulationResult(
            baseline_critical_gaps=baseline_critical_gaps,
            simulated_critical_gaps=baseline_critical_gaps,
            resolved_gaps=0,
            df=work_df,
        )

    # Compute normalized indicators for intervention weighting
    if "SINYAL_BURUK" in work_df.columns:
        norm_sinyal = _min_max_scale(work_df["SINYAL_BURUK"].astype(float))
    else:
        norm_sinyal = pd.Series(0.5, index=work_df.index)

    if "GOTONG_ROYONG" in work_df.columns:
        norm_gotong = _min_max_scale(work_df["GOTONG_ROYONG"].astype(float))
    else:
        norm_gotong = pd.Series(0.5, index=work_df.index)

    # Apply grounded intervention formula
    signal_term = 0.15 * (delta_signal / 100.0) * (1.0 - norm_sinyal)
    gotong_term = 0.15 * (delta_gotong / 100.0) * norm_gotong

    m_adj = np.clip(
        work_df["mitigation_score"].astype(float) + signal_term + gotong_term,
        0.0,
        1.0,
    )
    work_df["simulated_mitigation_score"] = m_adj

    # Recalculate gap score: R_i - M_i^{adj}
    r_score = work_df["risk_score"].astype(float)
    work_df["simulated_gap_score"] = np.clip(r_score - m_adj, -1.0, 1.0)

    # Assign simulated mitigation level against normative thresholds (M1 < 0.30, M2: 0.30-0.60, M3 > 0.60)
    sim_conds = [
        m_adj < MITIGATION_THRESHOLDS[0],
        (m_adj >= MITIGATION_THRESHOLDS[0]) & (m_adj <= MITIGATION_THRESHOLDS[1]),
        m_adj > MITIGATION_THRESHOLDS[1],
    ]
    sim_m_tercile = pd.Series(
        np.select(sim_conds, ["M1", "M2", "M3"], default="M1"),
        index=work_df.index,
        dtype=str,
    )
    work_df["simulated_mitigation_tercile"] = sim_m_tercile

    # Risk tercile remains fixed from baseline
    risk_tercile = work_df["risk_tercile"].astype(str)
    sim_category = risk_tercile + "-" + sim_m_tercile
    work_df["simulated_bivariate_category"] = sim_category

    # Map colors and labels
    work_df["simulated_bivariate_color"] = sim_category.map(
        lambda cat: BIVARIATE_MATRIX_HEX.get(
            cat,
            BIVARIATE_MATRIX_HEX.get(tuple(cat.split("-")), "#334155"),
        )
    )
    work_df["simulated_bivariate_label"] = sim_category.map(
        lambda cat: BIVARIATE_CATEGORY_LABELS.get(cat, cat)
    )

    work_df["is_critical_gap_simulated"] = sim_category == "R3-M1"
    work_df["is_resilient_simulated"] = sim_category == "R1-M3"

    simulated_critical_gaps = int(work_df["is_critical_gap_simulated"].sum())
    resolved_gaps = baseline_critical_gaps - simulated_critical_gaps

    return SimulationResult(
        baseline_critical_gaps=baseline_critical_gaps,
        simulated_critical_gaps=simulated_critical_gaps,
        resolved_gaps=resolved_gaps,
        df=work_df,
    )
