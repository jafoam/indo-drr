"""Bivariate 3x3 Risk vs Mitigation index calculation and spatial categorization."""
from typing import Dict, List, Optional, Union
import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import BIVARIATE_MATRIX_HEX

# Negative / penalty indicators where higher values represent greater vulnerability / burden
NEGATIVE_INDICATORS: List[str] = [
    "SINYAL_BURUK",
    "KEMISKINAN",
    "TERDAMPAK",
    "ADA_BENCANA",
]

BIVARIATE_CATEGORY_LABELS: Dict[str, str] = {
    "R3-M1": "Critical Gap",
    "R3-M2": "High Risk / Moderate Mitigation",
    "R3-M3": "High Risk / High Mitigation",
    "R2-M1": "Moderate Risk / Low Mitigation",
    "R2-M2": "Moderate Risk / Moderate Mitigation",
    "R2-M3": "Moderate Risk / High Mitigation",
    "R1-M1": "Low Risk / Low Mitigation",
    "R1-M2": "Low Risk / Moderate Mitigation",
    "R1-M3": "Resilient Prepared",
}

# Hybrid Threshold Matrix: Absolute normative thresholds for village mitigation capacity
MITIGATION_THRESHOLDS = (0.30, 0.60)


def _min_max_scale(series: pd.Series) -> pd.Series:
    """Normalize numeric series to [0, 1] range via min-max scaling."""
    s_min = series.min()
    s_max = series.max()
    denom = s_max - s_min
    if denom == 0 or np.isnan(denom):
        return pd.Series(0.0, index=series.index)
    return (series - s_min) / denom


def compute_composite_mitigation_score(
    df: Union[gpd.GeoDataFrame, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
    negative_indicators: Optional[List[str]] = None,
) -> pd.Series:
    """Calculate composite mitigation score ensuring negative indicators act as penalties.

    For positive indicators (e.g. EWS, RAMBU, PERKAP, RAWAT_SUNGAI):
        Normalized value contributes positively to preparedness.
    For negative/penalty indicators (e.g. SINYAL_BURUK):
        Inverted as (100.0 - value) before normalization, acting as a penalty
        (higher burden reduces preparedness index).

    Default baseline weights (DESIGN.md Section 2.3):
        0.30 * EWS + 0.25 * RAMBU + 0.25 * PERKAP + 0.20 * RAWAT_SUNGAI
    """
    if weights is None:
        weights = {
            "EWS": 0.30,
            "RAMBU": 0.25,
            "PERKAP": 0.25,
            "RAWAT_SUNGAI": 0.20,
        }
    if negative_indicators is None:
        negative_indicators = NEGATIVE_INDICATORS

    mitigation_score = pd.Series(0.0, index=df.index, dtype=float)
    total_weight = sum(weights.values())

    for ind, w in weights.items():
        if ind not in df.columns:
            continue
        series = df[ind].astype(float)
        if ind in negative_indicators:
            # Invert negative indicator: (100.0 - SINYAL_BURUK) acting as a penalty
            series = 100.0 - series
        mitigation_score += w * _min_max_scale(series)

    if total_weight > 0:
        mitigation_score = mitigation_score / total_weight

    return pd.Series(np.clip(mitigation_score, 0.0, 1.0), index=df.index)


def is_severe_preparedness_deficit(
    bivariate_category: str,
    gap_score: float,
    risk_score: float,
    mitigation_score: float,
) -> bool:
    """Check if a district exhibits severe preparedness deficit despite moderate/high mitigation.

    Triggered when bivariate_category is R3-M3 or R3-M2 and either:
        gap_score >= 0.25 OR risk_score >= 1.5 * mitigation_score.
    """
    if bivariate_category not in ("R3-M3", "R3-M2"):
        return False
    return (gap_score >= 0.25) or (risk_score >= 1.5 * mitigation_score)


def compute_bivariate_scores(
    df: Union[gpd.GeoDataFrame, pd.DataFrame],
    mitigation_weights: Optional[Dict[str, float]] = None,
    negative_indicators: Optional[List[str]] = None,
) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """Calculate composite risk, mitigation preparedness, and gap scores.

    Formulas (DESIGN.md Section 2.3):
        - Risk Score (R_i):
            0.50 * Norm(IRBI_2024) + 0.30 * Norm(ADA_BENCANA) + 0.20 * Norm(ln(TERDAMPAK + 1))
        - Mitigation Preparedness Score (M_i):
            0.30 * Norm(EWS) + 0.25 * Norm(RAMBU) + 0.25 * Norm(PERKAP) + 0.20 * Norm(RAWAT_SUNGAI)
            (Negative indicators like SINYAL_BURUK act as penalties via 100.0 - val).
        - Disaster Gap Score (G_i):
            R_i - M_i in [-1, 1]

    Terciles (q=3):
        Risk: R1 (Low), R2 (Med), R3 (High)
        Mitigation: M1 (Low), M2 (Med), M3 (High)

    Args:
        df: GeoDataFrame or DataFrame containing raw indicator columns.
        mitigation_weights: Optional custom indicator weights for mitigation composite.
        negative_indicators: Optional list of penalty indicators to invert.

    Returns:
        DataFrame or GeoDataFrame copy augmented with:
            risk_score, mitigation_score, gap_score, risk_tercile, mitigation_tercile,
            bivariate_category, bivariate_color, bivariate_label, is_critical_gap,
            is_resilient, has_preparedness_deficit.
    """
    required_indicators = [
        "IRBI_2024",
        "ADA_BENCANA",
        "TERDAMPAK",
        "EWS",
        "RAMBU",
        "PERKAP",
        "RAWAT_SUNGAI",
    ]
    missing = [c for c in required_indicators if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required indicator columns for bivariate scoring: {missing}")

    result = df.copy()

    # 1. Min-max normalization for Risk components
    norm_irbi = _min_max_scale(result["IRBI_2024"].astype(float))
    norm_ada = _min_max_scale(result["ADA_BENCANA"].astype(float))
    log_terdampak = np.log1p(np.maximum(result["TERDAMPAK"].astype(float), 0))
    norm_terdampak = _min_max_scale(log_terdampak)

    risk_score = 0.50 * norm_irbi + 0.30 * norm_ada + 0.20 * norm_terdampak
    # Ensure numerical stability within [0, 1]
    result["risk_score"] = np.clip(risk_score, 0.0, 1.0)

    # 2. Min-max normalization for Mitigation components (penalties handled for negative indicators)
    mitigation_score = compute_composite_mitigation_score(
        result,
        weights=mitigation_weights,
        negative_indicators=negative_indicators,
    )
    result["mitigation_score"] = mitigation_score

    # 3. Gap Score & Net Readiness Balance
    result["gap_score"] = np.clip(result["risk_score"] - result["mitigation_score"], -1.0, 1.0)
    result["readiness_balance"] = np.clip(result["mitigation_score"] - result["risk_score"], -1.0, 1.0)
    result["adequacy_ratio"] = np.where(
        result["risk_score"] > 0,
        np.clip((result["mitigation_score"] / result["risk_score"]) * 100.0, 0.0, 200.0),
        100.0,
    )

    # 4. Hybrid Thresholds: Empirical tertiles for Risk, Normative thresholds for Mitigation
    r_labels = ["R1", "R2", "R3"]
    m_labels = ["M1", "M2", "M3"]

    result["risk_tercile"] = pd.qcut(
        result["risk_score"],
        q=3,
        labels=r_labels,
        duplicates="drop",
    ).astype(str)

    # Mitigation Dimension: Absolute normative thresholds:
    # M1 (Low / Defisit Parah): < 0.30
    # M2 (Moderate / Siap Parsial): 0.30 - 0.60
    # M3 (High / Kapasitas Unggul): > 0.60
    mit_conditions = [
        result["mitigation_score"] < MITIGATION_THRESHOLDS[0],
        (result["mitigation_score"] >= MITIGATION_THRESHOLDS[0]) & (result["mitigation_score"] <= MITIGATION_THRESHOLDS[1]),
        result["mitigation_score"] > MITIGATION_THRESHOLDS[1],
    ]
    result["mitigation_tercile"] = np.select(
        mit_conditions,
        m_labels,
        default="M1",
    ).astype(str)

    # 5. Bivariate 3x3 category
    result["bivariate_category"] = (
        result["risk_tercile"] + "-" + result["mitigation_tercile"]
    )

    # 6. Map hex colors and human-readable labels
    result["bivariate_color"] = result["bivariate_category"].map(
        lambda cat: BIVARIATE_MATRIX_HEX.get(
            cat,
            BIVARIATE_MATRIX_HEX.get(tuple(cat.split("-")), "#334155"),
        )
    )

    result["bivariate_label"] = result["bivariate_category"].map(
        lambda cat: BIVARIATE_CATEGORY_LABELS.get(cat, cat)
    )

    result["is_critical_gap"] = result["bivariate_category"] == "R3-M1"
    result["is_resilient"] = result["bivariate_category"] == "R1-M3"

    # 7. Gap-aware severe preparedness deficit qualification
    result["has_preparedness_deficit"] = (
        result["bivariate_category"].isin(["R3-M3", "R3-M2"])
        & (
            (result["gap_score"] >= 0.25)
            | (result["risk_score"] >= 1.5 * result["mitigation_score"])
        )
    )

    return result
