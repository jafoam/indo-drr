"""Micro-Profile District Inspector Module.

Computes multi-dimensional disaster readiness, mitigation, and vulnerability profiles
for individual administrative districts (Kabupaten/Kota), benchmarking 7 micro-indicators
against both corridor-level and national-level distributions.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from src.analytics.corridors import assign_corridors

# 7 Core Micro-Indicators for District Inspection
PROFILE_INDICATORS: List[str] = [
    "EWS",
    "RAMBU",
    "PERKAP",
    "RAWAT_SUNGAI",
    "GOTONG_ROYONG",
    "SINYAL_BURUK",
    "IPM",
]

# Negative / penalty indicators where higher values represent greater vulnerability / burden
NEGATIVE_INDICATORS: List[str] = [
    "SINYAL_BURUK",
    "KEMISKINAN",
    "TERDAMPAK",
    "ADA_BENCANA",
]

INDICATOR_METADATA: Dict[str, Dict[str, str]] = {
    "EWS": {
        "label": "Early Warning System (%)",
        "description": "% villages with disaster early warning systems",
    },
    "RAMBU": {
        "label": "Evacuation Route Signage (%)",
        "description": "% villages equipped with evacuation route signage",
    },
    "PERKAP": {
        "label": "Safety Equipment (Boats, Tents, Gear) (%)",
        "description": "% villages equipped with disaster safety equipment (boats, tents, gear)",
    },
    "RAWAT_SUNGAI": {
        "label": "Water Area Maintenance (Rivers/Canals/Dykes) (%)",
        "description": "% villages maintaining rivers, canals, dykes, and related water infrastructure",
    },
    "RAWAT_AIR": {
        "label": "Water Area Maintenance (Rivers/Canals/Dykes) (%)",
        "description": "% villages maintaining rivers, canals, dykes, and related water infrastructure",
    },
    "ADA_BENCANA": {
        "label": "Disaster Occurrence (2023-2024) (%)",
        "description": "% villages experiencing disaster during 2023-April 2024",
    },
    "GOTONG_ROYONG": {
        "label": "Mutual Aid Participation (%)",
        "description": "Community mutual aid and solidarity participation index",
    },
    "SINYAL_BURUK": {
        "label": "Weak Telecom Coverage (%)",
        "description": "% villages with weak or non-existent cellular signal",
    },
    "IPM": {
        "label": "Human Development Index",
        "description": "Composite index of health, education, and standard of living",
    },
}


@dataclass
class IndicatorMetric:
    """Benchmark metric for an individual district indicator."""
    name: str
    label: str
    value: float
    national_pct: float
    corridor_pct: float
    national_median: float
    corridor_median: float
    description: str = ""
    is_negative: bool = False
    effective_national_pct: Optional[float] = None
    effective_corridor_pct: Optional[float] = None

    def __post_init__(self):
        if self.effective_national_pct is None:
            self.effective_national_pct = round(100.0 - self.national_pct, 2) if self.is_negative else self.national_pct
        if self.effective_corridor_pct is None:
            self.effective_corridor_pct = round(100.0 - self.corridor_pct, 2) if self.is_negative else self.corridor_pct

    @property
    def indicator(self) -> str:
        """Alias for name."""
        return self.name

    @property
    def raw_value(self) -> float:
        """Alias for value."""
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metric to dictionary."""
        return {
            "name": self.name,
            "label": self.label,
            "value": self.value,
            "national_pct": self.national_pct,
            "corridor_pct": self.corridor_pct,
            "national_median": self.national_median,
            "corridor_median": self.corridor_median,
            "description": self.description,
            "is_negative": self.is_negative,
            "effective_national_pct": self.effective_national_pct,
            "effective_corridor_pct": self.effective_corridor_pct,
        }


@dataclass
class RegencyProfile:
    """Complete diagnostic micro-profile for a regency / city."""
    kode_wil: int
    kab_kota: str
    provinsi: str
    corridor: str
    is_kota: int
    risk_score: float
    mitigation_score: float
    gap_score: float
    gap_rank: int
    bivariate_category: str
    bivariate_label: str
    bivariate_color: str
    indicators: Dict[str, IndicatorMetric] = field(default_factory=dict)
    has_preparedness_deficit: bool = False
    readiness_balance: float = 0.0
    adequacy_ratio: float = 100.0

    @property
    def regency_name(self) -> str:
        """Alias for kab_kota."""
        return self.kab_kota

    @property
    def province(self) -> str:
        """Alias for provinsi."""
        return self.provinsi

    @property
    def national_rank(self) -> int:
        """Alias for gap_rank."""
        return self.gap_rank

    @property
    def metrics(self) -> List[IndicatorMetric]:
        """List representation of indicators."""
        return list(self.indicators.values())

    def __getitem__(self, key: str) -> IndicatorMetric:
        """Direct dictionary access for indicators."""
        return self.indicators[key]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            "kode_wil": self.kode_wil,
            "kab_kota": self.kab_kota,
            "provinsi": self.provinsi,
            "corridor": self.corridor,
            "is_kota": self.is_kota,
            "risk_score": self.risk_score,
            "mitigation_score": self.mitigation_score,
            "gap_score": self.gap_score,
            "gap_rank": self.gap_rank,
            "bivariate_category": self.bivariate_category,
            "bivariate_label": self.bivariate_label,
            "bivariate_color": self.bivariate_color,
            "has_preparedness_deficit": self.has_preparedness_deficit,
            "readiness_balance": self.readiness_balance,
            "adequacy_ratio": self.adequacy_ratio,
            "indicators": {k: v.to_dict() for k, v in self.indicators.items()},
        }


def compute_regency_profile(
    gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
    bivariate_df: pd.DataFrame,
    kode_wil: Union[int, str],
) -> RegencyProfile:
    """Compute 7-indicator benchmark micro-profile for a target regency.

    Args:
        gdf: GeoDataFrame or DataFrame containing spatial geometries and regency attributes.
        bivariate_df: DataFrame containing computed bivariate scores, gap ranks, and indicators.
        kode_wil: Target BPS 4-digit code (e.g. 1101, 3171).

    Returns:
        RegencyProfile populated with district metrics and benchmarking percentiles.

    Raises:
        KeyError: If kode_wil is not found in the datasets.
    """
    try:
        kw = int(kode_wil)
    except (ValueError, TypeError):
        raise KeyError(f"Invalid KODE_WIL format: {kode_wil}")

    # Build reference dataset combining bivariate scores and raw indicators
    df = bivariate_df.copy()
    if "KODE_WIL" not in df.columns:
        raise KeyError("bivariate_df missing 'KODE_WIL' column")

    df["KODE_WIL"] = df["KODE_WIL"].astype(int)

    # Check if target regency exists in bivariate_df
    target_match = df[df["KODE_WIL"] == kw]
    if target_match.empty:
        # Check in gdf
        if "KODE_WIL" in gdf.columns:
            gdf_copy = gdf.copy()
            gdf_copy["KODE_WIL"] = gdf_copy["KODE_WIL"].astype(int)
            target_gdf = gdf_copy[gdf_copy["KODE_WIL"] == kw]
            if target_gdf.empty:
                raise KeyError(f"Regency with KODE_WIL {kw} not found")
        else:
            raise KeyError(f"Regency with KODE_WIL {kw} not found")

    # If any indicator is missing from bivariate_df but present in gdf, merge it
    missing_indicators = [ind for ind in PROFILE_INDICATORS if ind not in df.columns]
    if missing_indicators and "KODE_WIL" in gdf.columns:
        gdf_cols = ["KODE_WIL"] + [c for c in missing_indicators if c in gdf.columns]
        if len(gdf_cols) > 1:
            gdf_subset = gdf[gdf_cols].copy()
            gdf_subset["KODE_WIL"] = gdf_subset["KODE_WIL"].astype(int)
            df = df.merge(gdf_subset, on="KODE_WIL", how="left")

    # Ensure corridor column is present
    if "corridor" not in df.columns:
        df = assign_corridors(df)

    # Re-fetch target row
    target_rows = df[df["KODE_WIL"] == kw]
    if target_rows.empty:
        raise KeyError(f"Regency with KODE_WIL {kw} not found")
    target_row = target_rows.iloc[0]

    # Calculate national rank on gap_score (Rank 1 = acute vulnerability / highest gap)
    if "gap_score" in df.columns:
        ranks = df["gap_score"].rank(ascending=False, method="min").astype(int)
        df["_gap_rank"] = ranks
        gap_rank = int(df.loc[df["KODE_WIL"] == kw, "_gap_rank"].values[0])
    else:
        gap_rank = 1

    corridor_name = str(target_row["corridor"])
    corridor_df = df[df["corridor"] == corridor_name]

    # Compute 7 indicator metrics
    indicators: Dict[str, IndicatorMetric] = {}
    for ind in PROFILE_INDICATORS:
        if ind not in df.columns:
            raise ValueError(f"Required indicator column '{ind}' missing from dataset")

        raw_val = float(target_row[ind])
        nat_series = df[ind].dropna()
        corr_series = corridor_df[ind].dropna()

        # National percentile (kind='rank' computes percentage of values <= raw_val)
        nat_pct = float(percentileofscore(nat_series, raw_val, kind="rank"))
        corr_pct = float(percentileofscore(corr_series, raw_val, kind="rank"))

        # Guarantee bounds [0.0, 100.0]
        nat_pct = max(0.0, min(100.0, round(nat_pct, 2)))
        corr_pct = max(0.0, min(100.0, round(corr_pct, 2)))

        nat_median = float(nat_series.median())
        corr_median = float(corr_series.median())

        meta = INDICATOR_METADATA.get(ind, {"label": ind, "description": ""})
        is_neg = ind in NEGATIVE_INDICATORS
        eff_nat_pct = round(100.0 - nat_pct, 2) if is_neg else nat_pct
        eff_corr_pct = round(100.0 - corr_pct, 2) if is_neg else corr_pct

        indicators[ind] = IndicatorMetric(
            name=ind,
            label=meta["label"],
            value=raw_val,
            national_pct=nat_pct,
            corridor_pct=corr_pct,
            national_median=nat_median,
            corridor_median=corr_median,
            description=meta["description"],
            is_negative=is_neg,
            effective_national_pct=eff_nat_pct,
            effective_corridor_pct=eff_corr_pct,
        )

    # Extract regency metadata
    kab_kota = str(target_row.get("KAB_KOTA", target_row.get("KAB_KOTA_NAME", f"District {kw}")))
    provinsi = str(target_row.get("PROVINSI", "Unknown"))
    is_kota = int(target_row.get("IS_KOTA", 0))

    risk_score = float(target_row.get("risk_score", 0.0))
    mitigation_score = float(target_row.get("mitigation_score", 0.0))
    gap_score = float(target_row.get("gap_score", 0.0))

    biv_cat = str(target_row.get("bivariate_category", "R2-M2"))
    biv_label = str(target_row.get("bivariate_label", "Moderate Risk / Moderate Mitigation"))
    biv_color = str(target_row.get("bivariate_color", "#bfa29b"))

    has_deficit = bool(target_row.get("has_preparedness_deficit", False))
    if not has_deficit and biv_cat in ("R3-M3", "R3-M2"):
        has_deficit = (gap_score >= 0.25) or (risk_score >= 1.5 * mitigation_score)

    readiness_balance = float(target_row.get("readiness_balance", mitigation_score - risk_score))
    default_adequacy = (mitigation_score / risk_score * 100.0) if risk_score > 0 else 100.0
    adequacy_ratio = float(target_row.get("adequacy_ratio", min(200.0, max(0.0, default_adequacy))))

    return RegencyProfile(
        kode_wil=kw,
        kab_kota=kab_kota,
        provinsi=provinsi,
        corridor=corridor_name,
        is_kota=is_kota,
        risk_score=risk_score,
        mitigation_score=mitigation_score,
        gap_score=gap_score,
        gap_rank=gap_rank,
        bivariate_category=biv_cat,
        bivariate_label=biv_label,
        bivariate_color=biv_color,
        indicators=indicators,
        has_preparedness_deficit=has_deficit,
        readiness_balance=readiness_balance,
        adequacy_ratio=adequacy_ratio,
    )
