"""Provincial Profile Aggregation Engine for IDR-GE.

Aggregates multi-dimensional disaster readiness, mitigation capacity, and vulnerability
metrics across regencies and cities (Kabupaten/Kota) within a given province.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import geopandas as gpd

from src.analytics.corridors import CORRIDOR_MAPPING
from src.analytics.profile import PROFILE_INDICATORS, NEGATIVE_INDICATORS


def compute_provincial_profile(
    gdf: gpd.GeoDataFrame,
    provinsi_name: str,
    bivariate_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compute aggregated multi-dimensional disaster readiness profile for a province.

    Args:
        gdf: GeoDataFrame containing 514 districts with risk, mitigation, and indicator scores.
        provinsi_name: Target province name (matches 'PROVINSI' column, case-insensitive).
        bivariate_df: Optional DataFrame with bivariate scores if not already attached to gdf.

    Returns:
        Dict containing provincial aggregates, critical unit counts, indicators, and top bottleneck districts.
    """
    if "PROVINSI" not in gdf.columns:
        raise ValueError("GeoDataFrame must contain 'PROVINSI' column.")

    matches = gdf[gdf["PROVINSI"].astype(str).str.strip().str.upper() == provinsi_name.strip().upper()]
    if matches.empty:
        raise ValueError(f"Province '{provinsi_name}' not found in dataset.")

    canonical_name = str(matches["PROVINSI"].iloc[0])
    total_units = int(len(matches))

    corridor = str(
        matches["corridor"].iloc[0]
        if "corridor" in matches.columns
        else CORRIDOR_MAPPING.get(canonical_name, "National")
    )

    # Risk and mitigation aggregates
    risk_score = float(matches["risk_score"].mean())
    mitigation_score = float(matches["mitigation_score"].mean())
    gap_score = float(risk_score - mitigation_score)
    readiness_balance = float(mitigation_score - risk_score)
    adequacy_ratio = float((mitigation_score / risk_score * 100.0) if risk_score > 0 else 100.0)

    # Critical district count (R3-M1 or high gap)
    crit_mask = matches["bivariate_category"].astype(str).str.contains("R3-M1", na=False)
    critical_count = int(crit_mask.sum())
    critical_share_pct = round((critical_count / total_units) * 100.0, 1) if total_units > 0 else 0.0

    # Provincial readiness gap ranking among all 38 provinces
    prov_gaps = gdf.groupby("PROVINSI")["gap_score"].mean().sort_values(ascending=False)
    try:
        gap_rank = int(prov_gaps.index.get_loc(canonical_name) + 1)
    except Exception:
        gap_rank = 1
    total_provinces = int(len(prov_gaps))

    # Top 3-5 priority bottleneck districts within the province
    top_bottlenecks = []
    sorted_sub = matches.sort_values(by="gap_score", ascending=False).head(5)
    for _, row in sorted_sub.iterrows():
        rk = float(row["risk_score"])
        mk = float(row["mitigation_score"])
        gk = float(row.get("gap_score", rk - mk))
        bal = float(row.get("readiness_balance", mk - rk))
        ad = float(row.get("adequacy_ratio", (mk / rk * 100.0) if rk > 0 else 100.0))
        top_bottlenecks.append({
            "kode_wil": int(row["KODE_WIL"]),
            "kab_kota": str(row.get("KAB_KOTA", row.get("kab_kota", f"District {row['KODE_WIL']}"))),
            "risk_score": rk,
            "mitigation_score": mk,
            "gap_score": gk,
            "readiness_balance": bal,
            "adequacy_ratio": ad,
            "bivariate_category": str(row.get("bivariate_category", "R3-M1")),
        })

    # Aggregated 7 PODES Micro-Indicators
    corr_slice = gdf[gdf["corridor"] == corridor] if "corridor" in gdf.columns else gdf
    indicators = {}
    for code in PROFILE_INDICATORS:
        if code in matches.columns:
            p_mean = float(matches[code].mean())
            p_med = float(matches[code].median())
        else:
            p_mean = 0.0
            p_med = 0.0

        c_med = float(corr_slice[code].median()) if code in corr_slice.columns else 0.0
        n_med = float(gdf[code].median()) if code in gdf.columns else 0.0

        indicators[code] = {
            "value": p_mean,
            "provincial_median": p_med,
            "corridor_median": c_med,
            "national_median": n_med,
            "is_negative": code in NEGATIVE_INDICATORS,
        }

    return {
        "provinsi": canonical_name,
        "corridor": corridor,
        "total_units": total_units,
        "risk_score": risk_score,
        "mitigation_score": mitigation_score,
        "gap_score": gap_score,
        "readiness_balance": readiness_balance,
        "adequacy_ratio": adequacy_ratio,
        "gap_rank": gap_rank,
        "total_provinces": total_provinces,
        "critical_count": critical_count,
        "critical_share_pct": critical_share_pct,
        "top_bottlenecks": top_bottlenecks,
        "indicators": indicators,
    }
