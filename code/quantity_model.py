"""Reproduce an external section benchmark and evaluate explicit quantity scenarios.

This is a section-capacity calculation, not a member or building approval.
All design scenarios are analyst-defined, not observed construction projects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_epd_choices import MENU_KEYS, best


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "derived"
FC_REFERENCE_PSI = 5000.0
FY_PSI = 60000.0
ES_PSI = 29000000.0
AS_IN2 = 8.0
B_REFERENCE_IN = 16.0
PHI_COMPRESSION = 0.65
AXIAL_LIMIT = 0.80


def axial_nominal_kip(fc_psi, area_in2, steel_in2=AS_IN2):
    return (0.85 * fc_psi * (area_in2 - steel_in2) + FY_PSI * steel_in2) / 1000


PU_KIP = PHI_COMPRESSION * AXIAL_LIMIT * axial_nominal_kip(FC_REFERENCE_PSI, B_REFERENCE_IN**2)


def section_point(c_in, fc_psi=FC_REFERENCE_PSI, b_in=B_REFERENCE_IN, h_in=B_REFERENCE_IN):
    beta = max(0.65, 0.85 - 0.05 * max(0.0, fc_psi - 4000.0) / 1000.0)
    a_in = min(beta * c_in, h_in)
    y = np.array([2.5, h_in - 2.5])
    areas = np.array([4.0, 4.0])
    strain = 0.003 * (1.0 - y / c_in)
    stress = np.clip(ES_PSI * strain, -FY_PSI, FY_PSI)
    forces = areas * (stress - np.where(y <= a_in, 0.85 * fc_psi, 0.0)) / 1000.0
    concrete = 0.85 * fc_psi * b_in * a_in / 1000.0
    axial = concrete + forces.sum()
    moment = (concrete * (h_in / 2.0 - a_in / 2.0) + np.dot(forces, h_in / 2.0 - y)) / 12.0
    tensile_strain = max(0.0, -strain[-1])
    phi = float(np.clip(0.65 + 0.25 * (tensile_strain - FY_PSI / ES_PSI) / 0.003, 0.65, 0.90))
    return {"c_in": c_in, "nominal_P_kip": axial, "nominal_M_kip_ft": moment,
            "phi": phi, "design_P_kip": phi * axial, "design_M_kip_ft": phi * moment}


def external_benchmark():
    rows = []
    nominal = axial_nominal_kip(5000.0, 256.0)
    for name, calculated, reported in (
        ("nominal_axial_kip", nominal, 1534.0),
        ("factored_axial_kip", 0.65 * nominal, 997.1),
        ("capped_axial_kip", 0.80 * 0.65 * nominal, 797.7),
    ):
        rows.append({"point": name, "quantity": "P_kip", "calculated": calculated,
                     "published": reported, "published_pdf_page": 7})
    ey = FY_PSI / ES_PSI
    conditions = [
        ("zero_far_bar_stress", 0.0, 622.3, 169.86),
        ("half_yield_far_bar", 0.5 * ey, 421.9, 220.05),
        ("balanced", ey, 270.9, 250.77),
        ("tension_controlled", ey + 0.003, 171.6, 286.75),
    ]
    for name, et, pref, mref in conditions:
        point = section_point(13.5 * 0.003 / (0.003 + et))
        for quantity, value, ref in (("P_kip", point["design_P_kip"], pref),
                                    ("M_kip_ft", point["design_M_kip_ft"], mref)):
            rows.append({"point": name, "quantity": quantity, "calculated": value,
                         "published": ref, "published_pdf_page": 29})
    low, high = 0.1, 13.5
    for _ in range(100):
        mid = (low + high) / 2
        if section_point(mid)["nominal_P_kip"] < 0:
            low = mid
        else:
            high = mid
    pure = section_point((low + high) / 2)
    for quantity, value, ref in (("P_kip", pure["design_P_kip"], 0.0),
                                ("M_kip_ft", pure["design_M_kip_ft"], 213.96)):
        rows.append({"point": "pure_bending", "quantity": quantity, "calculated": value,
                     "published": ref, "published_pdf_page": 29})
    rows.append({"point": "pure_tension", "quantity": "P_kip", "calculated": 0.9 * FY_PSI * AS_IN2 / 1000,
                 "published": 432.0, "published_pdf_page": 29})
    result = pd.DataFrame(rows)
    result["absolute_difference"] = (result.calculated - result.published).abs()
    result["within_0_1_reported_units"] = result.absolute_difference.lt(0.1)
    if not result.within_0_1_reported_units.all():
        raise AssertionError(result.to_string(index=False))
    result.to_csv(OUT / "external_section_benchmark.csv", index=False)
    return result


def net_concrete_area(fc_psi, minimum_width_in, pu_kip=PU_KIP):
    fc = np.asarray(fc_psi, dtype=float)
    if np.any(fc <= 0) or minimum_width_in**2 <= AS_IN2:
        raise ValueError("Positive strength and a gross-area floor above the steel area are required")
    k = (pu_kip * 1000 / (PHI_COMPRESSION * AXIAL_LIMIT) - FY_PSI * AS_IN2) / 0.85
    if k <= 0:
        raise ValueError("Concrete must contribute positively to this specified axial demand")
    return np.maximum(k / fc, minimum_width_in**2 - AS_IN2)


def saturation_strength(minimum_width_in, pu_kip=PU_KIP):
    k = (pu_kip * 1000 / (PHI_COMPRESSION * AXIAL_LIMIT) - FY_PSI * AS_IN2) / 0.85
    return k / (minimum_width_in**2 - AS_IN2)


def evaluate_catalogues():
    data = pd.read_csv(OUT / "analysis_records.csv", keep_default_na=False)
    # The quantity rule is evaluated on the manuscript's strict primary
    # cohort, not on the concatenated plant/component/description tables.
    cases = pd.read_csv(OUT / "choices_description_series_1000.csv")
    groups = {k: g for k, g in data.groupby(MENU_KEYS, sort=False)}
    by_id = data.set_index("row_id")
    comparisons = []
    # These geometry floors and demand ratios are declared sensitivity scenarios.
    for width in (12.0, 14.0, 16.0, 18.0):
        for demand_ratio in (0.75, 1.0, 1.25):
            pu = PU_KIP * demand_ratio
            fstar = saturation_strength(width, pu)
            for case in cases.itertuples():
                base = by_id.loc[case.baseline_id]
                group = groups[tuple(base[k] for k in MENU_KEYS)]
                pool = group[
                    group.age_days.eq(28) & group.strength_psi.between(case.target_psi, case.target_psi + 1000)
                    & group.components.eq(base.components) & group.description_series.eq(base.description_series)
                ].copy()
                areas = net_concrete_area(pool.strength_psi, width, pu)
                pool["concrete_area"] = areas
                pool["carbon_per_length"] = pool.gwp * areas
                pool["proxy"] = pool.gwp / pool.strength_psi
                pool["capped_proxy"] = pool.gwp / np.minimum(pool.strength_psi, fstar)
                scalar = lambda col: best(pool[np.isclose(pool[col], pool[col].min(), rtol=1e-12, atol=1e-10)])
                actual, ordinary, capped = scalar("carbon_per_length"), scalar("proxy"), scalar("capped_proxy")
                if not np.isclose(actual.carbon_per_length, capped.carbon_per_length, rtol=1e-12):
                    raise AssertionError("Capped index must match the explicit fixed-steel quantity calculation")
                comparisons.append({
                    "menu_id": case.menu_id, "target_psi": case.target_psi, "company": case.company,
                    "minimum_width_in": width, "demand_ratio": demand_ratio, "demand_kip": pu,
                    "saturation_strength_psi": fstar, "quantity_selected_id": int(actual.row_id),
                    "ordinary_selected_id": int(ordinary.row_id), "capped_selected_id": int(capped.row_id),
                    "ordinary_carbon_excess": ordinary.carbon_per_length / actual.carbon_per_length - 1,
                    "ordinary_floor_active": bool(ordinary.strength_psi >= fstar),
                    "best_concrete_area_in2": actual.concrete_area,
                    "ordinary_concrete_area_in2": ordinary.concrete_area,
                    "ordinary_gwp": ordinary.gwp, "quantity_gwp": actual.gwp,
                })
    results = pd.DataFrame(comparisons)
    results.to_csv(OUT / "quantity_model_scenarios.csv", index=False)
    summaries = []
    for key, frame in results.groupby(["minimum_width_in", "demand_ratio"]):
        bad = frame.ordinary_carbon_excess.gt(1e-9)
        summaries.append({"minimum_width_in": key[0], "demand_ratio": key[1], "cases": len(frame),
                          "affected_cases": int(bad.sum()), "affected_fraction": float(bad.mean()),
                          "median_excess_affected": float(frame.loc[bad, "ordinary_carbon_excess"].median()) if bad.any() else 0,
                          "saturation_strength_psi": float(frame.saturation_strength_psi.iloc[0]),
                          "capped_quantity_identity": True})
    pd.DataFrame(summaries).to_csv(OUT / "quantity_model_summary.csv", index=False)
    return summaries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark = external_benchmark()
    summary = evaluate_catalogues()
    result = {"external_check": "14 section-capacity outputs against StructurePoint 2022, PDF pages 7 and 29",
              "max_absolute_benchmark_difference": float(benchmark.absolute_difference.max()),
              "benchmark_design_axial_kip": PU_KIP,
              "model_boundary": "fixed longitudinal steel area and equal length; axial section capacity plus imposed geometry floor; no member, durability, seismic, fire or construction qualification",
              "scenarios": summary}
    (OUT / "quantity_model_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
