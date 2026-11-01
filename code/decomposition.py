"""Keep the three comparisons separate.

C1  candidate-set expansion: fixed-volume choice from the exact-target set E
    versus the fixed-volume choice from the 1000 psi window W.
C2  objective choice: ordinary ratio g/f versus fixed volume g, both on W.
C3  quantity constraint: whether the capped selector displaces the ratio
    selector, restricted to the C2 cases where the two conventions disagree.
    C3' reports the matching displacement of fixed volume by the capped
    selector. The headline C3/C3' counts are selector-decision counts; the
    former carbon-suboptimality diagnostic is retained separately so that the
    denominator is explicit and the high-cap limit is auditable.

The script also checks Proposition 2 of the revised manuscript on every case
and every transition strength in a sweep: the capped choice lies on the
strength-GWP frontier between the fixed-volume and ratio choices, so it can
only change a decision where those two conventions disagree.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import core as rc

GRID_WIDTHS_IN = (12.0, 14.0, 16.0, 18.0)
GRID_DEMANDS = (0.75, 1.0, 1.25)
REF_STEEL_IN2 = 8.0
# StructurePoint reference section: 16 in square, 5000 psi, 8 in^2, phi 0.65, cap 0.80.
PU_REF_KIP = rc.PHI_COMPRESSION * rc.AXIAL_LIMIT * (0.85 * 5000.0 * (256.0 - 8.0) + 60000.0 * 8.0) / 1000.0
EPS = 1e-9


def base_comparisons(c: rc.Candidates) -> pd.DataFrame:
    fv_e = rc.select_fixed_volume(c, c.exact)
    fv = rc.select_fixed_volume(c)
    ratio = rc.select_ratio(c)
    g_e, g_fv, g_r = c.take(fv_e, c.gwp), c.take(fv, c.gwp), c.take(ratio, c.gwp)
    frame = c.cases[["menu_id", "target_psi", "company", "plant_issue_id"]].copy()
    frame["n_window"] = c.valid.sum(axis=1)
    frame["n_exact"] = c.exact.sum(axis=1)
    frame["exact_fv_row"] = c.take(fv_e, c.row_id)
    frame["window_fv_row"] = c.take(fv, c.row_id)
    frame["ratio_row"] = c.take(ratio, c.row_id)
    frame["exact_fv_gwp"], frame["window_fv_gwp"], frame["ratio_gwp"] = g_e, g_fv, g_r
    frame["window_fv_strength"] = c.take(fv, c.strength)
    frame["ratio_strength"] = c.take(ratio, c.strength)
    frame["c1_lower_gwp"] = g_fv < g_e - EPS
    frame["c1_gain_fraction"] = (g_e - g_fv) / g_e
    frame["c2_ratio_higher_gwp"] = g_r > g_fv + EPS
    frame["c2_ratio_excess"] = g_r / g_fv - 1.0
    frame["c2_break_even_volume_reduction"] = np.where(frame.c2_ratio_higher_gwp, 1.0 - g_fv / g_r, 0.0)
    return frame


def frontier_check(c: rc.Candidates, fstar_values) -> dict:
    """Proposition 2: f_FV <= f_Q <= f_R and g_FV <= g_Q <= g_R on every case."""
    fv, ratio = rc.select_fixed_volume(c), rc.select_ratio(c)
    g_fv, g_r = c.take(fv, c.gwp), c.take(ratio, c.gwp)
    violations = 0
    dominated = 0
    for fstar in fstar_values:
        q = rc.select_capped(c, fstar)
        g_q, f_q = c.take(q, c.gwp), c.take(q, c.strength)
        violations += int(np.sum((g_q < g_fv - EPS) | (g_q > g_r + EPS)))
        # The capped choice must not be dominated by a candidate with strictly lower GWP at
        # no lower strength.  (Equal-GWP candidates above f* tie in the capped objective.)
        dom = c.valid & (c.strength >= f_q[:, None]) & (c.gwp < g_q[:, None] - EPS)
        dominated += int(dom.any(axis=1).sum())
        # Where FV and R agree on GWP, Q must agree as well.
        same = np.abs(g_fv - g_r) <= EPS
        violations += int(np.sum(same & (np.abs(g_q - g_fv) > EPS)))
    return {"fstar_values_checked": len(fstar_values), "cases": c.n,
            "between_or_agreement_violations": violations, "dominated_capped_choices": dominated}


def capped_comparison(c: rc.Candidates, frame: pd.DataFrame, fstar: float, area: np.ndarray) -> dict:
    """C3/C3' under one transition strength, with an explicit carbon audit.

    The reported C3/C3' metrics count selector displacement inside C2. This
    makes the limiting case f* above every retained strength exact: Q=R and
    C3' equals C2 under the same case denominator and tie-break. The separate
    *_carbon_suboptimal fields preserve the older quantity-carbon diagnostic
    without conflating it with selector disagreement.
    """
    fv, ratio, q = rc.select_fixed_volume(c), rc.select_ratio(c), rc.select_capped(c, fstar)
    carbon = c.gwp * area
    c_fv, c_r, c_q = c.take(fv, carbon), c.take(ratio, carbon), c.take(q, carbon)
    g_q, g_fv, g_r = c.take(q, c.gwp), c.take(fv, c.gwp), c.take(ratio, c.gwp)
    f_q, f_r = c.take(q, c.strength), c.take(ratio, c.strength)
    c2 = frame.c2_ratio_higher_gwp.to_numpy()
    # Selector identity is the decision-level quantity. Using indices rather
    # than only GWP values honours the documented tie-breaking rule.
    q_is_fv = q == fv
    q_is_r = q == ratio
    c3_carbon = c_r > c_q * (1 + EPS)
    c3p_carbon = c_fv > c_q * (1 + EPS)
    c3 = c2 & ~q_is_r
    c3p = c2 & ~q_is_fv
    out = {
        "fstar_psi": fstar,
        "cases": int(c.n),
        "c3_cases": int(c3.sum()),
        "c3_rate": float(c3.mean()),
        "c3_median_excess_affected": float(np.median(c_r[c3] / c_q[c3] - 1)) if c3.any() else 0.0,
        "c3_carbon_suboptimal_cases": int(c3_carbon.sum()),
        "c3_carbon_suboptimal_rate": float(c3_carbon.mean()),
        "c3p_cases": int(c3p.sum()),
        "c3p_rate": float(c3p.mean()),
        "c3p_median_excess_affected": float(np.median(c_fv[c3p] / c_q[c3p] - 1)) if c3p.any() else 0.0,
        "c3p_carbon_suboptimal_cases": int(c3p_carbon.sum()),
        "c3p_carbon_suboptimal_rate": float(c3p_carbon.mean()),
        "c2_cases": int(c2.sum()),
        "c2_resolved_to_fixed_volume": int((c2 & q_is_fv).sum()),
        "c2_resolved_to_ratio": int((c2 & q_is_r & ~q_is_fv).sum()),
        "c2_resolved_intermediate": int((c2 & ~q_is_fv & ~q_is_r).sum()),
        "non_c2_cases_changed_by_cap": int((~c2 & ~q_is_fv & ~q_is_r).sum()),
    }
    # Audit identities: C2 partitions into fixed-volume, intermediate and
    # ratio resolutions; C3 counts the first two and C3' the last two, so the
    # two overlap by the intermediate count and are not complements.
    assert out["c2_resolved_to_fixed_volume"] + out["c2_resolved_intermediate"] \
        + out["c2_resolved_to_ratio"] == out["c2_cases"]
    assert out["c3_cases"] == out["c2_resolved_to_fixed_volume"] + out["c2_resolved_intermediate"]
    assert out["c3p_cases"] == out["c2_resolved_intermediate"] + out["c2_resolved_to_ratio"]
    return out


def grid_area(c: rc.Candidates, width: float, pu: float) -> np.ndarray:
    k = rc.k_constant(pu, REF_STEEL_IN2)
    return np.maximum(k / c.strength, width ** 2 - REF_STEEL_IN2)


def main() -> None:
    c = rc.build_candidates()
    verification = rc.verify_against_reference(c)
    frame = base_comparisons(c)
    frame.to_csv(rc.OUT / "r1_decomposition_cases.csv", index=False)
    other = frame.company.ne(rc.NRM)
    summary = {
        "verification_against_reference": verification,
        "cases": int(len(frame)),
        "c1": {"denominator": int(len(frame)), "lower_gwp_cases": int(frame.c1_lower_gwp.sum()),
               "rate": float(frame.c1_lower_gwp.mean()),
               "median_gain_affected": float(frame.loc[frame.c1_lower_gwp, "c1_gain_fraction"].median()),
               "lower_gwp_cases_outside_nrm": int((frame.c1_lower_gwp & other).sum()),
               "rate_outside_nrm": float(frame.loc[other, "c1_lower_gwp"].mean())},
        "c2": {"denominator": int(len(frame)), "ratio_higher_gwp_cases": int(frame.c2_ratio_higher_gwp.sum()),
               "rate": float(frame.c2_ratio_higher_gwp.mean()),
               "median_excess_affected": float(frame.loc[frame.c2_ratio_higher_gwp, "c2_ratio_excess"].median()),
               "median_break_even_volume_reduction": float(
                   frame.loc[frame.c2_ratio_higher_gwp, "c2_break_even_volume_reduction"].median()),
               "cases_outside_nrm": int((frame.c2_ratio_higher_gwp & other).sum()),
               "rate_outside_nrm": float(frame.loc[other, "c2_ratio_higher_gwp"].mean()),
               "overlap_with_c1": int((frame.c2_ratio_higher_gwp & frame.c1_lower_gwp).sum())},
    }
    grid = []
    for width in GRID_WIDTHS_IN:
        for demand in GRID_DEMANDS:
            pu = PU_REF_KIP * demand
            fstar = float(rc.transition_strength(pu, width ** 2, REF_STEEL_IN2))
            row = capped_comparison(c, frame, fstar, grid_area(c, width, pu))
            grid.append({"minimum_width_in": width, "demand_ratio": demand, "demand_kip": pu, **row})
    grid = pd.DataFrame(grid)
    grid.to_csv(rc.OUT / "r1_decomposition_grid.csv", index=False)
    # Sweep of transition strengths: share of C2 cases resolved to each convention.
    sweep_values = np.arange(2000.0, 10001.0, 100.0)
    sweep = []
    for fstar in sweep_values:
        # Pure quantity-model area for a unit demand: max(f*/f, 1) up to a constant.
        area = np.maximum(fstar / c.strength, 1.0)
        sweep.append(capped_comparison(c, frame, float(fstar), area))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(rc.OUT / "r1_decomposition_fstar_sweep.csv", index=False)
    summary["proposition2_check"] = frontier_check(c, sweep_values)
    if summary["proposition2_check"]["between_or_agreement_violations"] or \
            summary["proposition2_check"]["dominated_capped_choices"]:
        raise AssertionError(summary["proposition2_check"])
    summary["grid_c3_rate_range"] = [float(grid.c3_rate.min()), float(grid.c3_rate.max())]
    summary["grid_c3p_rate_range"] = [float(grid.c3p_rate.min()), float(grid.c3p_rate.max())]
    (rc.OUT / "r1_decomposition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
