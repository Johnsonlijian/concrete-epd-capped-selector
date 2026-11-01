"""Robustness of the complete selection process.

The submitted 5% and 10% bounds tested whether one fixed pair kept its order.
Here every declared GWP is perturbed and the whole selection process is re-run:
candidate choice, all three selectors, the C1/C2/C3 statistics and the
design-based optimum of the code-designed columns (minimum-size policy, whose
member quantities do not depend on GWP).

Noise models (declared values are treated as the centre; these are stress
scenarios, not estimated EPD uncertainty distributions):
  independent  g' = g exp(s z - s^2/2), one z per record, s = 2.5, 5 or 10 %
  structured   A1, A2 and A3 of every record in one declaration (plant-issue
               menu) share common factors (10 %, 20 %, 20 %), plus 2.5 %
               independent record noise.  Shared background-data error moves
               all mixes of one declaration together.
A supplier-cluster bootstrap (resampling the 28 supplier labels) gives
intervals that respect the concentration of cases in one catalogue.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import r1_core as rc
import r1_column_design as cd
from r1_decomposition import GRID_DEMANDS, GRID_WIDTHS_IN, PU_REF_KIP, REF_STEEL_IN2

SEED = 20261013
EPS = 1e-9
N_REP = 1000
N_REP_DESIGN = 200
DESIGN_TRIBUTARY = 625.0  # one bay size keeps the per-replicate design scoring tractable


def statistics(c: rc.Candidates, g: np.ndarray, fstars: dict) -> dict:
    """C1, C2 and C3 rates and the selector choices for one GWP realisation."""
    valid = c.valid
    gi = np.where(valid, g, np.inf)
    fv_e = np.argmin(np.where(c.exact, gi, np.inf), axis=1)
    fv = np.argmin(gi, axis=1)
    ratio = rc.select(gi / np.where(valid, c.strength, 1.0), valid, **rc.RATIO_TOL)
    g_e, g_fv, g_r = c.take(fv_e, g), c.take(fv, g), c.take(ratio, g)
    c2_mask = g_r > g_fv + EPS
    out = {"picks": {"fixed_volume": fv, "ratio": ratio},
           "c1": float(np.mean(g_fv < g_e - EPS)), "c2": float(np.mean(c2_mask))}
    for name, (fstar, area) in fstars.items():
        q = rc.select(gi / np.minimum(np.where(valid, c.strength, 1.0), fstar), valid, **rc.QUANTITY_TOL)
        carbon = np.where(valid, g * area, np.inf)
        q_is_ratio = q == ratio
        out[f"c3_{name}"] = float(np.mean(c2_mask & ~q_is_ratio))
        out[f"c3carbon_{name}"] = float(np.mean(c.take(ratio, carbon) > c.take(q, carbon) * (1 + EPS)))
        out["picks"][f"capped_{name}"] = q
    return out


def perturbation(c: rc.Candidates, rng: np.random.Generator, model: str, s: float,
                 uniq: np.ndarray, inverse: np.ndarray, menu_codes: np.ndarray) -> np.ndarray:
    """One perturbed GWP array, consistent for a record that appears in several cases."""
    z = rng.standard_normal(len(uniq))
    if model == "independent":
        factor = np.exp(s * z - s * s / 2)[inverse]
        return np.where(c.valid, c.gwp * factor.reshape(c.gwp.shape), np.nan)
    a1, a2, a3 = (c.extra[k] for k in ("a1", "a2", "a3"))
    m = menu_codes.max() + 1
    f1 = np.exp(0.10 * rng.standard_normal(m) - 0.005)[menu_codes][:, None]
    f2 = np.exp(0.20 * rng.standard_normal(m) - 0.02)[menu_codes][:, None]
    f3 = np.exp(0.20 * rng.standard_normal(m) - 0.02)[menu_codes][:, None]
    ind = np.exp(0.025 * z - 0.025 ** 2 / 2)[inverse].reshape(c.gwp.shape)
    modules = a1 * f1 + a2 * f2 + a3 * f3
    scale = c.gwp / np.where(a1 + a2 + a3 > 0, a1 + a2 + a3, np.nan)
    return np.where(c.valid, modules * scale * ind, np.nan)


def supplier_cluster_bootstrap(c: rc.Candidates, fstars: dict) -> pd.DataFrame:
    """Supplier-cluster bootstrap of the unperturbed case-level statistics."""
    agreement_cases = pd.read_csv(rc.OUT / "r1_design_case_agreement.csv")
    dec = pd.read_csv(rc.OUT / "r1_decomposition_cases.csv")
    frame = dec[["company", "c1_lower_gwp", "c2_ratio_higher_gwp"]].copy()
    for col in agreement_cases.columns:
        if col.startswith("min_carbon|"):
            frame[col] = agreement_cases[col].to_numpy()
    st = statistics(c, c.gwp, fstars)
    ratio = st["picks"]["ratio"]
    c2_mask = c.take(ratio, c.gwp) > c.take(st["picks"]["fixed_volume"], c.gwp) + EPS
    for name in fstars:
        frame[f"c3_{name}"] = c2_mask & (st["picks"][f"capped_{name}"] != ratio)
    companies = frame.company.unique()
    groups = {k: v.index.to_numpy() for k, v in frame.groupby("company")}
    rng = np.random.default_rng(SEED + 2)
    cols = [x for x in frame.columns if x != "company"]
    boot = []
    for _ in range(2000):
        draw = rng.choice(companies, size=len(companies), replace=True)
        idx = np.concatenate([groups[k] for k in draw])
        boot.append(frame.loc[idx, cols].astype(float).mean())
    boot = pd.DataFrame(boot)
    point = frame[cols].astype(float).mean()
    ci = pd.DataFrame({"point": point, "lo": boot.quantile(0.025), "hi": boot.quantile(0.975)})
    ci.to_csv(rc.OUT / "r1_bootstrap_supplier_clusters.csv")
    return ci


def main() -> None:
    records = rc.load_records()
    c = rc.pareto_order(rc.build_candidates(records))
    modules = records[["row_id"]].assign(**{k: pd.to_numeric(records[k], errors="coerce") for k in ("a1", "a2", "a3")})
    rc.attach_fields(c, modules, ["a1", "a2", "a3"])
    for k in ("a1", "a2", "a3"):
        c.extra[k] = c.extra[k].astype(float)
    if np.isnan(np.where(c.valid, c.extra["a1"] + c.extra["a2"] + c.extra["a3"], 0.0)).any():
        raise ValueError("A1-A3 modules missing for a candidate")
    uniq, inverse = np.unique(np.where(c.valid, c.row_id, -1).ravel(), return_inverse=True)
    menu_codes = pd.factorize(c.cases.plant_issue_id)[0]
    fstars = {}
    for width in GRID_WIDTHS_IN:
        for demand in GRID_DEMANDS:
            pu = PU_REF_KIP * demand
            fstar = float(rc.transition_strength(pu, width ** 2, REF_STEEL_IN2))
            fstars[f"{width:g}in_{demand:g}"] = (fstar, np.maximum(rc.k_constant(pu, REF_STEEL_IN2) / c.strength,
                                                                    width ** 2 - REF_STEEL_IN2))
    if "--bootstrap-only" in sys.argv[1:]:
        print(supplier_cluster_bootstrap(c, fstars).to_string())
        return
    base = statistics(c, c.gwp, fstars)
    rng = np.random.default_rng(SEED)
    rows = []
    scenarios = [("independent", 0.025), ("independent", 0.05), ("independent", 0.10), ("structured", 0.0)]
    for model, s in scenarios:
        for rep in range(N_REP):
            g = perturbation(c, rng, model, s, uniq, inverse, menu_codes)
            st = statistics(c, g, fstars)
            row = {"model": model, "sigma": s, "replicate": rep, "c1": st["c1"], "c2": st["c2"]}
            row.update({k: v for k, v in st.items() if k.startswith("c3_")})
            for name, pick in st["picks"].items():
                row[f"stable_{name}"] = float(np.mean(pick == base["picks"][name]))
            rows.append(row)
    mc = pd.DataFrame(rows)
    mc.to_csv(rc.OUT / "r1_robustness_montecarlo.csv", index=False)
    summary = mc.drop(columns="replicate").groupby(["model", "sigma"]).quantile([0.025, 0.5, 0.975])
    summary.to_csv(rc.OUT / "r1_robustness_montecarlo_summary.csv")

    # Design-based optimum under perturbed GWP (minimum-size policy: quantities fixed).
    dem, _, res = cd.member_quantities(c, c.strength)
    keep = dem.tributary_ft2.eq(DESIGN_TRIBUTARY).to_numpy()
    dem_k = dem.loc[keep].reset_index(drop=True)
    target = c.cases.target_psi.to_numpy(dtype=float)
    design_rows = []
    rng = np.random.default_rng(SEED + 1)
    for model, s in scenarios:
        for rep in range(N_REP_DESIGN + 1):
            g = c.gwp if rep == 0 else perturbation(c, rng, model, s, uniq, inverse, menu_codes)
            st = statistics(c, g, {})
            for floor in cd.FLOORS_IN:
                conc, steel, _ = res[("min_size", floor)]
                conc, steel = conc[:, :, keep], steel[:, :, keep]
                # Selector choices are re-made on the perturbed values (whole-process re-run).
                picks = {"fixed_volume": st["picks"]["fixed_volume"], "ratio": st["picks"]["ratio"]}
                agree = {k: 0.0 for k in cd.SELECTORS}
                regret = {k: 0.0 for k in cd.SELECTORS}
                for d, dm in dem_k.iterrows():
                    carbon = g * conc[:, :, d] + cd.REBAR_KG_PER_M3 * steel[:, :, d]
                    carbon = np.where(c.valid, carbon, np.inf)
                    best = rc.select_objective(c, carbon)
                    c_best = c.take(best, carbon)
                    amin = floor ** 2
                    fstar = float(rc.transition_strength(dm.pu_kip, amin, cd.RHO_MIN * amin, floor_governed_ok=True))
                    b_eff = cd.governing_minimum(floor, dm.pu_kip, dm.beta_dns, target, cd.MAIN)
                    fstar_eff = rc.transition_strength(dm.pu_kip, b_eff ** 2, cd.RHO_MIN * b_eff ** 2,
                                                       floor_governed_ok=True)
                    gi = np.where(c.valid, g, np.inf)
                    f = np.where(c.valid, c.strength, 1.0)
                    cur = dict(picks)
                    cur["capped"] = rc.select(gi / np.minimum(f, fstar), c.valid, **rc.QUANTITY_TOL)
                    cur["capped_effective"] = rc.select(gi / np.minimum(f, fstar_eff[:, None]), c.valid,
                                                        **rc.QUANTITY_TOL)
                    for k in cd.SELECTORS:
                        r = c.take(cur[k], carbon) / c_best - 1.0
                        agree[k] += float(np.mean(r <= 1e-9)) / len(dem_k)
                        regret[k] += float(np.mean(r)) / len(dem_k)
                for k in cd.SELECTORS:
                    design_rows.append({"model": model, "sigma": s, "replicate": rep, "floor_in": floor,
                                        "selector": k, "agreement": agree[k], "mean_regret": regret[k]})
    design = pd.DataFrame(design_rows)
    design.to_csv(rc.OUT / "r1_robustness_design.csv", index=False)
    pert = design[design.replicate > 0]
    dsum = pert.groupby(["model", "sigma", "floor_in", "selector"]).agreement.quantile([0.025, 0.5, 0.975]).unstack()
    dsum.to_csv(rc.OUT / "r1_robustness_design_summary.csv")
    # Probability that the capped selector beats the ratio (and fixed volume) in agreement.
    wide = pert.pivot_table(index=["model", "sigma", "replicate", "floor_in"], columns="selector", values="agreement")
    dominance = wide.assign(capped_beats_ratio=wide.capped > wide.ratio,
                            capped_beats_fixed=wide.capped > wide.fixed_volume,
                            effective_beats_ratio=wide.capped_effective > wide.ratio).groupby(
        ["model", "sigma", "floor_in"])[["capped_beats_ratio", "capped_beats_fixed", "effective_beats_ratio"]].mean()
    dominance.to_csv(rc.OUT / "r1_robustness_design_dominance.csv")

    ci = supplier_cluster_bootstrap(c, fstars)
    out = {"seed": SEED, "replicates": N_REP, "design_replicates": N_REP_DESIGN,
           "unperturbed": {k: v for k, v in base.items() if k != "picks"}}
    (rc.OUT / "r1_robustness_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(summary.to_string())
    print(dsum.to_string())
    print(dominance.to_string())
    print(ci.to_string())


if __name__ == "__main__":
    main()
