"""Additional checks requested by the internal audit of R1.

1. Paired supplier-cluster bootstrap of the differences in design agreement between
   selectors.  Selectors are scored on the same cases, so the difference is the
   quantity to test; intervals resample the 28 supplier labels.
2. Column-admissible subset: cases whose declarations label the mixtures only for
   non-structural uses (paving, hardscape, fill, grout, mass concrete, shotcrete,
   slab on grade) are removed before the design test.
3. Window sensitivity: the candidate window of 1000 psi is replaced by 500 and
   2000 psi (the submitted case sets for those windows) and C1, C2 and the design
   test are repeated.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

import r1_column_design as cd
import r1_core as rc

SEED = 20261013
EPS = 1e-9
NON_STRUCTURAL = ["Application Category: Hardscape", "Application Category: Paving", "Application Category: Filler",
                  "App: Sidewalk, Curb, and Hardscape?", "App: Paving?", "App: Mass Concrete?", "App: Flowable Fill?",
                  "App: Shotcrete?", "App: Cement and Masonry Grout?", "App: Slab on Grade?"]
STRUCTURAL = ["Application Category: Structural", "App: Structural (General)?", "App: Column?",
              "App: Foundation, Footing?", "App: Wall?", "App: Elevated Horizontal?", "App: Bridge Deck and Pier?"]


def paired_bootstrap(n_boot: int = 2000) -> pd.DataFrame:
    shares = pd.read_csv(rc.OUT / "r1_design_case_agreement.csv")
    rng = np.random.default_rng(SEED)
    groups = {k: v.index.to_numpy() for k, v in shares.groupby("company")}
    companies = np.array(sorted(groups))
    rows = []
    for policy in cd.POLICIES:
        overall = {}
        for sel in cd.SELECTORS:
            cols = [c for c in shares.columns if c.startswith(f"{policy}|") and c.endswith(f"|{sel}")]
            overall[sel] = shares[cols].mean(axis=1).to_numpy()
        pairs = [("capped", "ratio"), ("capped_effective", "ratio"), ("capped", "fixed_volume"),
                 ("capped_effective", "capped"), ("ratio", "fixed_volume")]
        draws = {p: [] for p in pairs}
        for _ in range(n_boot):
            idx = np.concatenate([groups[k] for k in rng.choice(companies, size=len(companies), replace=True)])
            for a, b in pairs:
                draws[(a, b)].append(float(np.mean(overall[a][idx] - overall[b][idx])))
        for a, b in pairs:
            d = np.array(draws[(a, b)])
            rows.append({"policy": policy, "comparison": f"{a} - {b}",
                         "difference": float(np.mean(overall[a] - overall[b])),
                         "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)),
                         "share_cases_a_ge_b": float(np.mean(overall[a] >= overall[b] - 1e-12))})
    return pd.DataFrame(rows)


def admissible_mask(cases: pd.DataFrame) -> np.ndarray:
    raw = pd.read_csv(rc.RAW, low_memory=False, usecols=NON_STRUCTURAL + STRUCTURAL)
    raw.insert(0, "row_id", np.arange(len(raw)) + 2)
    labels = raw.set_index("row_id").loc[cases.baseline_id]
    flag = lambda cols: labels[cols].astype(str).apply(lambda s: s.str.lower().eq("true")).any(axis=1).to_numpy()
    return ~(flag(NON_STRUCTURAL) & ~flag(STRUCTURAL))


def c1_c2(c: rc.Candidates) -> dict:
    fv_e, fv, ratio = rc.select_fixed_volume(c, c.exact), rc.select_fixed_volume(c), rc.select_ratio(c)
    g_e, g_fv, g_r = c.take(fv_e, c.gwp), c.take(fv, c.gwp), c.take(ratio, c.gwp)
    return {"cases": int(c.n), "c1_rate": float(np.mean(g_fv < g_e - EPS)), "c2_rate": float(np.mean(g_r > g_fv + EPS))}


def design_summary(c: rc.Candidates, label: str) -> pd.DataFrame:
    contexts = cd.evaluate(rc.pareto_order(c), replace(cd.MAIN, label=label))
    agg = cd.aggregate(contexts)
    return agg[agg.floor_in.astype(str).eq("all")]


def main() -> None:
    boot = paired_bootstrap()
    boot.to_csv(rc.OUT / "r1_bootstrap_selector_differences.csv", index=False)
    records = rc.load_records()
    cases = rc.load_cases()
    primary = rc.build_candidates(records, cases)
    rc.verify_against_submission(primary)
    keep = admissible_mask(cases)
    subset_rows, design_rows = [], []
    adm = primary.subset(keep)
    subset_rows.append({"subset": "column_admissible", **c1_c2(adm), "removed_cases": int((~keep).sum())})
    design_rows.append(design_summary(adm, "column_admissible"))
    for window in (500, 2000):
        wcases = pd.read_csv(rc.DATA / f"choices_description_series_{window}.csv", low_memory=False)
        cw = rc.build_candidates(records, wcases, window=float(window))
        # The submitted window tables are reproduced exactly by the vectorised selectors.
        assert (cw.take(rc.select_fixed_volume(cw, cw.exact), cw.row_id) == wcases.baseline_id.to_numpy()).all()
        assert (cw.take(rc.select_fixed_volume(cw), cw.row_id) == wcases.candidate_id.to_numpy()).all()
        subset_rows.append({"subset": f"window_{window}psi", **c1_c2(cw), "removed_cases": None})
        design_rows.append(design_summary(cw, f"window_{window}psi"))
    pd.DataFrame(subset_rows).to_csv(rc.OUT / "r1_extended_subsets.csv", index=False)
    pd.concat(design_rows, ignore_index=True).to_csv(rc.OUT / "r1_extended_design_summary.csv", index=False)
    print(boot.to_string(index=False))
    print(pd.DataFrame(subset_rows).to_string(index=False))
    print(pd.concat(design_rows, ignore_index=True)[["label", "policy", "selector", "agreement", "mean_regret"]].to_string(index=False))


if __name__ == "__main__":
    main()
