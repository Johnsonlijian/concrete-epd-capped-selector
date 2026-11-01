"""Stricter comparability subset and comparability ladder.

The PCR for Concrete (NSF 1112-19, Section 5.5) allows A1-A3 comparisons only if
the same secondary data sets are used, the functional units are the same and the
mixes have the same or superior structural, thermal and exposure properties.
The compilation records the secondary data sets of every mix, so the condition
can be tested record by record.  A candidate is retained in the STRICT set only
if, relative to the baseline (exact-target, fixed-volume) record of its case:

  S1  no conflicting secondary data set: for every 'Primary LCI Data Source'
      field, values are equal whenever both records report one;
  S2  identical constituent classes: the same 'Primary LCI Data Source' fields
      are populated (e.g. the same admixture types, SCMs, aggregates, transport);
  S3  identical reported A3 (plant-gate manufacturing) GWP, used only as an
      observable compatibility proxy. Equality of an impact value does not
      establish identical allocation of plant operations per cubic metre;
  S4  identical application labels, specification fields (w/c, fly ash, slag)
      and validity end date.

Higher strength within the 1000 psi window satisfies the "same or superior
structural property" condition by construction.  Cases left without a
higher-strength candidate are dropped.  The comparability ladder reports the
same statistics for the plant-issue, component, description-series (primary)
and strict scopes with every baseline held fixed.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pandas as pd

import r1_core as rc
import r1_column_design as cd
from r1_decomposition import GRID_DEMANDS, GRID_WIDTHS_IN, PU_REF_KIP, REF_STEEL_IN2

EPS = 1e-9
DATASET_PREFIX = "Primary LCI Data Source"
APP_COLUMNS_PREFIX = ("Labeled by", "Primary Application", "Other Application", "Application Category",
                      "App:", "Amount of Different")
SPEC_COLUMNS = ["W/C Specified?", "Percent Fly Ash Specified?", "Percent Slag Specified?"]


def _norm(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9.%]+", "", str(value).lower())


def raw_fields() -> pd.DataFrame:
    raw = pd.read_csv(rc.RAW, low_memory=False)
    raw.insert(0, "row_id", np.arange(len(raw)) + 2)
    datasets = [c for c in raw.columns if c.startswith(DATASET_PREFIX)]
    apps = [c for c in raw.columns if c.startswith(APP_COLUMNS_PREFIX)]
    norm = raw[datasets].map(_norm)
    out = pd.DataFrame({"row_id": raw.row_id})
    out["datasets"] = [tuple(r) for r in norm.to_numpy()]
    out["presence"] = ["".join("1" if v else "0" for v in r) for r in norm.to_numpy()]
    out["a3"] = pd.to_numeric(raw["A3 GWP"], errors="coerce")
    out["labels"] = raw[apps + SPEC_COLUMNS + ["EPD Valid Until Date"]].astype(str).agg("|".join, axis=1)
    return out, datasets


def strict_mask(c: rc.Candidates) -> tuple[np.ndarray, dict]:
    base = rc.select_fixed_volume(c, c.exact)
    ds, pres, a3, lab = (c.extra[k] for k in ("datasets", "presence", "a3", "labels"))
    n, w = c.row_id.shape
    s1 = np.ones((n, w), bool)
    s2, s3, s4 = (np.ones((n, w), bool) for _ in range(3))
    for i in range(n):
        b = base[i]
        for j in range(w):
            if not c.valid[i, j]:
                continue
            x, y = ds[i, j], ds[i, b]
            s1[i, j] = all((not u) or (not v) or u == v for u, v in zip(x, y))
            s2[i, j] = pres[i, j] == pres[i, b]
            s3[i, j] = np.isclose(a3[i, j], a3[i, b]) if np.isfinite(a3[i, j]) and np.isfinite(a3[i, b]) else False
            s4[i, j] = lab[i, j] == lab[i, b]
    keep = c.valid & s1 & s2 & s3 & s4
    parts = {name: float((c.valid & m).sum() / c.valid.sum()) for name, m in
             (("S1_datasets", s1), ("S2_constituents", s2), ("S3_A3", s3), ("S4_labels", s4), ("all", keep))}
    return keep, parts


def statistics(c: rc.Candidates, baseline_gwp: np.ndarray, label: str) -> dict:
    fv, ratio = rc.select_fixed_volume(c), rc.select_ratio(c)
    g_fv, g_r = c.take(fv, c.gwp), c.take(ratio, c.gwp)
    other = c.cases.company.ne(rc.NRM).to_numpy()
    c1 = g_fv < baseline_gwp - EPS
    c2 = g_r > g_fv + EPS
    out = {"scope": label, "cases": int(c.n), "suppliers": int(c.cases.company.nunique()),
           "candidates_per_case_median": float(np.median(c.valid.sum(axis=1))),
           "c1_rate": float(c1.mean()), "c1_cases": int(c1.sum()), "c1_rate_outside_nrm": float(c1[other].mean()),
           "c2_rate": float(c2.mean()), "c2_cases": int(c2.sum()), "c2_rate_outside_nrm": float(c2[other].mean()),
           "c2_median_excess": float(np.median(g_r[c2] / g_fv[c2] - 1)) if c2.any() else np.nan}
    rates = []
    for width in GRID_WIDTHS_IN:
        for demand in GRID_DEMANDS:
            pu = PU_REF_KIP * demand
            fstar = float(rc.transition_strength(pu, width ** 2, REF_STEEL_IN2))
            q = rc.select_capped(c, fstar)
            # Headline C3 is a selector-displacement count inside C2, not a
            # carbon-suboptimality diagnostic.  Use the same definition as
            # r1_decomposition.capped_comparison so that the ladder and the
            # analyst-grid table report the same statistic.
            rate = float(np.mean(c2 & (q != ratio)))
            out[f"c3_{width:g}in_{demand:g}"] = rate
            rates.append(rate)
    out["c3_grid_min"], out["c3_grid_max"] = float(min(rates)), float(max(rates))
    return out


def main() -> None:
    records = rc.load_records()
    cases = rc.load_cases()
    fields, datasets = raw_fields()
    primary = rc.build_candidates(records, cases)
    rc.verify_against_submission(primary)
    rc.attach_fields(primary, fields, ["datasets", "presence", "a3", "labels"])
    baseline_gwp = cases.baseline_gwp.to_numpy(dtype=float)
    ladder = []
    for scope in ("plant", "component"):
        cs = rc.build_candidates(records, cases, scope=scope)
        ladder.append(statistics(cs, baseline_gwp, f"{scope}_scope"))
    ladder.append(statistics(primary, baseline_gwp, "description_series_primary"))
    keep, parts = strict_mask(primary)
    strict = primary.restrict(keep)
    # A case remains only if a higher-strength candidate survives next to the baseline.
    has_higher = (strict.valid & (strict.strength > cases.target_psi.to_numpy()[:, None] + EPS)).any(axis=1)
    strict = strict.subset(has_higher)
    ladder.append(statistics(strict, baseline_gwp[has_higher], "strict_pcr_5_5"))
    current = strict.cases.index[pd.to_datetime(
        records.set_index("row_id").loc[strict.cases.baseline_id, "expires"].to_numpy(),
        format="%m/%d/%Y", errors="coerce") >= pd.Timestamp("2026-10-13")]
    strict_current = strict.subset(np.isin(np.arange(strict.n), current))
    ladder.append(statistics(strict_current, baseline_gwp[has_higher][current], "strict_pcr_5_5_current"))
    ladder = pd.DataFrame(ladder)
    ladder.to_csv(rc.OUT / "r1_comparability_ladder.csv", index=False)
    # Design-based validation on the strict subset (main configuration).
    contexts = cd.evaluate(rc.pareto_order(strict), replace(cd.MAIN, label="strict_pcr_5_5"))
    contexts.to_csv(rc.OUT / "r1_strict_design_contexts.csv", index=False)
    agg = cd.aggregate(contexts)
    agg.to_csv(rc.OUT / "r1_strict_design_summary.csv", index=False)
    summary = {"candidate_share_passing": parts, "primary_cases": int(primary.n),
               "strict_cases": int(strict.n), "strict_current_cases": int(strict_current.n),
               "strict_suppliers": int(strict.cases.company.nunique()),
               "strict_nrm_cases": int(strict.cases.company.eq(rc.NRM).sum()),
               "dataset_fields_checked": len(datasets)}
    (rc.OUT / "r1_strict_subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(ladder.T.to_string())
    print(agg[agg.floor_in.eq("all")].to_string(index=False))


if __name__ == "__main__":
    main()
