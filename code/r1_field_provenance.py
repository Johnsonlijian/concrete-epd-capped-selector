"""Strength-field definition, provenance checks and reinterpretation sensitivity (comment 1).

Documentary evidence (cited in the manuscript, not computed here):
  * PCR for Concrete (NSF 1112-19, 7.1.2): EPDs shall report the "specified
    compressive strength at specified age in days".
  * Original source (Knife River NRMCA EPD 10023): tables headed
    "Compressive Strength (Psi) at 28 days"; client-dependent rows exist.
  * Procurement practice indexes EPD GWP limits by specified strength f'c.

Data checks computed here on the analysis records:
  1. value pattern of the strength field (nominal classes vs measured means);
  2. concordance of the structured field with the explicit PSI token that the
     producer wrote into the mixture description;
  3. cross-declaration concordance for Knife River mixes whose mix identifiers
     appear both in the hash-locked 2018 NRMCA EPD and in the 2022-2023 ASTM
     declarations of the compilation (same plants);
  4. reinterpretation sensitivity: if the field were the required average
     strength f'cr of ACI 301 (no-record case), the design strength would be
     f'c = f - 1000 (f < 4000), 3000 (4000-4199), f - 1200 (4200-6200) or
     (f - 700)/1.1 (> 6200) psi.  C2, C3 and the design validation are re-run.
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

TOKEN = re.compile(r"(?<![\d.])(\d{1,2},?\d{3}(?:\.0+)?)\s*p\.?s\.?i\.?\b", flags=re.I)
EPS = 1e-9


def fc_from_required_average(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    out = np.where(f < 4000.0, f - 1000.0, np.nan)
    out = np.where((f >= 4000.0) & (f < 4200.0), 3000.0, out)
    out = np.where((f >= 4200.0) & (f <= 6200.0), f - 1200.0, out)
    return np.where(f > 6200.0, (f - 700.0) / 1.1, out)


def value_pattern(records: pd.DataFrame) -> dict:
    s = pd.to_numeric(records.strength_psi, errors="coerce")
    age28 = records.age_days.astype(float).eq(28)
    s28 = s[age28]
    return {"records_28d": int(age28.sum()),
            "share_multiple_of_500": float((s28 % 500 == 0).mean()),
            "share_multiple_of_250": float((s28 % 250 == 0).mean()),
            "share_multiple_of_100": float((s28 % 100 == 0).mean()),
            "distinct_values": int(s28.nunique()),
            "top_values": {str(int(k)): int(v) for k, v in s28.value_counts().head(10).items()}}


def token_concordance(records: pd.DataFrame) -> dict:
    age28 = records.age_days.astype(float).eq(28)
    d = records.loc[age28, ["description", "strength_psi", "company"]]
    counts = {"none": 0, "one_concordant": 0, "one_discordant": 0, "multiple": 0}
    discordant_examples = []
    for desc, f, company in d.itertuples(index=False):
        tokens = [float(m.replace(",", "")) for m in TOKEN.findall(str(desc))]
        if not tokens:
            counts["none"] += 1
        elif len(tokens) > 1:
            counts["multiple"] += 1
        elif abs(tokens[0] - float(f)) < 1e-6:
            counts["one_concordant"] += 1
        else:
            counts["one_discordant"] += 1
            if len(discordant_examples) < 8:
                discordant_examples.append({"company": company, "description": desc, "field_psi": float(f)})
    single = counts["one_concordant"] + counts["one_discordant"]
    return {**counts, "records_28d": int(len(d)),
            "concordance_among_single_token": counts["one_concordant"] / single if single else np.nan,
            "discordant_examples": discordant_examples}


def knife_river_cross_declaration(records_raw: pd.DataFrame, source_rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    comp = records_raw[records_raw.Company.str.contains("knife river", case=False, na=False)].copy()
    comp["plant_key"] = comp.Plant.str.replace(" Plant", "", regex=False).str.strip().str.lower()
    comp["mix_key"] = comp["Mixture Label"].str.replace(r"^\s*mix\s+", "", case=False, regex=True).str.strip().str.lower()
    comp = comp[comp.plant_key.isin({"coffee lake", "hillsboro", "linnton", "sundial"})]
    src = source_rows.copy()
    src["plant_key"] = src.plant.str.strip().str.lower()
    src["mix_key"] = src.mix_id.astype(str).str.strip().str.lower()
    merged = src.merge(comp, on=["plant_key", "mix_key"], how="inner", suffixes=("_2018", "_cmp"))
    merged["strength_cmp"] = pd.to_numeric(merged["Concrete Compressive Strength (psi)"], errors="coerce")
    merged["gwp_cmp"] = pd.to_numeric(merged["A1-A3 Global Warming Potential (kg CO2-eq)"], errors="coerce")
    merged["strength_2018"] = pd.to_numeric(merged.strength_psi, errors="coerce")
    merged["gwp_2018_m3"] = pd.to_numeric(merged.gwp_kgco2e_m3, errors="coerce")
    numeric = merged.dropna(subset=["strength_2018", "strength_cmp"])
    same = np.isclose(numeric.strength_2018, numeric.strength_cmp)
    checked = numeric.extraction_status.eq("SOURCE_CHECKED").to_numpy()
    ratio = (merged.gwp_cmp / merged.gwp_2018_m3).dropna()
    out = {"source_rows": int(len(src)), "compilation_rows_same_plants": int(len(comp)),
           "matched_pairs": int(len(merged)),
           "matched_pairs_numeric_strength_both": int(len(numeric)),
           "strength_identical_share": float(same.mean()) if len(numeric) else np.nan,
           "matched_source_checked_pairs": int(checked.sum()),
           "strength_identical_share_source_checked": float(same[checked].mean()) if checked.any() else np.nan,
           "strength_identical_share_quarantined": float(same[~checked].mean()) if (~checked).any() else np.nan,
           "gwp_ratio_compilation_to_2018_median": float(ratio.median()) if len(ratio) else np.nan,
           "gwp_ratio_q25_q75": [float(ratio.quantile(0.25)), float(ratio.quantile(0.75))] if len(ratio) else None,
           "issue_dates_compilation": sorted(comp["EPD Date of Issue"].unique().tolist()),
           "operators_compilation": sorted(comp["EPD Program Operator"].unique().tolist())}
    cols = ["plant_key", "mix_key", "description_2018" if "description_2018" in merged else "description",
            "strength_2018", "strength_cmp", "gwp_2018_m3", "gwp_cmp", "Mixture Description", "EPD Date of Issue"]
    cols = [c for c in cols if c in merged.columns]
    return out, merged[cols]


def reinterpretation(c: rc.Candidates) -> dict:
    fc = fc_from_required_average(c.strength)
    admissible = c.valid & (fc >= 2500.0 - EPS)          # ACI 318-19 19.2.1.1 minimum f'c
    cs = c.restrict(admissible)
    keep = (cs.valid.sum(axis=1) >= 2) & (np.nanmax(np.where(cs.valid, fc, -np.inf), axis=1)
                                          > np.nanmin(np.where(cs.valid, fc, np.inf), axis=1))
    cs = cs.subset(keep)
    fc_s = fc_from_required_average(cs.strength)
    fv, ratio = rc.select_fixed_volume(cs), rc.select_ratio(cs, fc_s)
    ratio_declared = rc.select_ratio(cs)
    g_fv, g_r = cs.take(fv, cs.gwp), cs.take(ratio, cs.gwp)
    out = {"cases_retained": int(cs.n), "cases_dropped_below_2500psi_or_single_strength": int((~keep).sum()),
           "c2_rate": float(np.mean(g_r > g_fv + EPS)),
           "ratio_choice_changed_by_reinterpretation": float(np.mean(ratio != ratio_declared))}
    rates = []
    for width in GRID_WIDTHS_IN:
        for demand in GRID_DEMANDS:
            pu = PU_REF_KIP * demand
            fstar = float(rc.transition_strength(pu, width ** 2, REF_STEEL_IN2))
            area = np.maximum(rc.k_constant(pu, REF_STEEL_IN2) / fc_s, width ** 2 - REF_STEEL_IN2)
            q = rc.select_capped(cs, fstar, fc_s)
            carbon = np.where(cs.valid, cs.gwp * area, np.inf)
            rates.append(float(np.mean(cs.take(ratio, carbon) > cs.take(q, carbon) * (1 + EPS))))
    out["c3_grid_range"] = [min(rates), max(rates)]
    # Design validation with every candidate (and each case target) at its reinterpreted f'c.
    cases_fc = cs.cases.assign(target_psi=np.maximum(fc_from_required_average(cs.cases.target_psi), 2500.0))
    reinterpreted = rc.Candidates(cases=cases_fc, row_id=cs.row_id, gwp=cs.gwp, strength=fc_s,
                                  valid=cs.valid, exact=cs.exact, extra=cs.extra)
    contexts = cd.evaluate(rc.pareto_order(reinterpreted), replace(cd.MAIN, label="fcr_reinterpretation"))
    contexts.to_csv(rc.OUT / "r1_fcr_design_contexts.csv", index=False)
    agg = cd.aggregate(contexts)
    agg.to_csv(rc.OUT / "r1_fcr_design_summary.csv", index=False)
    tot = agg[agg.floor_in.eq("all")]
    out["design_agreement"] = {f"{p}|{s}": float(v) for p, s, v in
                               zip(tot.policy, tot.selector, tot.agreement)}
    return out


def main() -> None:
    records = rc.load_records()
    raw = pd.read_csv(rc.RAW, low_memory=False)
    src_dir = next(p for p in (rc.ROOT / "sources" / "agent_nrmca", rc.ROOT.parent / "sources" / "agent_nrmca")
                   if p.exists())
    source_rows = pd.read_csv(src_dir / "knife_river_all_rows.csv", low_memory=False)
    summary = {"value_pattern": value_pattern(records), "token_concordance": token_concordance(records)}
    kr, pairs = knife_river_cross_declaration(raw, source_rows)
    pairs.to_csv(rc.OUT / "r1_knife_river_cross_declaration_pairs.csv", index=False)
    summary["knife_river_cross_declaration"] = kr
    c = rc.build_candidates(records)
    rc.verify_against_submission(c)
    summary["primary_candidates_value_pattern"] = {
        "share_multiple_of_500": float(np.mean(c.strength[c.valid] % 500 == 0)),
        "distinct_values": int(len(np.unique(c.strength[c.valid])))}
    summary["fcr_reinterpretation"] = reinterpretation(c)
    (rc.OUT / "r1_field_provenance_summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                                              encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
