"""Reanalyse reported EPD options; never infer engineering approval from metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd

from reconcile_aliases import SOURCE_CLUSTER_MAP, reconcile_aliases, write_audits


ROOT = Path(__file__).resolve().parents[1]
RAW_NAME = "Compiled_Concrete_EPD_Data_Version_4c_Final_Published.csv"
# The revision workspace reads the checksum-locked input from the parent
# package instead of duplicating it; the hash check below is unchanged.
RAW = next((p for p in (ROOT / "raw" / RAW_NAME, ROOT.parent / "raw" / RAW_NAME) if p.exists()),
           ROOT / "raw" / RAW_NAME)
OUT = ROOT / "derived"
EXPECTED_HASH = "6b6eaff116aac5c849a6f1031bdd0787ac9c545c728707de81d755e0654ac0af"
TARGETS = [2500, 3000, 3500, 4000, 4500, 5000, 6000]
MENU_KEYS = ["company", "plant", "address", "issued", "operator", "source"]
FAMILIES = {
    "plant_issue": [],
    "component_set": ["components"],
    "description_series": ["components", "description_series"],
}


def norm(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def signature(description: str, strength: float) -> str:
    """Replace one explicit, concordant PSI value; leave every other token intact."""
    pattern = r"(?<![\d.])(\d{1,2},?\d{3}(?:\.0+)?)\s*p\.?s\.?i\.?\b"
    matches = list(re.finditer(pattern, description, flags=re.I))
    if len(matches) != 1:
        return ""
    match = matches[0]
    if abs(float(match[1].replace(",", "")) - strength) > 1e-6:
        return ""
    return norm(description[: match.start()] + "<strength> psi " + description[match.end() :])


def prepare() -> tuple[pd.DataFrame, dict]:
    actual = hashlib.file_digest(RAW.open("rb"), "sha256").hexdigest()
    if actual != EXPECTED_HASH:
        raise ValueError(f"Raw file checksum mismatch: {actual}")
    raw = pd.read_csv(RAW, low_memory=False)
    mapping = {
        "Company": "company", "Plant": "plant", "Full Address": "address",
        "EPD Date of Issue": "issued", "EPD Valid Until Date": "expires",
        "EPD Program Operator": "operator", "EPD Source Link": "source",
        "Mixture Label": "mix", "Mixture Description": "description",
        "Concrete Compressive Strength (psi)": "strength_psi",
        "Concrete Curation Time": "age_raw", "Declared Unit": "unit",
        "Product Components": "components_raw",
        "A1-A3 Global Warming Potential (kg CO2-eq)": "gwp",
        "A1 GWP": "a1", "A2 GWP": "a2", "A3 GWP": "a3",
        "W/C Specified?": "wc_raw", "Plant Location - State": "state",
    }
    d = raw[list(mapping)].rename(columns=mapping).copy()
    d.insert(0, "row_id", np.arange(len(d)) + 2)  # Data-row position + header; not physical text-line number.
    for col in [*MENU_KEYS, "mix", "description", "components_raw", "unit"]:
        d[col] = d[col].map(norm)
    for col in ["strength_psi", "gwp", "a1", "a2", "a3"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["age_days"] = pd.to_numeric(d.age_raw.str.extract(r"^PSI at (\d+) days$", expand=False))
    d["components"] = d.components_raw.map(lambda s: " | ".join(sorted(set(x.strip() for x in s.split(",")))))
    d["description_series"] = [signature(s, f) for s, f in zip(d.description, d.strength_psi)]
    d["wc"] = pd.to_numeric(d.wc_raw, errors="coerce")
    d["expires_parsed"] = pd.to_datetime(d.expires, format="%m/%d/%Y", errors="coerce")
    d["current_at_analysis"] = d.expires_parsed.ge(pd.Timestamp("2026-09-09"))
    exclusion_text = d.description + " " + d.components_raw
    d["specialty_flag"] = exclusion_text.str.contains(
        r"light\s*weight|lightweight|\bltwt\b|\blw\b|shotcrete|\bshot\b|flowable|\bclsm\b|\bgrout\b", regex=True
    )
    valid = (
        d.unit.eq("1 m3 of concrete") & d.strength_psi.between(2500, 8000)
        & d.gwp.gt(0) & np.isfinite(d.gwp) & d.age_days.ge(28)
        & ~d.specialty_flag & d[MENU_KEYS].ne("").all(axis=1)
    )
    # Ambiguous same-label records with conflicting declared values are quarantined.
    id_keys = MENU_KEYS + ["mix", "age_days"]
    counts = d.groupby(id_keys, dropna=False)[["strength_psi", "gwp"]].transform("nunique")
    d["identity_conflict"] = counts.gt(1).any(axis=1)
    valid &= ~d.identity_conflict
    profile = {
        "raw_rows": len(raw), "raw_columns": len(raw.columns),
        "raw_full_duplicate_rows": int(raw.duplicated().sum()),
        "companies_raw": int(d.company.nunique()),
        "physical_plant_keys_raw": int(d[["company", "plant", "address"]].drop_duplicates().shape[0]),
        "unit_counts": raw["Declared Unit"].value_counts().to_dict(),
        "age_counts": d.age_days.value_counts().sort_index().to_dict(),
        "identity_conflict_rows_raw": int(d.identity_conflict.sum()),
        "missing_wc_rows_raw": int(d.wc.isna().sum()),
        "filter_rule": "2500-8000 psi; declared age >=28; 1m3; positive finite A1-A3 GWP; nonempty menu identifiers; no flagged specialty or same-label conflicts",
        "eligible_before_dedup": int(valid.sum()), "sha256": actual,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    d.to_csv(OUT / "record_screening.csv", index=False)
    keep = d.loc[valid].drop_duplicates(id_keys + ["strength_psi", "gwp", "description", "components"])
    profile["within_label_rows"] = len(keep)
    profile["within_label_duplicates_removed"] = int(valid.sum() - len(keep))
    # Preserve the exact pre-alias frame used by the reconciliation stage so
    # regression tests exercise screening and alias handling independently.
    keep.to_csv(OUT / "alias_input_records.csv", index=False)
    keep, alias_metadata = reconcile_aliases(keep)
    write_audits(alias_metadata, OUT)
    profile["alias_reconciliation"] = alias_metadata["summary"]
    # Clusters link known source labels; this is not a legal-ownership registry.
    keep["company"] = keep.company.replace(SOURCE_CLUSTER_MAP)
    profile["analysis_rows"] = len(keep)
    profile["analysis_28_day_rows"] = int(keep.age_days.eq(28).sum())
    profile["analysis_companies"] = int(keep.company.nunique())
    profile["explicit_psi_series_rows_28d"] = int((keep.age_days.eq(28) & keep.description_series.ne("")).sum())
    profile["deduplicated_rows_removed"] = profile["within_label_duplicates_removed"]
    profile["alias_policy_rows_removed"] = profile["within_label_rows"] - len(keep)
    profile["analysis_plant_issue_keys"] = int(keep[MENU_KEYS].drop_duplicates().shape[0])
    keep.to_csv(OUT / "analysis_records.csv", index=False)
    return keep, profile


def best(frame: pd.DataFrame) -> pd.Series:
    # Stable row order is a provenance convention, not a preference among equal GWPs.
    return frame.sort_values(["gwp", "row_id"], kind="stable").iloc[0]


def choices(d: pd.DataFrame, family: str, window: int = 1000) -> pd.DataFrame:
    extra = FAMILIES[family]
    if family == "description_series":
        d = d[d.description_series.ne("")]
    result = []
    for keys, group in d.groupby(MENU_KEYS + extra, sort=True):
        same_age = group[group.age_days.eq(28)]
        if same_age.strength_psi.nunique() < 2:
            continue
        menu_id = hashlib.sha256(repr(keys).encode()).hexdigest()[:16]
        for target in TARGETS:
            exact = same_age[same_age.strength_psi.eq(target)]
            expanded = same_age[same_age.strength_psi.between(target, target + window)]
            if exact.empty or expanded.strength_psi.nunique() < 2:
                continue
            b, c = best(exact), best(expanded)
            aged = group[group.strength_psi.between(target, target + window)]
            any_age = best(aged)
            min28 = c.gwp
            late = aged[aged.age_days.gt(28)]
            only_late_better = not late.empty and late.gwp.min() < min28 - 1e-9
            series_id = hashlib.sha256(repr(keys[: len(MENU_KEYS)]).encode()).hexdigest()[:16]
            rec = {
                "family": family, "window_psi": window, "menu_id": menu_id,
                "plant_issue_id": series_id, "company": b.company,
                "plant": b.plant, "state": b.state, "source": b.source,
                "target_psi": target, "n_exact": len(exact), "n_expanded": len(expanded),
                "baseline_id": int(b.row_id), "candidate_id": int(c.row_id),
                "baseline_mix": b.mix, "candidate_mix": c.mix,
                "baseline_description": b.description, "candidate_description": c.description,
                "baseline_gwp": b.gwp, "candidate_gwp": c.gwp,
                "candidate_strength": c.strength_psi, "gain_kg_m3": b.gwp - c.gwp,
                "gain_fraction": (b.gwp - c.gwp) / b.gwp,
                "break_even_relative_perturbation": (b.gwp - c.gwp) / (b.gwp + c.gwp),
                "different_age_improves": only_late_better,
                "any_age_candidate_id": int(any_age.row_id), "any_age_gwp": any_age.gwp,
                "any_age_days": int(any_age.age_days),
                "late_age_gain_fraction": (min28 - any_age.gwp) / min28,
                "n_late": len(late),
                "baseline_wc": b.wc, "candidate_wc": c.wc,
                "baseline_a1": b.a1, "candidate_a1": c.a1,
                "baseline_a2": b.a2, "candidate_a2": c.a2,
                "baseline_a3": b.a3, "candidate_a3": c.a3,
            }
            assert rec["gain_kg_m3"] >= -1e-9
            assert c.age_days == b.age_days == 28 and c.strength_psi >= target
            if extra:
                assert all(b[x] == c[x] for x in extra)
            result.append(rec)
    return pd.DataFrame(result)


def summary(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"analysis": label, "cases": 0}
    positive = frame.gain_kg_m3.gt(1e-9)
    comp = frame.assign(positive=positive).groupby("company").agg(
        positive_rate=("positive", "mean"), mean_gain=("gain_fraction", "mean")
    )
    return {
        "analysis": label, "cases": len(frame), "companies": frame.company.nunique(),
        "plant_issue_keys": frame.plant_issue_id.nunique(), "menus": frame.menu_id.nunique(),
        "positive_cases": int(positive.sum()), "positive_rate": float(positive.mean()),
        "equal_company_positive_rate": float(comp.positive_rate.mean()),
        "median_gain_all": float(frame.gain_fraction.median()),
        "mean_gain_all": float(frame.gain_fraction.mean()),
        "median_gain_positive": float(frame.loc[positive, "gain_fraction"].median()) if positive.any() else 0,
        "median_kg_gain_positive": float(frame.loc[positive, "gain_kg_m3"].median()) if positive.any() else 0,
        "robust_at_5pct_cases": int(frame.break_even_relative_perturbation.gt(.05).sum()),
        "robust_at_10pct_cases": int(frame.break_even_relative_perturbation.gt(.10).sum()),
        "robust_at_20pct_cases": int(frame.break_even_relative_perturbation.gt(.20).sum()),
        "menus_with_late_age_records": int(frame.n_late.gt(0).sum()),
        "late_age_improves_cases": int(frame.different_age_improves.sum()),
    }


def common_cohort(data: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    """Change candidate scope while keeping every baseline and target fixed."""
    by_id = data.set_index("row_id")
    groups = {k: g for k, g in data.groupby(MENU_KEYS, sort=False)}
    results = []
    for r in strict.itertuples():
        base = by_id.loc[r.baseline_id]
        group = groups[tuple(base[k] for k in MENU_KEYS)]
        pool = group[group.strength_psi.between(r.target_psi, r.target_psi + 1000)]
        age28 = pool[pool.age_days.eq(28)]
        component = age28[age28.components.eq(base.components)]
        desc = component[component.description_series.eq(base.description_series)]
        row = {"menu_id": r.menu_id, "company": r.company, "target_psi": r.target_psi,
               "baseline_id": r.baseline_id, "baseline_gwp": r.baseline_gwp}
        for name, choices_ in [("plant", age28), ("component", component), ("series", desc)]:
            chosen = best(choices_)
            row[f"{name}_candidate_id"] = int(chosen.row_id)
            row[f"{name}_gwp"] = chosen.gwp
            row[f"{name}_gain"] = (base.gwp - chosen.gwp) / base.gwp
        assert np.isclose(row["series_gwp"], r.candidate_gwp)
        assert row["plant_gwp"] <= row["component_gwp"] + 1e-9 <= row["series_gwp"] + 2e-9
        ratio = desc.assign(ratio=desc.gwp / desc.strength_psi)
        tied = ratio[np.isclose(ratio.ratio, ratio.ratio.min(), rtol=0, atol=1e-12)]
        normalized = best(tied)
        row["ratio_candidate_id"] = int(normalized.row_id)
        row["ratio_gwp"] = normalized.gwp
        row["ratio_excess_for_fixed_volume"] = (normalized.gwp - r.candidate_gwp) / r.candidate_gwp
        # Additional requirements are checked as data fields, not assumed from the signature.
        row["both_wc_reported"] = pd.notna(base.wc) and pd.notna(by_id.loc[r.candidate_id].wc)
        row["wc_not_increased"] = row["both_wc_reported"] and by_id.loc[r.candidate_id].wc <= base.wc
        results.append(row)
    return pd.DataFrame(results)


def checks() -> None:
    assert signature("4000 PSI AE", 4000) == "<strength> psi ae"
    assert signature("4,000 PSI AE", 4000) == "<strength> psi ae"
    assert signature("3500 PSI AE", 4000) == ""
    assert signature("4000 PSI / 5000 PSI", 4000) == ""
    assert signature("4000 AE", 4000) == ""
    assert signature("4000PSI PU PL", 4000) == "<strength> psi pu pl"
    assert signature("4000 PSI 4-6SL", 4000) != signature("5000 PSI 5-7SL", 5000)
    for base, candidate in [(300., 240.), (300., 300.), (199., 195.)]:
        u = (base - candidate) / (base + candidate)
        assert np.isclose(candidate * (1 + u), base * (1 - u))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    checks()
    data, profile = prepare()
    results, summaries = [], []
    for family in FAMILIES:
        for window in [500, 1000, 2000]:
            table = choices(data, family, window)
            table.to_csv(OUT / f"choices_{family}_{window}.csv", index=False)
            summaries.append(summary(table, f"{family}_window_{window}"))
            if window == 1000:
                results.append(table)
    main_frames = pd.concat(results, ignore_index=True)
    main_frames.to_csv(OUT / "choices_primary.csv", index=False)
    current = choices(data[data.current_at_analysis], "description_series", 1000)
    current.to_csv(OUT / "choices_current_declarations.csv", index=False)
    summaries.append(summary(current, "description_series_current_on_2026-09-09"))
    series = results[-1]
    common = common_cohort(data, series)
    common.to_csv(OUT / "common_baseline_comparisons.csv", index=False)
    company_table = series.assign(positive=series.gain_kg_m3.gt(1e-9)).groupby("company").agg(
        cases=("positive", "size"), positive=("positive", "sum"), positive_rate=("positive", "mean"),
        mean_gain=("gain_fraction", "mean"), median_gain=("gain_fraction", "median"),
    ).sort_values("cases", ascending=False)
    company_table.to_csv(OUT / "supplier_summary.csv")
    summaries.append(summary(series[series.company.ne("national ready mix")], "description_series_exclude_national_ready_mix"))
    summaries.append(summary(series[series.n_exact.ge(2)], "description_series_at_least_two_exact_options"))
    known_wc = series.merge(common[["menu_id", "target_psi", "wc_not_increased"]], on=["menu_id", "target_psi"])
    summaries.append(summary(known_wc[known_wc.wc_not_increased], "description_series_both_wc_reported_not_increased"))
    ratio_positive = common.ratio_excess_for_fixed_volume.gt(1e-9)
    extra_results = {
        "strict_case_denominator": len(series),
        "common_cohort_positive_by_scope": {x: int(common[f"{x}_gain"].gt(1e-9).sum()) for x in ["plant", "component", "series"]},
        "ratio_excess_cases": int(ratio_positive.sum()),
        "ratio_excess_median_among_positive": float(common.loc[ratio_positive, "ratio_excess_for_fixed_volume"].median()),
        "ratio_excess_companies": int(common.loc[ratio_positive, "company"].nunique()),
        "ratio_excess_cases_without_nrm": int((ratio_positive & common.company.ne("national ready mix")).sum()),
        "known_wc_cases": int(common.both_wc_reported.sum()),
        "known_wc_not_increased_cases": int(common.wc_not_increased.sum()),
    }
    (OUT / "additional_results.json").write_text(json.dumps(extra_results, indent=2), encoding="utf-8")
    for target, frame in series.groupby("target_psi"):
        summaries.append(summary(frame, f"description_series_target_{target}"))
    for cutoff in [5, 10]:
        frame = series[series.n_expanded.ge(cutoff)]
        summaries.append(summary(frame, f"description_series_min_candidates_{cutoff}"))
    loo = []
    for company in series.company.unique():
        s = summary(series[series.company.ne(company)], "leave_one_company_out")
        s["omitted_company"] = company
        loo.append(s)
    pd.DataFrame(loo).to_csv(OUT / "leave_one_company_out.csv", index=False)
    # All targets and controls are retained, including null and contrary cases.
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    (OUT / "profile.json").write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    print(json.dumps(profile, indent=2, default=str))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
