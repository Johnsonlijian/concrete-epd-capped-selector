"""Generate submission figures, tables, and manuscript number macros.

All plotted values are read from frozen derived outputs. This script does not
download data, edit the manuscript, or infer engineering approval.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "derived"
F = ROOT / "figures"
M = ROOT / "manuscript"
S = next((p for p in (ROOT / "sources" / "agent_nrmca", ROOT.parent / "sources" / "agent_nrmca") if p.exists()),
         ROOT / "sources" / "agent_nrmca")

TEAL, RED, GOLD, BLUE = "#007C83", "#B43F4A", "#AD8100", "#2D5F8B"
GRAY, LIGHT = "#7A8187", "#D9E4E5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def save(fig: plt.Figure, name: str) -> None:
    F.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(F / f"{name}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    F.mkdir(exist_ok=True)
    M.mkdir(exist_ok=True)
    choices = pd.read_csv(D / "choices_description_series_1000.csv", low_memory=False)
    common = pd.read_csv(D / "common_baseline_comparisons.csv", low_memory=False)
    quantity = pd.read_csv(D / "quantity_model_scenarios.csv", low_memory=False)
    qsummary = pd.read_csv(D / "quantity_model_summary.csv", low_memory=False)
    summary = pd.read_csv(D / "summary.csv", low_memory=False).set_index("analysis")
    profile = json.loads((D / "profile.json").read_text(encoding="utf-8"))
    knife_audit = json.loads((S / "knife_river_extraction_audit.json").read_text(encoding="utf-8"))

    # Figure 1: piecewise quantity rule. This is an analytical illustration,
    # not a claim about a project design.
    widths, loads = [12, 16, 18], [0.75, 1.0, 1.25]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), layout="constrained")
    strength = np.linspace(2500, 9000, 300)
    fy, steel_area = 60000.0, 8.0
    pu_ref = 797.68 / 0.8
    for width, color in zip(widths, [BLUE, TEAL, RED]):
        for ratio in loads:
            k = (ratio * pu_ref / 0.65 - fy * steel_area / 1000.0) / 0.85 * 1000.0
            amin = width * width - steel_area
            fstar = k / amin
            area = np.maximum(k / strength, amin)
            axes[0].plot(strength, area / (width * width), color=color,
                         alpha=0.35 + 0.2 * (ratio - 0.75),
                         label=f"{width} in, {ratio:.2g} Pu" if ratio == 1.0 else None)
            if ratio == 1.0:
                axes[0].axvline(fstar, color=color, lw=.8, ls="--")
    axes[0].set(xlabel="Reported compressive strength (psi)",
                ylabel="Concrete area / gross area",
                title="a  Geometry floor stops the quantity credit")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fstar = 5000.0
    index = 1 / np.minimum(strength, fstar)
    axes[1].plot(strength, index / index[0], color=TEAL)
    axes[1].axvline(fstar, color=GRAY, ls="--", lw=.9)
    axes[1].annotate("strength credit\nactive", xy=(3900, .45), xytext=(2800, .22),
                     arrowprops=dict(arrowstyle="->", lw=.7), fontsize=8)
    axes[1].annotate("floor active", xy=(6900, .2), xytext=(7100, .48),
                     arrowprops=dict(arrowstyle="->", lw=.7), fontsize=8)
    axes[1].set(xlabel="Reported compressive strength (psi)",
                ylabel=r"Relative $g/\min(f,f^*)$ index",
                title="b  The capped index is piecewise")
    save(fig, "Fig1_quantity_constraint")

    # Figure 2: quantity-model consequences across the scenario grid.
    q = qsummary.copy().sort_values(["minimum_width_in", "demand_ratio"])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), layout="constrained")
    width_values = sorted(q.minimum_width_in.unique())
    demand_values = sorted(q.demand_ratio.unique())
    x = np.arange(len(width_values))
    demand_colors = [BLUE, TEAL, RED]
    bar_width = 0.22
    for index, demand in enumerate(demand_values):
        subset = q[q.demand_ratio.eq(demand)].set_index("minimum_width_in").loc[width_values]
        offset = (index - (len(demand_values) - 1) / 2) * bar_width
        label = f"{demand:.2f}".rstrip("0").rstrip(".") + r" $P_{u,ref}$"
        axes[0].bar(x + offset, subset.affected_fraction * 100, color=demand_colors[index],
                    width=bar_width, label=label)
        axes[1].bar(x + offset, subset.median_excess_affected * 100, color=demand_colors[index],
                    width=bar_width, label=label)
    axes[0].set(xticks=x, xticklabels=[f"{int(w)} in" for w in width_values],
                ylabel="Ordinary $g/f$ is carbon-suboptimal (%)",
                title="a  Regime prevalence in the scenario grid")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left", title="Demand ratio", title_fontsize=7)
    axes[1].set(xticks=x, xticklabels=[f"{int(w)} in" for w in width_values],
                ylabel="Median excess GWP among affected cases (%)",
                title="b  Excess when the floor changes the choice")
    fig.suptitle("Fixed-steel quantity model applied to the declaration menus", fontsize=10)
    save(fig, "Fig2_quantity_scenarios")

    # Figure 3: empirical opportunity and supplier concentration.
    nrm = choices.company.eq("national ready mix")
    cohorts = [("All", choices, TEAL), ("NRM", choices[nrm], GOLD),
               ("Other", choices[~nrm], BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), layout="constrained")
    rates = [x[1].gain_kg_m3.gt(1e-9).mean() * 100 for x in cohorts]
    axes[0].bar(np.arange(3), rates, color=[x[2] for x in cohorts], width=.6)
    for i, value in enumerate(rates):
        axes[0].text(i, value + 1.2, f"{value:.1f}%", ha="center", fontsize=8)
    axes[0].set(xticks=np.arange(3), xticklabels=[x[0] for x in cohorts],
                ylabel="Cases with lower reported GWP (%)",
                title="a  A wider strength window", ylim=(0, max(rates) + 10))
    robust5 = [x[1].break_even_relative_perturbation.gt(.05).mean() * 100 for x in cohorts]
    robust10 = [x[1].break_even_relative_perturbation.gt(.10).mean() * 100 for x in cohorts]
    xx = np.arange(3)
    axes[1].bar(xx - .16, robust5, width=.3, color=TEAL, label="5% bound")
    axes[1].bar(xx + .16, robust10, width=.3, color=RED, label="10% bound")
    axes[1].set(xticks=xx, xticklabels=[x[0] for x in cohorts],
                ylabel="Fixed-pair ordering retained (%)",
                title="b  Deterministic stress bounds")
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, "Fig3_epd_opportunity_and_robustness")

    # Figure 4: independent source extraction audit.
    plant = pd.DataFrame(knife_audit["by_plant"]).T
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), layout="constrained")
    xx = np.arange(len(plant))
    axes[0].bar(xx, plant.verified, color=TEAL, label="Source-checked")
    axes[0].bar(xx, plant.declared - plant.verified, bottom=plant.verified,
                color=LIGHT, edgecolor=GRAY, linewidth=.5, label="Quarantined")
    axes[0].set(xticks=xx, xticklabels=plant.index, ylabel="Plant-level records",
                title="a  Historical source-complete extraction")
    axes[0].legend(frameon=False, fontsize=7)
    source_rows = pd.read_csv(S / "knife_river_verified.csv", low_memory=False)
    source_rows = source_rows[source_rows.family_eligible]
    axes[1].scatter(source_rows.strength_psi, source_rows.gwp_kgco2e_m3,
                    s=12, alpha=.38, color=BLUE, edgecolors="none")
    axes[1].set(xlabel="Reported strength (psi)",
                ylabel=r"A1--A3 GWP (kg CO$_2$-eq/m$^3$)",
                title="b  Verified rows retained for rule checking")
    save(fig, "Fig4_source_audit")

    # Main-text tables.  Keep the row ending in one place so generated
    # LaTeX rows cannot silently lose one of the two backslashes.
    row_end = r"\\"
    coverage = [r"\begin{tabular}{lr}", r"\toprule", r"Evidence stage & Count " + row_end, r"\midrule",
                f"Raw compilation records & {profile['raw_rows']:,} " + row_end,
                f"After eligibility and local duplicate handling & {profile['analysis_rows']:,} " + row_end,
                f"28-day records & {profile['analysis_28_day_rows']:,} " + row_end,
                f"Concordant explicit-PSI records & {profile['explicit_psi_series_rows_28d']:,} " + row_end,
                f"Paired primary scenarios & {len(choices):,} " + row_end,
                f"Suppliers represented in primary scenarios & {choices.company.nunique():,} " + row_end,
                r"\bottomrule", r"\end{tabular}"]
    (M / "table_coverage.tex").write_text("\n".join(coverage) + "\n", encoding="ascii")
    quantity_table = [r"\begin{tabular}{rrr rrr}", r"\toprule",
                      r"Minimum width & Demand & Cases & Floor active & Median excess & $f^*$ at reference " + row_end,
                      r"(in) & ($P_u/P_{u,ref}$) & & (\%) & (\%) & (psi) " + row_end, r"\midrule"]
    for _, row in q.iterrows():
        demand_label = f"{row.demand_ratio:.2f}".rstrip("0").rstrip(".")
        quantity_table.append(f"{row.minimum_width_in:.0f} & {demand_label} & {int(row.cases):,} & "
                              f"{row.affected_fraction*100:.1f} & {row.median_excess_affected*100:.1f} & "
                              f"{row.saturation_strength_psi:,.0f} " + row_end)
    quantity_table.extend([r"\bottomrule", r"\end{tabular}"])
    (M / "table_quantity_scenarios.tex").write_text("\n".join(quantity_table) + "\n", encoding="ascii")
    source_table = [r"\begin{tabular}{lrrr}", r"\toprule",
                    r"Source plant & Declared & Verified & Quarantined " + row_end, r"\midrule"]
    for plant_name, row in plant.iterrows():
        source_table.append(f"{plant_name} & {int(row.declared)} & {int(row.verified)} & {int(row.declared-row.verified)} " + row_end)
    source_table.extend([f"Total & {knife_audit['properties']} & {knife_audit['verified']} & {knife_audit['quarantined']} " + row_end,
                         r"\bottomrule", r"\end{tabular}"])
    (M / "table_source_audit.tex").write_text("\n".join(source_table) + "\n", encoding="ascii")

    # Macros are regenerated from the same outputs used by the figures.
    ratio = common.ratio_excess_for_fixed_volume.gt(1e-9)
    other = choices.company.ne("national ready mix")
    age = choices[choices.n_late.gt(0)]
    qjson = json.loads((D / "quantity_model_results.json").read_text(encoding="utf-8"))
    stats = {
        "RawRecords": profile["raw_rows"], "AnalysisRecords": profile["analysis_rows"],
        "AgeRecords": profile["analysis_28_day_rows"], "SeriesRecords": profile["explicit_psi_series_rows_28d"],
        "Cases": len(choices), "Menus": choices.menu_id.nunique(), "PlantIssues": choices.plant_issue_id.nunique(),
        "Companies": choices.company.nunique(), "Improved": int(choices.gain_kg_m3.gt(1e-9).sum()),
        "ImprovedPct": choices.gain_kg_m3.gt(1e-9).mean() * 100,
        "NrmCases": int(nrm.sum()), "NrmImproved": int((nrm & choices.gain_kg_m3.gt(1e-9)).sum()),
        "NrmImprovedShare": (nrm & choices.gain_kg_m3.gt(1e-9)).sum() / choices.gain_kg_m3.gt(1e-9).sum() * 100,
        "OtherCases": int(other.sum()), "OtherImproved": int((other & choices.gain_kg_m3.gt(1e-9)).sum()),
        "OtherImprovedPct": (other & choices.gain_kg_m3.gt(1e-9)).sum() / other.sum() * 100,
        "EqualCompanyPct": summary.loc["description_series_window_1000", "equal_company_positive_rate"] * 100,
        "ImprovedMedianPct": choices.loc[choices.gain_kg_m3.gt(1e-9), "gain_fraction"].median() * 100,
        "ImprovedMedianKg": choices.loc[choices.gain_kg_m3.gt(1e-9), "gain_kg_m3"].median(),
        "RatioExcess": int(ratio.sum()), "RatioExcessPct": ratio.mean() * 100,
        "RatioMedianPct": common.loc[ratio, "ratio_excess_for_fixed_volume"].median() * 100,
        "VolumeThresholdPct": (1 - common.loc[ratio, "series_gwp"] / common.loc[ratio, "ratio_gwp"]).median() * 100,
        "RatioOther": int((ratio & other.to_numpy()).sum()), "RatioOtherPct": (ratio & other.to_numpy()).sum() / other.sum() * 100,
        "RobustFive": int(choices.break_even_relative_perturbation.gt(.05).sum()),
        "RobustTen": int(choices.break_even_relative_perturbation.gt(.10).sum()),
        "OtherRobustFive": int((choices.loc[other, "break_even_relative_perturbation"] > .05).sum()),
        "OtherRobustTen": int((choices.loc[other, "break_even_relative_perturbation"] > .10).sum()),
        "LateCases": len(age), "LateImproved": int(age.different_age_improves.sum()),
        "ConflictRecords": profile["identity_conflict_rows_raw"],
        "AliasQuarantinedRows": profile["alias_reconciliation"]["quarantined_rows"],
        "AliasDuplicateRows": profile["alias_reconciliation"]["duplicate_rows_removed"],
        "BenchmarkPoints": 14, "BenchmarkMaxDiff": qjson["max_absolute_benchmark_difference"],
        "SourceVerified": knife_audit["verified"], "SourceQuarantined": knife_audit["quarantined"],
    }
    lines = ["% Generated from derived CSV and JSON outputs. Do not edit manually."]
    for key, value in stats.items():
        if isinstance(value, (float, np.floating)):
            formatted = f"{float(value):.3f}".rstrip("0").rstrip(".")
        else:
            formatted = str(value)
        lines.append(f"\\newcommand{{\\{key}}}{{{formatted}}}")
    (M / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="ascii")
    (D / "manuscript_numbers.json").write_text(json.dumps(stats, indent=2, default=float) + "\n", encoding="utf-8")
    pd.DataFrame([{
        "source": "Knife River NRMCA EPD 10023", "pdf_sha256": knife_audit["source_sha256"],
        "declared_rows": knife_audit["properties"], "source_checked_rows": knife_audit["verified"],
        "quarantined_rows": knife_audit["quarantined"], "claim_boundary": knife_audit["claim_boundary"],
    }]).to_csv(D / "source_audit_summary.csv", index=False)
    print(json.dumps(stats, indent=2, default=float))


if __name__ == "__main__":
    main()
