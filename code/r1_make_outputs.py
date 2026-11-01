"""Figures, tables and number macros for the revised manuscript (R1).

Every number printed in the revised manuscript is read from a derived file
written by the analysis scripts; nothing is typed by hand.  Quantities are
computed in US customary units (the data and ACI 318 are US based) and are
reported in SI units, with US customary values in parentheses where useful.

Figures follow the journal's artwork rules: 174 mm final width, sans-serif
lettering of 8-10 pt, panel letters but no titles inside the artwork, markers,
line styles and hatching in addition to colour, and no transparency (EPS).

Outputs:
  manuscript/numbers_r1.tex      LaTeX macros
  manuscript/tab_r1_*.tex        main-text table bodies
  manuscript/tab_si_*.tex        supplementary table bodies
  figures/Fig{1..4}_*.{pdf,eps,svg,png}
"""

from __future__ import annotations

import json
import os

# Fixed file timestamps (13 October 2026, UTC) so that PDF, EPS and SVG outputs are byte-reproducible.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1791849600")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import r1_core as rc

D = rc.OUT
F = rc.ROOT / "figures"
M = rc.ROOT / "manuscript"

PSI_TO_MPA = 0.006894757
IN_TO_MM = 25.4
KIP_TO_KN = 4.448222
FT_TO_M = 0.3048

TEAL, RED, GOLD, BLUE = "#007C83", "#B43F4A", "#8A6A00", "#2D5F8B"
GRAY = "#6B7177"
SEL_ORDER = ("fixed_volume", "ratio", "capped", "capped_effective")
SEL_STYLE = {"fixed_volume": dict(color=GOLD, marker="s", ls=":"),
             "ratio": dict(color=RED, marker="^", ls="--"),
             "capped": dict(color=TEAL, marker="o", ls="-"),
             "capped_effective": dict(color=BLUE, marker="D", ls="-.")}
SEL_LABELS = {"fixed_volume": "Fixed volume $g$", "ratio": "Ratio $g/f$",
              "capped": "Capped, nominal floor", "capped_effective": "Capped, governing minimum"}
FONT = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
WIDTH_IN = 174.0 / 25.4
plt.rcParams.update({
    "font.family": FONT, "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0, "legend.fontsize": 8.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "lines.linewidth": 1.4, "lines.markersize": 4.5, "hatch.linewidth": .6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "maas-r1",
    "savefig.facecolor": "white",
})


def mpa(psi):
    return np.asarray(psi, dtype=float) * PSI_TO_MPA


def save(fig, name: str) -> None:
    F.mkdir(exist_ok=True)
    for suffix in ("pdf", "eps", "svg", "png"):
        fig.savefig(F / f"{name}.{suffix}", dpi=600 if suffix == "png" else None)
    plt.close(fig)


def panel(ax, letter: str, x: float = -0.20) -> None:
    ax.text(x, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def pct(x: float, nd: int = 1) -> str:
    value = 100.0 * float(x)
    if nd == 1 and 99.95 <= value < 100.0:
        return r"$>$99.9"
    return f"{value:.{nd}f}"


def num(x: float, nd: int = 0) -> str:
    return f"{float(x):,.{nd}f}"


def read_json(name: str) -> dict:
    return json.loads((D / name).read_text(encoding="utf-8"))


def selector_legend(fig, selectors=SEL_ORDER) -> None:
    handles = [Line2D([0], [0], color=SEL_STYLE[s]["color"], marker=SEL_STYLE[s]["marker"],
                      ls=SEL_STYLE[s]["ls"], label=SEL_LABELS[s]) for s in selectors]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
               handlelength=2.6, columnspacing=1.4)


def figure_concept() -> float:
    c = rc.build_candidates()
    # An illustrative menu whose capped choice is intermediate for some f* (largest such menu up to 12 candidates).
    fv, ratio = rc.select_fixed_volume(c), rc.select_ratio(c)
    chosen, chosen_fs = None, None
    for i in np.argsort(-c.valid.sum(axis=1), kind="stable"):
        if c.valid[i].sum() > 12 or c.take(ratio, c.gwp)[i] <= c.take(fv, c.gwp)[i] + 1e-9:
            continue
        for fs in np.arange(2600.0, 7000.0, 50.0):
            q = rc.select_capped(c, fs)[i]
            if not np.isclose(c.gwp[i, q], c.gwp[i, fv[i]]) and not np.isclose(c.gwp[i, q], c.gwp[i, ratio[i]]):
                chosen, chosen_fs = i, fs
                break
        if chosen is not None:
            break
    i = chosen
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.55), layout="constrained")
    strength = np.linspace(2500, 9000, 400)
    for fs, color, ls in ((3200.0, GOLD, "-."), (5000.0, TEAL, "-"), (8000.0, RED, "--")):
        axes[0].plot(mpa(strength), np.maximum(fs / strength, 1.0), color=color, ls=ls,
                     label=f"$f^*$ = {mpa(fs):.0f} MPa")
        axes[0].axvline(mpa(fs), color=color, lw=.6, ls=":")
    axes[0].set(xlabel="Specified strength $f$ (MPa)", ylabel=r"Concrete area / $(A_{\min}-A_s)$")
    axes[0].legend(frameon=False, loc="upper right", handlelength=2.4)
    fs = 5000.0
    axes[1].plot(mpa(strength), 2500.0 / strength, color=RED, ls="--", label="Ratio $g/f$")
    axes[1].plot(mpa(strength), np.full_like(strength, 0.5), color=GOLD, ls=":", lw=1.8, label="Fixed volume $g$")
    axes[1].plot(mpa(strength), 2500.0 / np.minimum(strength, fs), color=TEAL, ls="-", lw=1.0,
                 label=r"Capped $g/\min(f,f^*)$")
    axes[1].set(xlabel="Specified strength $f$ (MPa)", ylabel="Index per unit $g$ (scaled)")
    axes[1].legend(frameon=False, loc="upper right", handlelength=2.4)
    v = c.valid[i]
    axes[2].scatter(mpa(c.strength[i][v]), c.gwp[i][v], s=14, color=GRAY, marker="o", zorder=2, label="Candidate")
    for idx, sel in ((fv[i], "fixed_volume"), (ratio[i], "ratio"), (rc.select_capped(c, chosen_fs)[i], "capped")):
        st = SEL_STYLE[sel]
        axes[2].scatter(mpa(c.strength[i, idx]), c.gwp[i, idx], s=70, facecolors="none", edgecolors=st["color"],
                        marker=st["marker"], linewidths=1.4, zorder=3,
                        label={"fixed_volume": "Fixed volume", "ratio": "Ratio", "capped": "Capped"}[sel])
    gv = c.gwp[i][v]
    axes[2].set(xlabel="Specified strength $f$ (MPa)", ylabel="A1\u2013A3 GWP $g$ (kg CO$_2$-eq m$^{-3}$)",
                ylim=(gv.min() - 0.08 * np.ptp(gv), gv.max() + 0.6 * np.ptp(gv)))
    top = min(t for t in axes[2].get_yticks() if t >= gv.max())
    ticks = [t for t in axes[2].get_yticks() if gv.min() - 0.08 * np.ptp(gv) <= t <= top]
    axes[2].set_yticks(ticks)
    axes[2].spines["left"].set_bounds(gv.min() - 0.08 * np.ptp(gv), ticks[-1])
    axes[2].legend(frameon=False, loc="upper center", ncol=2, columnspacing=0.8, handletextpad=0.2)
    for ax, letter in zip(axes, "abc"):
        panel(ax, letter)
    save(fig, "Fig1_selector_concept")
    return float(mpa(chosen_fs))


def figure_comparisons(dec: pd.DataFrame, boot: pd.DataFrame, sweep: pd.DataFrame, grid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.75), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.35]})
    nrm = dec.company.eq(rc.NRM)
    groups = [("All", dec), ("Largest\ncatalogue", dec[nrm]), ("Other\nsuppliers", dec[~nrm])]
    x = np.arange(3)
    for k, (col, color, hatch, lab) in enumerate((("c1_lower_gwp", BLUE, None, "C1: wider window lowers $g$"),
                                                   ("c2_ratio_higher_gwp", RED, "////", "C2: ratio picks higher $g$"))):
        vals = [grp[col].mean() * 100 for _, grp in groups]
        axes[0].bar(x + (k - .5) * .36, vals, width=.34, color=color, hatch=hatch, edgecolor="white", lw=0,
                    label=lab)
        lo, hi = boot.loc[col, "lo"] * 100, boot.loc[col, "hi"] * 100
        axes[0].errorbar(x[0] + (k - .5) * .36, vals[0], yerr=[[vals[0] - lo], [hi - vals[0]]], color="black",
                         capsize=2.5, lw=.9)
        for j, (xi, val) in enumerate(zip(x, vals)):
            top = hi if j == 0 else val
            axes[0].text(xi + (k - .5) * .36, top + 2, f"{val:.0f}", ha="center", fontsize=8)
    axes[0].set(xticks=x, xticklabels=[g[0] for g in groups], ylabel="Share of matched cases (%)", ylim=(0, 132),
                yticks=np.arange(0, 101, 20))
    axes[0].spines["left"].set_bounds(0, 100)
    axes[0].legend(frameon=False, loc="upper left")
    s = sweep.copy()
    tot = s.c2_cases.to_numpy(dtype=float)
    layers = [(s.c2_resolved_to_fixed_volume / tot * 100, GOLD, "....", "Capped = fixed volume"),
              (s.c2_resolved_intermediate / tot * 100, TEAL, "xxxx", "Capped = intermediate"),
              (s.c2_resolved_to_ratio / tot * 100, RED, "////", "Capped = ratio")]
    xs = mpa(s.fstar_psi.to_numpy())
    base = np.zeros_like(xs)
    for share, color, hatch, lab in layers:
        axes[1].fill_between(xs, base, base + share.to_numpy(), facecolor=color, hatch=hatch, edgecolor="white",
                             lw=0, label=lab)
        base = base + share.to_numpy()
    for fstar in grid.fstar_psi:
        if fstar <= s.fstar_psi.max():
            axes[1].axvline(mpa(fstar), color="black", lw=.6, ls=":")
    axes[1].set(xlabel=r"Transition strength $f^*$ (MPa)", ylabel="Resolution of C2 cases (%)",
                xlim=(xs.min(), xs.max()), ylim=(0, 100))
    axes[1].legend(frameon=True, framealpha=1, edgecolor="white", loc="center right")
    for ax, letter in zip(axes, "ab"):
        panel(ax, letter, x=-0.14)
    save(fig, "Fig2_three_comparisons")


SENS_LABELS = [("main", "Main setting"), ("strict_pcr_5_5", "Strict observable-data subset"),
               ("column_admissible", "Column-admissible subset"), ("fcr_reinterpretation", r"Strength read as $f'_{cr}$"),
               ("window_500psi", "Window 3.4 MPa"), ("window_2000psi", "Window 13.8 MPa"),
               ("slenderness_always", "Magnifier on all columns"), ("storey_10ft", "3.05 m storeys"),
               ("size_step_0.5in", "13 mm size module"), ("rebar_0.5", "Rebar 0.5 kg/kg"),
               ("rebar_2.0", "Rebar 2.0 kg/kg"), ("tie_break_submitted", "Record-order tie-break")]


def figure_design(bands: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 3.35), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    order = ["1-2", "3-5", "6-10", "11-15", "16-20"]
    for ax, floor in ((axes[0], 12.0), (axes[1], 20.0)):
        part = bands[(bands.label == "main") & (bands.policy == "min_carbon") & (bands.floor_in == floor)]
        for sel in SEL_ORDER:
            p = part[part.selector == sel].set_index("band").reindex(order)
            st = SEL_STYLE[sel]
            ax.plot(np.arange(5), p.agreement * 100, color=st["color"], marker=st["marker"], ls=st["ls"])
        ax.set(xticks=np.arange(5), ylim=(0, 104), xlabel="Levels supported")
        ax.set_xticklabels([b.replace("-", "\u2013") for b in order], fontsize=7.5)
        ax.text(0.5, 0.05, f"Nominal floor {floor * IN_TO_MM:.0f} mm", transform=ax.transAxes, ha="center",
                fontsize=8)
    axes[0].set_ylabel("Agreement with design optimum (%)")
    axes[1].set_yticklabels([])
    tot = summary[(summary.floor_in.astype(str) == "all") & (summary.policy == "min_carbon")]
    labels = [(lab, txt) for lab, txt in SENS_LABELS if (tot.label == lab).any()]
    y = np.arange(len(labels))[::-1]
    for sel in SEL_ORDER:
        vals = [tot[(tot.label == lab) & (tot.selector == sel)].agreement.mean() * 100 for lab, _ in labels]
        st = SEL_STYLE[sel]
        axes[2].scatter(vals, y, s=20, color=st["color"], marker=st["marker"], zorder=3)
    axes[2].set(yticks=y, yticklabels=[t for _, t in labels], xlim=(40, 100), xlabel="Agreement, all contexts (%)")
    axes[2].grid(axis="y", color="#E3E6E8", lw=.6)
    for ax, letter, x in zip(axes, "abc", (-0.20, -0.08, -0.95)):
        panel(ax, letter, x=x)
    selector_legend(fig)
    save(fig, "Fig3_design_validation")


def figure_robustness(mc: pd.DataFrame, design_rob: pd.DataFrame, ladder: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.95), layout="constrained")
    models = [("independent", 0.025, "2.5%"), ("independent", 0.05, "5%"), ("independent", 0.10, "10%"),
              ("structured", 0.0, "Struct.")]
    xx = np.arange(len(models))
    for k, (col, sel) in enumerate((("stable_fixed_volume", "fixed_volume"), ("stable_ratio", "ratio"),
                                    ("stable_capped_16in_1", "capped"))):
        med, lo, hi = [], [], []
        for model, sig, _ in models:
            p = mc[(mc.model == model) & np.isclose(mc.sigma, sig)][col] * 100
            med.append(p.median()); lo.append(p.quantile(.025)); hi.append(p.quantile(.975))
        st = SEL_STYLE[sel]
        axes[0].errorbar(xx + (k - 1) * .2, med, yerr=[np.array(med) - lo, np.array(hi) - med], fmt=st["marker"],
                         color=st["color"], capsize=2, lw=.9, ms=4.5)
    axes[0].set(xticks=xx, xticklabels=[m[2] for m in models], ylim=(50, 100), xlabel="Perturbation model",
                ylabel="Choices unchanged (%)")
    pert = design_rob[design_rob.replicate > 0]
    for k, sel in enumerate(SEL_ORDER):
        med, lo, hi = [], [], []
        for model, sig, _ in models:
            p = pert[(pert.model == model) & np.isclose(pert.sigma, sig) & (pert.selector == sel)]
            per_rep = p.groupby("replicate").agreement.mean() * 100
            med.append(per_rep.median()); lo.append(per_rep.quantile(.025)); hi.append(per_rep.quantile(.975))
        st = SEL_STYLE[sel]
        axes[1].errorbar(xx + (k - 1.5) * .18, med, yerr=[np.array(med) - lo, np.array(hi) - med], fmt=st["marker"],
                         color=st["color"], capsize=2, lw=.9, ms=4.5)
    axes[1].set(xticks=xx, xticklabels=[m[2] for m in models], ylim=(40, 100), xlabel="Perturbation model",
                ylabel="Agreement with design optimum (%)")
    lab = {"plant_scope": "Any\nmix", "component_scope": "Same\ncomp.", "description_series_primary": "Primary\nrule",
           "strict_pcr_5_5": "Strict\nobs.", "strict_pcr_5_5_current": "Strict\ncurrent"}
    xl = np.arange(len(ladder))
    for col, color, marker, ls, txt in (("c1_rate", BLUE, "o", "-", "C1"), ("c2_rate", RED, "^", "--", "C2"),
                                        ("c3_16in_1", TEAL, "s", ":", "C3")):
        axes[2].plot(xl, ladder[col] * 100, color=color, marker=marker, ls=ls, label=txt)
    axes[2].set(xticks=xl, ylim=(0, 100), ylabel="Share of matched cases (%)")
    axes[2].set_xticklabels([lab[s] for s in ladder.scope], fontsize=7.5)
    axes[2].legend(frameon=False, loc="upper right")
    for ax, letter in zip(axes, "abc"):
        panel(ax, letter)
    selector_legend(fig)
    save(fig, "Fig4_robustness")


def main() -> None:
    M.mkdir(exist_ok=True)
    from r1_decomposition import PU_REF_KIP
    import r1_column_design as cd

    dsum = read_json("r1_decomposition_summary.json")
    dec = pd.read_csv(D / "r1_decomposition_cases.csv")
    grid = pd.read_csv(D / "r1_decomposition_grid.csv")
    sweep = pd.read_csv(D / "r1_decomposition_fstar_sweep.csv")
    boot = pd.read_csv(D / "r1_bootstrap_supplier_clusters.csv", index_col=0)
    design = pd.read_csv(D / "r1_design_validation_summary.csv")
    bands = pd.read_csv(D / "r1_design_validation_bands.csv")
    strict_design = pd.read_csv(D / "r1_strict_design_summary.csv")
    fcr_design = pd.read_csv(D / "r1_fcr_design_summary.csv")
    ext_design = pd.read_csv(D / "r1_extended_design_summary.csv")
    design_all = pd.concat([design, strict_design, fcr_design, ext_design], ignore_index=True)
    demands = pd.read_csv(D / "r1_design_demands.csv")
    mc = pd.read_csv(D / "r1_robustness_montecarlo.csv")
    design_rob = pd.read_csv(D / "r1_robustness_design.csv")
    dominance = pd.read_csv(D / "r1_robustness_design_dominance.csv")
    ladder = pd.read_csv(D / "r1_comparability_ladder.csv")
    strict = read_json("r1_strict_subset_summary.json")
    prov = read_json("r1_field_provenance_summary.json")
    bench = pd.read_csv(D / "external_section_benchmark.csv")
    governing = pd.read_csv(D / "r1_design_governing.csv")
    stack = pd.read_csv(D / "r1_design_stack.csv")
    diffs = pd.read_csv(D / "r1_bootstrap_selector_differences.csv")
    subsets = pd.read_csv(D / "r1_extended_subsets.csv").set_index("subset")

    example_fstar = figure_concept()
    figure_comparisons(dec, boot, sweep, grid)
    figure_design(bands, design_all)
    figure_robustness(mc, design_rob, ladder)

    def agree(label, policy, sel, floor="all", col="agreement", frame=design_all):
        p = frame[(frame.label == label) & (frame.policy == policy) & (frame.selector == sel)
                  & (frame.floor_in.astype(str) == str(floor))]
        if len(p) != 1:
            raise KeyError((label, policy, sel, floor, len(p)))
        return float(p[col].iloc[0])

    sel_tags = (("fixed_volume", "Fixed"), ("ratio", "Ratio"), ("capped", "Capped"), ("capped_effective", "Effective"))
    macros = {"ExampleFstar": f"{example_fstar:.0f}"}
    # Provenance.
    vp, tc, kr = prov["value_pattern"], prov["token_concordance"], prov["knife_river_cross_declaration"]
    macros.update(StrRecords=num(vp["records_28d"]), StrMultFiveHundred=pct(vp["share_multiple_of_500"]),
                  StrMultTwoFifty=pct(vp["share_multiple_of_250"]), StrDistinct=str(vp["distinct_values"]),
                  PrimaryMultFiveHundred=pct(prov["primary_candidates_value_pattern"]["share_multiple_of_500"]),
                  PrimaryDistinct=str(prov["primary_candidates_value_pattern"]["distinct_values"]),
                  TokenSingle=num(tc["one_concordant"] + tc["one_discordant"]), TokenConcordant=num(tc["one_concordant"]),
                  TokenDiscordant=num(tc["one_discordant"]), TokenNone=num(tc["none"]),
                  TokenConcordantPct=pct(tc["concordance_among_single_token"]),
                  KRPairs=str(kr["matched_pairs"]), KRNumericPairs=str(kr["matched_pairs_numeric_strength_both"]),
                  KRCheckedPairs=str(kr["matched_source_checked_pairs"]),
                  KRCheckedSame=pct(kr["strength_identical_share_source_checked"]),
                  KRQuarantinedSame=pct(kr["strength_identical_share_quarantined"]),
                  KRGwpRatio=f"{kr['gwp_ratio_compilation_to_2018_median']:.2f}",
                  KRGwpRatioLo=f"{kr['gwp_ratio_q25_q75'][0]:.2f}", KRGwpRatioHi=f"{kr['gwp_ratio_q25_q75'][1]:.2f}")
    fr = prov["fcr_reinterpretation"]
    macros.update(FcrCases=num(fr["cases_retained"]), FcrDropped=num(fr["cases_dropped_below_2500psi_or_single_strength"]),
                  FcrRatioChanged=pct(fr["ratio_choice_changed_by_reinterpretation"]), FcrCTwo=pct(fr["c2_rate"]),
                  FcrCapped=pct(fr["design_agreement"]["min_carbon|capped"]),
                  FcrEffective=pct(fr["design_agreement"]["min_carbon|capped_effective"]),
                  FcrRatio=pct(fr["design_agreement"]["min_carbon|ratio"]),
                  FcrFixed=pct(fr["design_agreement"]["min_carbon|fixed_volume"]))
    # Decomposition.
    c1, c2 = dsum["c1"], dsum["c2"]
    nrm_share = (dec.c1_lower_gwp & dec.company.eq(rc.NRM)).sum() / dec.c1_lower_gwp.sum()
    macros.update(COneNrmShare=pct(nrm_share), NrmCasesR=num(dec.company.eq(rc.NRM).sum()))
    macros.update(COneCases=num(c1["lower_gwp_cases"]), COneRate=pct(c1["rate"]), COneOther=pct(c1["rate_outside_nrm"]),
                  COneOtherCases=num(c1["lower_gwp_cases_outside_nrm"]),
                  COneMedianGain=pct(c1["median_gain_affected"]),
                  CTwoCases=num(c2["ratio_higher_gwp_cases"]), CTwoRate=pct(c2["rate"]),
                  CTwoOther=pct(c2["rate_outside_nrm"]), CTwoOtherCases=num(c2["cases_outside_nrm"]),
                  CTwoExcess=pct(c2["median_excess_affected"]), CTwoBreakEven=pct(c2["median_break_even_volume_reduction"]),
                  CTwoCOneOverlap=num(c2["overlap_with_c1"]),
                  PropFstarCount=str(dsum["proposition2_check"]["fstar_values_checked"]),
                  CThreeMin=pct(dsum["grid_c3_rate_range"][0]), CThreeMax=pct(dsum["grid_c3_rate_range"][1]),
                  CThreePMin=pct(dsum["grid_c3p_rate_range"][0]), CThreePMax=pct(dsum["grid_c3p_rate_range"][1]),
                  COneLo=pct(boot.loc["c1_lower_gwp", "lo"]), COneHi=pct(boot.loc["c1_lower_gwp", "hi"]),
                  CTwoLo=pct(boot.loc["c2_ratio_higher_gwp", "lo"]), CTwoHi=pct(boot.loc["c2_ratio_higher_gwp", "hi"]))
    g16 = grid[(grid.minimum_width_in == 16) & (grid.demand_ratio == 1.0)].iloc[0]
    macros.update(GridSixteenCThree=pct(g16.c3_rate), GridSixteenCThreeP=pct(g16.c3p_rate),
                  GridSixteenToFV=num(g16.c2_resolved_to_fixed_volume), GridSixteenToRatio=num(g16.c2_resolved_to_ratio),
                  GridSixteenMid=num(g16.c2_resolved_intermediate))
    # Design set.
    macros.update(DemandMinKN=num(demands.pu_kip.min() * KIP_TO_KN), DemandMaxKN=num(demands.pu_kip.max() * KIP_TO_KN),
                  DemandMin=num(demands.pu_kip.min()), DemandMax=num(demands.pu_kip.max()), DemandCount=str(len(demands)),
                  ContextsPerPolicy=str(int(design[(design.label == "main") & (design.floor_in.astype(str) == "all")
                                                   & (design.policy == "min_carbon") & (design.selector == "capped")]
                                            .contexts.iloc[0])))
    for pol, tag in (("min_carbon", "MC"), ("min_size", "MS")):
        for sel, stag in sel_tags:
            macros[f"Agree{tag}{stag}"] = pct(agree("main", pol, sel))
            macros[f"AgreeCTwo{tag}{stag}"] = pct(agree("main", pol, sel, col="agreement_c2"))
            macros[f"Regret{tag}{stag}"] = f"{100 * agree('main', pol, sel, col='mean_regret'):.2f}"
            macros[f"BigRegret{tag}{stag}"] = pct(agree("main", pol, sel, col="share_regret_gt_5pct"))
            macros[f"Strict{tag}{stag}"] = pct(agree("strict_pcr_5_5", pol, sel))
            st = stack[(stack.policy == pol) & (stack.selector == sel)]
            macros[f"Stack{tag}{stag}"] = f"{100 * st.mean_stack_excess.mean():.2f}"
        for floor, ftag in ((12.0, "Twelve"), (20.0, "Twenty")):
            for sel, stag in sel_tags:
                macros[f"Agree{tag}{ftag}{stag}"] = pct(agree("main", pol, sel, floor=floor))
    for lab, tag in (("slenderness_always", "Slender"), ("storey_10ft", "TenFt"), ("size_step_0.5in", "Fine"),
                     ("rebar_0.5", "RebarLow"), ("rebar_2.0", "RebarHigh"), ("tie_break_submitted", "TieRow"),
                     ("column_admissible", "Adm"), ("window_500psi", "WinFive"), ("window_2000psi", "WinTwo")):
        for sel, stag in sel_tags:
            macros[f"Sens{tag}{stag}"] = pct(agree(lab, "min_carbon", sel))
    design_settings = ("slenderness_always", "storey_10ft", "size_step_0.5in", "rebar_0.5", "rebar_2.0",
                       "tie_break_submitted")
    for sel, stag in sel_tags:
        vals = [agree(lab, "min_carbon", sel) for lab in design_settings]
        macros[f"SensRange{stag}Lo"] = pct(min(vals))
        macros[f"SensRange{stag}Hi"] = pct(max(vals))
    order = [lab for lab in design_settings + ("strict_pcr_5_5", "column_admissible", "fcr_reinterpretation",
                                               "window_500psi", "window_2000psi")
             if agree(lab, "min_carbon", "capped_effective") > agree(lab, "min_carbon", "capped")
             > agree(lab, "min_carbon", "ratio") > agree(lab, "min_carbon", "fixed_volume")]
    macros["SensOrderHeld"] = str(len(order))
    macros.update(AdmCases=num(subsets.loc["column_admissible", "cases"]),
                  AdmRemoved=num(subsets.loc["column_admissible", "removed_cases"]),
                  WinFiveCases=num(subsets.loc["window_500psi", "cases"]),
                  WinTwoCases=num(subsets.loc["window_2000psi", "cases"]),
                  WinFiveCOne=pct(subsets.loc["window_500psi", "c1_rate"]),
                  WinFiveCTwo=pct(subsets.loc["window_500psi", "c2_rate"]),
                  WinTwoCOne=pct(subsets.loc["window_2000psi", "c1_rate"]),
                  WinTwoCTwo=pct(subsets.loc["window_2000psi", "c2_rate"]))
    gv = governing[governing.policy == "min_carbon"]
    for cat, ctag in (("nominal floor", "Floor"), ("slenderness", "Slender"), ("axial strength", "Axial")):
        rows = gv[gv.governing == cat].set_index("selector")
        macros[f"Gov{ctag}Share"] = pct(rows.share_of_case_contexts.iloc[0])
        for sel, stag in sel_tags:
            macros[f"Gov{ctag}{stag}"] = pct(rows.loc[sel, "agreement"])
        macros[f"Gov{ctag}SameRatioCapped"] = pct(rows.loc["capped", "same_gwp_as_ratio"])
        macros[f"Gov{ctag}SameRatioEffective"] = pct(rows.loc["capped_effective", "same_gwp_as_ratio"])
    dmc = diffs[diffs.policy == "min_carbon"].set_index("comparison")
    for comp, tag in (("capped - ratio", "CappedRatio"), ("capped_effective - ratio", "EffectiveRatio"),
                      ("capped - fixed_volume", "CappedFixed"), ("capped_effective - capped", "EffectiveCapped")):
        macros[f"Diff{tag}"] = f"{100 * dmc.loc[comp, 'difference']:.1f}"
        macros[f"Diff{tag}Lo"] = f"{100 * dmc.loc[comp, 'lo']:.1f}"
        macros[f"Diff{tag}Hi"] = f"{100 * dmc.loc[comp, 'hi']:.1f}"
    macros["CasesCappedGeRatio"] = pct(dmc.loc["capped - ratio", "share_cases_a_ge_b"])
    mb = bands[(bands.label == "main") & (bands.policy == "min_carbon")]

    def band_val(floor, band, sel, col="agreement"):
        return float(mb[(mb.floor_in == floor) & (mb.band == band) & (mb.selector == sel)][col].iloc[0])
    macros.update(BandTopTwelveFixed=pct(band_val(12.0, "1-2", "fixed_volume")),
                  BandTopTwelveRatio=pct(band_val(12.0, "1-2", "ratio")),
                  BandTopTwelveCapped=pct(band_val(12.0, "1-2", "capped")),
                  BandTopTwelveEffective=pct(band_val(12.0, "1-2", "capped_effective")),
                  BandLowTwelveFixed=pct(band_val(12.0, "16-20", "fixed_volume")),
                  BandLowTwelveRatio=pct(band_val(12.0, "16-20", "ratio")),
                  BandLowTwelveCapped=pct(band_val(12.0, "16-20", "capped")),
                  BandMidTwelveCapped=pct(band_val(12.0, "3-5", "capped")),
                  BandMidTwelveEffective=pct(band_val(12.0, "3-5", "capped_effective")),
                  BandMidTwelveFixed=pct(band_val(12.0, "3-5", "fixed_volume")),
                  OptimumAtFloorTwelve=pct(agree("main", "min_carbon", "capped", floor=12.0,
                                                 col="optimum_at_nominal_floor")),
                  OptimumAtFloorTwenty=pct(agree("main", "min_carbon", "capped", floor=20.0,
                                                 col="optimum_at_nominal_floor")))
    # Robustness.
    ind = mc[mc.model == "independent"]
    for sig, tag in ((0.025, "Low"), (0.05, "Mid"), (0.10, "High")):
        p = ind[np.isclose(ind.sigma, sig)]
        macros[f"Stable{tag}Fixed"] = pct(p.stable_fixed_volume.median())
        macros[f"Stable{tag}Ratio"] = pct(p.stable_ratio.median())
        macros[f"Stable{tag}Capped"] = pct(p.stable_capped_16in_1.median())
        macros[f"MCCOne{tag}"] = pct(p.c1.median())
        macros[f"MCCTwo{tag}"] = pct(p.c2.median())
    st = mc[mc.model == "structured"]
    macros.update(StableStructCapped=pct(st.stable_capped_16in_1.median()), StableStructRatio=pct(st.stable_ratio.median()),
                  StableStructFixed=pct(st.stable_fixed_volume.median()),
                  DominanceMin=pct(dominance[["capped_beats_ratio", "capped_beats_fixed"]].min().min(), 0),
                  MCReplicates=num(mc.replicate.max() + 1), MCDesignReplicates=num(design_rob.replicate.max()))
    # Strict subset and ladder.
    lad = ladder.set_index("scope")
    macros.update(StrictCases=num(strict["strict_cases"]), StrictCurrentCases=num(strict["strict_current_cases"]),
                  StrictSuppliers=str(strict["strict_suppliers"]),
                  StrictShare=pct(strict["candidate_share_passing"]["all"]),
                  StrictSOne=pct(strict["candidate_share_passing"]["S1_datasets"]),
                  StrictSOneFail=pct(1 - strict["candidate_share_passing"]["S1_datasets"]),
                  StrictSTwo=pct(strict["candidate_share_passing"]["S2_constituents"]),
                  StrictSThree=pct(strict["candidate_share_passing"]["S3_A3"]),
                  StrictSThreeFail=pct(1 - strict["candidate_share_passing"]["S3_A3"]),
                  StrictSFour=pct(strict["candidate_share_passing"]["S4_labels"]),
                  StrictCOne=pct(lad.loc["strict_pcr_5_5", "c1_rate"]), StrictCTwo=pct(lad.loc["strict_pcr_5_5", "c2_rate"]),
                  StrictCThreeMax=pct(lad.loc["strict_pcr_5_5", "c3_grid_max"]),
                  PlantCOne=pct(lad.loc["plant_scope", "c1_rate"]), ComponentCOne=pct(lad.loc["component_scope", "c1_rate"]),
                  PlantCTwo=pct(lad.loc["plant_scope", "c2_rate"]), ComponentCTwo=pct(lad.loc["component_scope", "c2_rate"]),
                  CurrentCOne=pct(lad.loc["strict_pcr_5_5_current", "c1_rate"]),
                  CurrentCTwo=pct(lad.loc["strict_pcr_5_5_current", "c2_rate"]))
    macros.update(BenchmarkMaxDiffR=f"{bench.absolute_difference.max():.3f}")

    def levels(at, ratio):
        x = demands[demands.tributary_ft2 == at]
        return float(np.interp(PU_REF_KIP * ratio, x.pu_kip, x.levels_supported))
    macros.update(PuRefKN=num(PU_REF_KIP * KIP_TO_KN), PuRef=num(PU_REF_KIP),
                  GridLevelsMin=f"{levels(900.0, 0.75):.1f}", GridLevelsMax=f"{levels(400.0, 1.25):.1f}",
                  GridRefLevelsLo=f"{levels(900.0, 1.0):.1f}", GridRefLevelsHi=f"{levels(400.0, 1.0):.1f}",
                  ExemptSizeMM=num(cd.MAIN.exempt_size() * IN_TO_MM), ExemptSize=f"{cd.MAIN.exempt_size():.1f}",
                  ClearHeightM=f"{cd.MAIN.lu_in * IN_TO_MM / 1000:.2f}", StoreyM=f"{cd.MAIN.storey_in * IN_TO_MM / 1000:.2f}",
                  ClearHeight=f"{cd.MAIN.lu_in:.0f}", StoreyFt=f"{cd.MAIN.storey_in / 12:.0f}",
                  RebarFactor=num(cd.REBAR_KG_PER_KG * 1000.0),
                  FixedOptimalShare=pct(agree("main", "min_carbon", "capped", col="fixed_volume_is_optimal")),
                  OptimumAtFloorAll=pct(agree("main", "min_carbon", "capped", col="optimum_at_nominal_floor")))
    lines = ["% Generated by r1_make_outputs.py from derived files. Do not edit manually."]
    for k, v in macros.items():
        if not k.isalpha():
            raise ValueError(f"Invalid macro name {k}")
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (M / "numbers_r1.tex").write_text("\n".join(lines) + "\n", encoding="ascii")
    (D / "manuscript_numbers_r1.json").write_text(json.dumps(macros, indent=2) + "\n", encoding="utf-8")

    end = r"\\"
    t = [r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.19\textwidth}>{\raggedright\arraybackslash}p{0.31\textwidth}rrr}",
         r"\toprule",
         r"Comparison & Question & Cases & Rate (\%) & 95\% supplier interval " + end, r"\midrule",
         f"C1 candidate set & Does a 6.9 MPa (1000 psi) window contain a lower declared $g$? & {macros['COneCases']} & "
         f"{macros['COneRate']} & {macros['COneLo']}--{macros['COneHi']} {end}",
         f"C2 objective & Does $g/f$ choose a higher $g$ than fixed volume? & {macros['CTwoCases']} & "
         f"{macros['CTwoRate']} & {macros['CTwoLo']}--{macros['CTwoHi']} {end}",
         f"C3 quantity constraint & Does the cap displace $g/f$ among the C2 cases? & 0--{macros['CTwoCases']}"
         f" & {macros['CThreeMin']}--{macros['CThreeMax']} & scenario-dependent {end}",
         f"C3$'$ (fixed-volume displacement) & Does the cap displace fixed volume among the C2 cases? & 0--{macros['CTwoCases']} & "
         f"{macros['CThreePMin']}--{macros['CThreePMax']} & scenario-dependent {end}",
         r"\bottomrule", r"\end{tabular}"]
    (M / "tab_r1_comparisons.tex").write_text("\n".join(t) + "\n", encoding="ascii")

    rows = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
            r"Design policy & Selector & \multicolumn{2}{c}{Agreement (\%)} & Mean & Regret & Stack & Strict " + end,
            r" & & all & C2 & regret (\%) & $>$5\% (\%) & excess (\%) & subset (\%) " + end, r"\midrule"]
    for pol, ptxt in (("min_carbon", "Minimum carbon"), ("min_size", "Minimum size")):
        tag = "MC" if pol == "min_carbon" else "MS"
        for sel, stxt in (("fixed_volume", "Fixed volume $g$"), ("ratio", "Ratio $g/f$"),
                          ("capped", "Capped, nominal floor"), ("capped_effective", "Capped, governing minimum")):
            stag = dict(sel_tags)[sel]
            rows.append(f"{ptxt if sel == 'fixed_volume' else ''} & {stxt} & {pct(agree('main', pol, sel))} & "
                        f"{pct(agree('main', pol, sel, col='agreement_c2'))} & "
                        f"{100 * agree('main', pol, sel, col='mean_regret'):.2f} & "
                        f"{pct(agree('main', pol, sel, col='share_regret_gt_5pct'))} & "
                        f"{macros[f'Stack{tag}{stag}']} & {pct(agree('strict_pcr_5_5', pol, sel))} {end}")
        if pol == "min_carbon":
            rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_r1_design.tex").write_text("\n".join(rows) + "\n", encoding="ascii")

    prof = json.loads((D / "profile.json").read_text(encoding="utf-8"))
    cov = [r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.72\columnwidth}r}", r"\toprule",
           r"Evidence stage & Count " + end, r"\midrule",
           f"Raw compilation records & {prof['raw_rows']:,} {end}",
           f"After eligibility, duplicate and identity screening & {prof['analysis_rows']:,} {end}",
           f"28-day records & {prof['analysis_28_day_rows']:,} {end}",
           f"Records with one explicit strength token concordant with the strength field & {prof['explicit_psi_series_rows_28d']:,} {end}",
           f"Matched within-declaration cases (primary rule) & {len(dec):,} {end}",
           f"\\quad of which in the largest catalogue & {int(dec.company.eq(rc.NRM).sum()):,} {end}",
           f"Cases in the strict observable-data subset (recorded PCR fields) & {strict['strict_cases']:,} {end}",
           f"\\quad of which with a declaration current on 13 October 2026 & {strict['strict_current_cases']:,} {end}",
           f"Supplier labels in primary cases & {dec.company.nunique()} {end}",
           f"Designed column contexts per design policy & {macros['ContextsPerPolicy']} {end}",
           r"\bottomrule", r"\end{tabular}"]
    (M / "tab_r1_coverage.tex").write_text("\n".join(cov) + "\n", encoding="ascii")
    supplementary_tables(grid, design_all, demands, mc, design_rob, ladder, governing, stack, diffs, subsets)
    print(json.dumps(macros, indent=1))


def supplementary_tables(grid, design_all, demands, mc, design_rob, ladder, governing, stack, diffs, subsets) -> None:
    end = r"\\"
    sels = [("fixed_volume", "Fixed"), ("ratio", "Ratio"), ("capped", "Capped"), ("capped_effective", "Gov. min.")]
    t = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
         r"Width & Demand & $f^*$ & C3 & C3$'$ & \multicolumn{3}{c}{C2 cases resolved to} & Median C3 " + end,
         r"mm (in) & ratio & (MPa) & (\%) & (\%) & fixed vol. & interm. & ratio & excess (\%) " + end, r"\midrule"]
    for r in grid.sort_values(["minimum_width_in", "demand_ratio"]).itertuples():
        t.append(f"{r.minimum_width_in * IN_TO_MM:.0f} ({r.minimum_width_in:.0f}) & {r.demand_ratio:g} & "
                 f"{r.fstar_psi * PSI_TO_MPA:.1f} & {pct(r.c3_rate)} & {pct(r.c3p_rate)} & {r.c2_resolved_to_fixed_volume} & "
                 f"{r.c2_resolved_intermediate} & {r.c2_resolved_to_ratio} & {pct(r.c3_median_excess_affected)} {end}")
    t += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_si_grid.tex").write_text("\n".join(t) + "\n", encoding="ascii")
    t = [r"\begin{tabular}{rrrrrl}", r"\toprule",
         r"Bay m (ft) & Levels & $D$ (kN) & $L$ reduced (kN) & $P_u$ kN (kip) & Governing combination " + end, r"\midrule"]
    for r in demands[demands.levels_supported.isin([1, 2, 5, 10, 15, 20])].itertuples():
        bay = round(float(np.sqrt(r.tributary_ft2)))
        t.append(f"{bay * FT_TO_M:.1f} ({bay}) & {r.levels_supported} & {r.dead_kip * KIP_TO_KN:,.0f} & "
                 f"{r.live_kip * KIP_TO_KN:,.0f} & {r.pu_kip * KIP_TO_KN:,.0f} ({r.pu_kip:,.0f}) & "
                 f"${r.governing_combination.replace('Lr', 'L_r')}$ {end}")
    t += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_si_demands.tex").write_text("\n".join(t) + "\n", encoding="ascii")
    t = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"Policy & Floor & \multicolumn{4}{c}{Agreement (\%)} & \multicolumn{4}{c}{Mean regret (\%)} " + end,
         " & mm (in) & " + " & ".join(s for _, s in sels) + " & " + " & ".join(s for _, s in sels) + " " + end,
         r"\midrule"]
    main = design_all[design_all.label == "main"]
    for pol, ptxt in (("min_carbon", "Min. carbon"), ("min_size", "Min. size")):
        for floor in ("10.0", "12.0", "16.0", "20.0", "all"):
            p = main[(main.policy == pol) & (main.floor_in.astype(str) == floor)].set_index("selector")
            ftxt = f"{float(floor) * IN_TO_MM:.0f} ({float(floor):.0f})" if floor != "all" else "all"
            t.append(f"{ptxt if floor == '10.0' else ''} & {ftxt} & "
                     + " & ".join(pct(p.loc[s, "agreement"]) for s, _ in sels) + " & "
                     + " & ".join(f"{100 * p.loc[s, 'mean_regret']:.2f}" for s, _ in sels) + f" {end}")
    t += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_si_design_floors.tex").write_text("\n".join(t) + "\n", encoding="ascii")
    tot = design_all[(design_all.floor_in.astype(str) == "all") & (design_all.policy == "min_carbon")]
    t = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Setting & \multicolumn{4}{c}{Agreement (\%)} & \multicolumn{4}{c}{Mean regret (\%)} " + end,
         " & " + " & ".join(s for _, s in sels) + " & " + " & ".join(s for _, s in sels) + " " + end, r"\midrule"]
    for lab, txt in SENS_LABELS:
        p = tot[tot.label == lab].set_index("selector")
        if p.empty:
            continue
        t.append(f"{txt} & " + " & ".join(pct(p.loc[s, "agreement"]) for s, _ in sels) + " & "
                 + " & ".join(f"{100 * p.loc[s, 'mean_regret']:.2f}" for s, _ in sels) + f" {end}")
    t += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_si_sensitivity.tex").write_text("\n".join(t) + "\n", encoding="ascii")
    t = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Model & C1 (\%) & C2 (\%) & \multicolumn{3}{c}{Choices unchanged (\%)} & Capped beats ratio " + end,
         r" & median & median & fixed vol. & ratio & capped & (\% of replicates) " + end, r"\midrule"]
    pert = design_rob[design_rob.replicate > 0]
    wide = pert.pivot_table(index=["model", "sigma", "replicate", "floor_in"], columns="selector", values="agreement")
    beats = (wide.capped > wide.ratio).groupby(level=["model", "sigma"]).mean()
    for (model, sig), part in mc.groupby(["model", "sigma"]):
        name = f"independent, $\\sigma$ = {100 * sig:g}\\%" if model == "independent" else "structured (A1--A3 shared)"
        t.append(f"{name} & {pct(part.c1.median())} & {pct(part.c2.median())} & {pct(part.stable_fixed_volume.median())} & "
                 f"{pct(part.stable_ratio.median())} & {pct(part.stable_capped_16in_1.median())} & "
                 f"{pct(beats.loc[(model, sig)], 0)} {end}")
    t += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_si_montecarlo.tex").write_text("\n".join(t) + "\n", encoding="ascii")
    names = {"plant_scope": "Any mixture of the declaration context", "component_scope": "Same component list",
             "description_series_primary": "Primary rule (same description series)",
             "strict_pcr_5_5": "Strict observable-data subset", "strict_pcr_5_5_current": "Strict, declaration current"}
    t = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Candidate scope & Cases & Median candidates & C1 (\%) & C2 (\%) & C3 max (\%) " + end, r"\midrule"]
    for r in ladder.itertuples():
        t.append(f"{names[r.scope]} & {r.cases:,} & {r.candidates_per_case_median:g} & {pct(r.c1_rate)} & "
                 f"{pct(r.c2_rate)} & {pct(r.c3_grid_max)} {end}")
    for subset, txt in (("window_500psi", "Primary rule, 3.4 MPa (500 psi) window"),
                        ("window_2000psi", "Primary rule, 13.8 MPa (2000 psi) window"),
                        ("column_admissible", "Primary rule, column-admissible labels")):
        r = subsets.loc[subset]
        t.append(f"{txt} & {int(r.cases):,} & -- & {pct(r.c1_rate)} & {pct(r.c2_rate)} & -- {end}")
    t += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_si_ladder.tex").write_text("\n".join(t) + "\n", encoding="ascii")
    t = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Policy & Size of optimal & Share & \multicolumn{4}{c}{Agreement (\%)} & \multicolumn{2}{c}{Same GWP as ratio (\%)} " + end,
         r" & column set by & (\%) & " + " & ".join(s for _, s in sels) + r" & Capped & Gov. min. " + end, r"\midrule"]
    for pol, ptxt in (("min_carbon", "Min. carbon"), ("min_size", "Min. size")):
        for k, cat in enumerate(("nominal floor", "slenderness", "axial strength")):
            g = governing[(governing.policy == pol) & (governing.governing == cat)].set_index("selector")
            t.append(f"{ptxt if k == 0 else ''} & {cat} & {pct(g.share_of_case_contexts.iloc[0])} & "
                     + " & ".join(pct(g.loc[s, "agreement"]) for s, _ in sels) + " & "
                     + " & ".join(pct(g.loc[s, "same_gwp_as_ratio"]) for s in ("capped", "capped_effective"))
                     + f" {end}")
    t += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_si_governing.tex").write_text("\n".join(t) + "\n", encoding="ascii")
    labels = {"capped - ratio": "Capped minus ratio", "capped_effective - ratio": "Governing minimum minus ratio",
              "capped - fixed_volume": "Capped minus fixed volume",
              "capped_effective - capped": "Governing minimum minus capped",
              "ratio - fixed_volume": "Ratio minus fixed volume"}
    t = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Policy & Difference in agreement & Points & 95\% supplier interval & Cases with $\geq$ (\%) " + end, r"\midrule"]
    for pol, ptxt in (("min_carbon", "Min. carbon"), ("min_size", "Min. size")):
        for k, r in enumerate(diffs[diffs.policy == pol].itertuples()):
            t.append(f"{ptxt if k == 0 else ''} & {labels[r.comparison]} & {100 * r.difference:+.1f} & "
                     f"{100 * r.lo:+.1f} to {100 * r.hi:+.1f} & {pct(r.share_cases_a_ge_b)} {end}")
    t += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_si_bootstrap_diff.tex").write_text("\n".join(t) + "\n", encoding="ascii")
    t = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Policy & Selector & Mean stack excess (\%) & 95th percentile (\%) & Stacks at optimum (\%) " + end, r"\midrule"]
    for pol, ptxt in (("min_carbon", "Min. carbon"), ("min_size", "Min. size")):
        for k, (sel, stxt) in enumerate(sels):
            s = stack[(stack.policy == pol) & (stack.selector == sel)]
            t.append(f"{ptxt if k == 0 else ''} & {stxt} & {100 * s.mean_stack_excess.mean():.2f} & "
                     f"{100 * s.p95_stack_excess.mean():.2f} & {pct(s.share_stacks_optimal.mean())} {end}")
    t += [r"\bottomrule", r"\end{tabular}"]
    (M / "tab_si_stack.tex").write_text("\n".join(t) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
