"""Code-constrained column designs used to test EPD selectors.

The designs are produced by an ACI 318-19 procedure that does NOT use the capped
selector or its fixed-steel algebra.  For each candidate concrete (specified
strength f'c from the EPD) and each design context, a square tied column is
designed, its embodied carbon is computed from the candidate's declared GWP and
a rebar factor, and the minimum-carbon candidate is the design-based answer
against which the fixed-volume, ratio and capped selectors are scored.

Implemented provisions (US customary units: psi, in, kip):
  ACI 318-19 22.4.2.1-2   Pn,max = 0.80 Po (tied), Po = 0.85 f'c (Ag - Ast) + fy Ast
  ACI 318-19 22.2         strain compatibility, eps_cu = 0.003, rectangular stress block,
                          beta1 per Table 22.2.2.4.3; displaced concrete deducted
  ACI 318-19 21.2.2       phi = 0.65 (compression-controlled, ties) to 0.90 (eps_t >= eps_ty + 0.003)
  ACI 318-19 10.6.1.1     Ast >= 0.01 Ag (a 0.04 Ag practical ceiling is used for lap splices)
  ACI 318-19 10.7.3.1     at least four bars in a rectangular tied column
  ACI 318-19 25.2.3       clear spacing of column bars >= max(1.5 in, 1.5 db, 4/3 dagg), dagg = 0.75 in
  ACI 318-19 25.7.2.2     No. 3 ties for No. 10 and smaller bars, No. 4 ties otherwise
  ACI 318-19 Table 20.5.1.3.1 and IBC 2021 722.2.4: 1.5 in clear cover to ties and at least
                          2 in cover to longitudinal bars for 2-3 h fire ratings
  ACI 318-19 6.2.5.1(b)   braced columns: slenderness neglected if k lu / r <= 34 + 12 M1/M2 <= 40;
                          M1/M2 = 0 is assumed for gravity columns (limit 34), r = 0.3 h, k = 1.0
  ACI 318-19 6.6.4        otherwise the nonsway moment magnifier: M2,min = Pu (0.6 + 0.03 h), Cm = 1.0,
                          (EI)eff = 0.4 Ec Ig / (1 + beta_dns), Pc = pi^2 (EI)eff / (k lu)^2
  ACI 318-19 6.2.6        second-order moment <= 1.4 x first-order moment
  ACI 318-19 19.2.2.1     Ec = 57,000 sqrt(f'c)
A sensitivity setting ('always') applies the magnifier to every column (no 6.2.5.1 exemption).
Loads (ASCE 7-16): combinations 2.3.1 (1.4D; 1.2D + 1.6L + 0.5Lr; 1.2D + 1.6Lr + 0.5L),
live-load reduction 4.7 (K_LL = 4, limits 0.5/0.4), roof live-load reduction 4.8.
Geometry floors: IBC 2021 Table 722.2.4 (siliceous aggregate: 10 in for 2 h, 12 in for 3 h);
ACI 318-19 18.7.2.1 (12 in, special moment frames); 16 and 20 in correspond to the 20 db
joint-depth requirement of ACI 318-19 18.8.2 for No. 6 and No. 8 beam bars.

This is a computational design set, not an external validation against built projects.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import r1_core as rc

FY = 60000.0
ES = 29.0e6
EPS_CU = 0.003
EPS_TY = FY / ES
PHI_C, PHI_T = 0.65, 0.90
AXIAL_CAP = 0.80
RHO_MIN, RHO_MAX = 0.01, 0.04
COVER_TO_BAR_IN = 2.0          # >= 1.5 in clear to ties + tie, and >= 2 in (IBC 722.2.4) to main bars
AGG_IN = 0.75
K_EFF = 1.0
SLENDER_LIMIT = 34.0           # 34 + 12 M1/M2 with M1/M2 = 0
DELTA_MAX = 1.4
SIZE_MIN_IN, SIZE_MAX_IN = 10.0, 72.0
# Bar designation: (diameter in, area in^2), ASTM A615.
BARS = {6: (0.750, 0.44), 7: (0.875, 0.60), 8: (1.000, 0.79), 9: (1.128, 1.00),
        10: (1.270, 1.27), 11: (1.410, 1.56), 14: (1.693, 2.25), 18: (2.257, 4.00)}
BARS_PER_FACE = range(2, 12)   # 4 to 40 perimeter bars
# Embodied carbon of fabricated rebar, CRSI industry-wide EPD (2022), A1-A3: 854 kg CO2e/t.
REBAR_KG_PER_KG = 0.854
STEEL_DENSITY = 7850.0          # kg/m3
REBAR_KG_PER_M3 = REBAR_KG_PER_KG * STEEL_DENSITY

# Gravity loads (psf) and prototype geometry.
SLAB_IN = 9.0                                   # normalweight flat plate
FLOOR_DEAD_PSF = 150.0 * SLAB_IN / 12.0 + 20.0  # slab + superimposed dead load
ROOF_DEAD_PSF = FLOOR_DEAD_PSF + 10.0
FLOOR_LIVE_PSF = 50.0 + 15.0    # office + partition allowance (ASCE 7-16 4.3.1, 4.3.2)
ROOF_LIVE_PSF = 20.0
TRIBUTARY_FT2 = (400.0, 625.0, 900.0)  # 20, 25 and 30 ft bays
LEVELS = range(1, 21)                  # levels supported (roof + floors)
FLOORS_IN = (10.0, 12.0, 16.0, 20.0)
POLICIES = ("min_size", "min_carbon")
SELECTORS = ("fixed_volume", "ratio", "capped", "capped_effective")


@dataclass(frozen=True)
class Setting:
    label: str = "main"
    storey_in: float = 156.0           # 13 ft
    size_step_in: float = 2.0
    slenderness: str = "exempt"        # 'exempt' (ACI 6.2.5.1) or 'always' (magnify every column)
    rebar_factor: float = REBAR_KG_PER_M3

    @property
    def lu_in(self) -> float:
        return self.storey_in - SLAB_IN

    @property
    def column_sw_kip(self) -> float:
        return 0.150 * (20.0 / 12.0) ** 2 * self.storey_in / 12.0   # 20 in column allowance per storey

    def sizes(self) -> np.ndarray:
        return np.round(np.arange(SIZE_MIN_IN, SIZE_MAX_IN + 1e-9, self.size_step_in), 6)

    def exempt_size(self) -> float:
        """Smallest b with k lu / (0.3 b) <= 34 (slenderness may be neglected)."""
        return K_EFF * self.lu_in / (0.3 * SLENDER_LIMIT) if self.slenderness == "exempt" else np.inf


MAIN = Setting()


def beta1(fc: float) -> float:
    return float(np.clip(0.85 - 0.05 * (fc - 4000.0) / 1000.0, 0.65, 0.85))


@dataclass(frozen=True)
class Layout:
    bar: int
    per_face: int

    @property
    def n_bars(self) -> int:
        return 4 * (self.per_face - 1)

    @property
    def area(self) -> float:
        return self.n_bars * BARS[self.bar][1]

    @property
    def db(self) -> float:
        return BARS[self.bar][0]


def cover_to_bar_centre(lay: Layout) -> float:
    tie = 0.5 if lay.bar >= 11 else 0.375
    return max(COVER_TO_BAR_IN, 1.5 + tie) + lay.db / 2


def layouts_for(b: float) -> list[Layout]:
    out = []
    for bar, (db, _) in BARS.items():
        for per_face in BARS_PER_FACE:
            lay = Layout(bar, per_face)
            d1 = cover_to_bar_centre(lay)
            clear = (b - 2 * d1 - (per_face - 1) * db) / (per_face - 1)
            if clear + 1e-9 < max(1.5, 1.5 * db, 4.0 / 3.0 * AGG_IN):
                continue
            rho = lay.area / (b * b)
            if RHO_MIN - 1e-12 <= rho <= RHO_MAX + 1e-12:
                out.append(lay)
    return out


def layers(b: float, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    """Bar layers for bending about an axis parallel to one face."""
    d1 = cover_to_bar_centre(lay)
    y = d1 + np.arange(lay.per_face) * (b - 2 * d1) / (lay.per_face - 1)
    counts = np.full(lay.per_face, 2.0)
    counts[[0, -1]] = lay.per_face
    return y, counts * BARS[lay.bar][1]


def section_curve(fc: float, b: float, y: np.ndarray, area: np.ndarray, n_points: int = 600):
    """Design interaction curve (phi Pn, phi Mn) for a rectangular b x b section [kip, kip-in]."""
    h = b
    c = h * np.geomspace(0.02, 4.0, n_points)
    a = np.minimum(beta1(fc) * c, h)
    strain = EPS_CU * (c[:, None] - y[None, :]) / c[:, None]          # compression positive
    stress = np.clip(ES * strain, -FY, FY)
    displaced = np.where(y[None, :] <= a[:, None], 0.85 * fc, 0.0)
    forces = area[None, :] * (stress - displaced)
    cc = 0.85 * fc * b * a
    pn = (cc + forces.sum(axis=1)) / 1000.0
    mn = (cc * (h / 2 - a / 2) + (forces * (h / 2 - y[None, :])).sum(axis=1)) / 1000.0
    eps_t = EPS_CU * (y.max() - c) / c                                  # tension positive
    phi = np.clip(PHI_C + (PHI_T - PHI_C) * (eps_t - EPS_TY) / 0.003, PHI_C, PHI_T)
    po = (0.85 * fc * (b * b - area.sum()) + FY * area.sum()) / 1000.0
    return phi * pn, phi * mn, PHI_C * AXIAL_CAP * po


def moment_capacity(phi_p: np.ndarray, phi_m: np.ndarray, pu: np.ndarray) -> np.ndarray:
    """Largest phi Mn on the curve at phi Pn = Pu (vectorised over Pu); -inf if none."""
    diff = phi_p[:, None] - pu[None, :]
    cross = (np.sign(diff[:-1]) != np.sign(diff[1:])) | (diff[:-1] == 0)
    denom = diff[:-1] - diff[1:]
    t = np.where(cross, diff[:-1] / np.where(denom == 0, 1, denom), 0.0)
    m = phi_m[:-1, None] + t * (phi_m[1:, None] - phi_m[:-1, None])
    return np.where(cross, m, -np.inf).max(axis=0)


def magnifier(fc: float, b: float, pu: np.ndarray, beta_dns: np.ndarray, lu_in: float) -> np.ndarray:
    ec = 57000.0 * math.sqrt(fc)
    eiff = 0.4 * ec * b ** 4 / 12.0 / (1.0 + beta_dns)
    pc = math.pi ** 2 * eiff / (K_EFF * lu_in) ** 2 / 1000.0
    ratio = pu / (0.75 * pc)
    return np.where(ratio < 1.0, np.maximum(1.0 / np.maximum(1.0 - ratio, 1e-12), 1.0), np.inf)


def magnifier_size(pu, beta_dns, fc, lu_in: float) -> np.ndarray:
    """Smallest (continuous) b with delta <= 1.4 (ACI 318-19 6.2.6) for k = 1.0."""
    pc_req = np.asarray(pu) * 1000.0 / (0.75 * (1.0 - 1.0 / DELTA_MAX))     # lb
    ec = 57000.0 * np.sqrt(fc)
    b4 = pc_req * (K_EFF * lu_in) ** 2 * 12.0 * (1.0 + np.asarray(beta_dns)) / (math.pi ** 2 * 0.4 * ec)
    return b4 ** 0.25


def governing_minimum(floor: float, pu, beta_dns, fc, s: Setting) -> np.ndarray:
    """Minimum size that the selector is told about in the 'effective' variant.

    The larger of the nominal floor and the smallest module size that satisfies the
    slenderness provisions (exemption size or delta <= 1.4 at the lowest candidate
    strength), rounded up to the size module.  No candidate design is used.
    """
    step = s.size_step_in
    b_mag = np.ceil(magnifier_size(pu, beta_dns, fc, s.lu_in) / step - 1e-9) * step
    b_ex = np.ceil(s.exempt_size() / step - 1e-9) * step if np.isfinite(s.exempt_size()) else np.inf
    return np.maximum(floor, np.minimum(b_mag, b_ex))


def demands(s: Setting = MAIN) -> pd.DataFrame:
    rows = []
    for at in TRIBUTARY_FT2:
        r1 = 1.0 if at <= 200 else (1.2 - 0.001 * at if at < 600 else 0.6)
        lr_psf = max(12.0, ROOF_LIVE_PSF * r1)
        for n in LEVELS:
            floors = n - 1
            dead = at * (ROOF_DEAD_PSF + floors * FLOOR_DEAD_PSF) / 1000.0 + n * s.column_sw_kip
            live0 = at * floors * FLOOR_LIVE_PSF / 1000.0
            factor = 1.0
            if floors >= 1 and 4.0 * at * floors >= 400.0:
                factor = 0.25 + 15.0 / math.sqrt(4.0 * at * floors)
                factor = min(1.0, max(factor, 0.5 if floors == 1 else 0.4))
            live = live0 * factor
            roof = at * lr_psf / 1000.0
            combos = {"1.4D": 1.4 * dead, "1.2D+1.6L+0.5Lr": 1.2 * dead + 1.6 * live + 0.5 * roof,
                      "1.2D+1.6Lr+0.5L": 1.2 * dead + 1.6 * roof + 0.5 * live}
            gov = max(combos, key=combos.get)
            pu = combos[gov]
            rows.append({"tributary_ft2": at, "levels_supported": n, "dead_kip": dead, "live_kip": live,
                         "roof_live_kip": roof, "live_reduction_factor": factor, "pu_kip": pu,
                         "governing_combination": gov, "beta_dns": 1.0 if gov == "1.4D" else 1.2 * dead / pu})
    return pd.DataFrame(rows)


def feasibility_table(strengths, dem: pd.DataFrame, s: Setting = MAIN) -> pd.DataFrame:
    """Every (f'c, size, layout) section with the demands it satisfies."""
    pu = dem.pu_kip.to_numpy()
    beta_dns = dem.beta_dns.to_numpy()
    records = []
    for fc in strengths:
        for b in s.sizes():
            slender = (K_EFF * s.lu_in / (0.3 * b) > SLENDER_LIMIT) or s.slenderness == "always"
            if slender:
                delta = magnifier(fc, b, pu, beta_dns, s.lu_in)
                mc = delta * pu * (0.6 + 0.03 * b)
            for lay in layouts_for(b):
                y, area = layers(b, lay)
                phi_p, phi_m, cap = section_curve(fc, b, y, area)
                ok = pu <= cap + 1e-9
                if slender:
                    mcap = moment_capacity(phi_p, phi_m, pu)
                    ok &= (mc <= mcap + 1e-9) & (delta <= DELTA_MAX + 1e-12)
                records.append({"fc_psi": fc, "b_in": b, "bar": lay.bar, "n_bars": lay.n_bars,
                                "steel_in2": lay.area, "concrete_in2": b * b - lay.area, "feasible": ok})
    return pd.DataFrame(records)


def designs(table: pd.DataFrame, candidates_by_fc: dict, rebar_factor: float) -> dict:
    """Design quantities per (policy, floor, f'c): arrays (n_demand, n_gwp).

    min_size:   smallest b >= floor with a feasible layout, then least steel at that b.
    min_carbon: least g * Ac + rebar * Ast over every feasible (b >= floor, layout).
    """
    out = {}
    for floor in FLOORS_IN:
        for fc, gvals in candidates_by_fc.items():
            part = table[(table.fc_psi == fc) & (table.b_in >= floor - 1e-9)]
            feas = np.stack(part.feasible.to_numpy())               # (n_sections, nd)
            conc = part.concrete_in2.to_numpy()
            steel = part.steel_in2.to_numpy()
            size = part.b_in.to_numpy()
            if not feas.any(axis=0).all():
                raise ValueError(f"No feasible design for f'c={fc}, floor={floor}")
            key = size * 1e4 + steel
            idx = np.where(feas, key[:, None], np.inf).argmin(axis=0)
            out[("min_size", floor, fc)] = tuple(np.repeat(v[idx][:, None], len(gvals), axis=1)
                                                 for v in (conc, steel, size))
            g = np.asarray(gvals, dtype=float)
            jdx = np.empty((feas.shape[1], len(g)), dtype=int)
            for start in range(0, len(g), 16):                          # chunked to bound memory
                gg = g[start:start + 16]
                carbon = conc[:, None, None] * gg[None, None, :] + rebar_factor * steel[:, None, None]
                carbon = np.where(feas[:, :, None], carbon, np.inf)   # (ns, nd, chunk)
                jdx[:, start:start + 16] = carbon.argmin(axis=0)
            out[("min_carbon", floor, fc)] = (conc[jdx], steel[jdx], size[jdx])
    return out


def member_quantities(c: rc.Candidates, strength: np.ndarray, s: Setting = MAIN):
    """Concrete/steel areas and size for every (policy, floor) -> arrays (n, w, n_demand)."""
    dem = demands(s)
    fc_values = sorted({float(v) for v in strength[c.valid]})
    table = feasibility_table(fc_values, dem, s)
    pairs = {fc: sorted({float(v) for v in c.gwp[c.valid & np.isclose(strength, fc)]}) for fc in fc_values}
    quantities = designs(table, pairs, s.rebar_factor)
    n, w = c.gwp.shape
    result = {}
    for policy in POLICIES:
        for floor in FLOORS_IN:
            conc = np.full((n, w, len(dem)), np.nan)
            steel = np.full((n, w, len(dem)), np.nan)
            size = np.full((n, w, len(dem)), np.nan)
            for fc, gvals in pairs.items():
                qc, qs, qb = quantities[(policy, floor, fc)]
                gindex = {g: k for k, g in enumerate(gvals)}
                sel = np.argwhere(c.valid & np.isclose(strength, fc))
                cols = np.array([gindex[float(c.gwp[i, j])] for i, j in sel])
                conc[sel[:, 0], sel[:, 1], :] = qc[:, cols].T
                steel[sel[:, 0], sel[:, 1], :] = qs[:, cols].T
                size[sel[:, 0], sel[:, 1], :] = qb[:, cols].T
            result[(policy, floor)] = (conc, steel, size)
    return dem, table, result


def selector_picks(c: rc.Candidates, strength: np.ndarray, dem: pd.DataFrame, floor: float,
                   s: Setting = MAIN) -> dict:
    """Selector choices for one floor; capped selectors need the context (Pu, minimum size)."""
    picks = {"fixed_volume": np.repeat(rc.select_fixed_volume(c)[:, None], len(dem), axis=1),
             "ratio": np.repeat(rc.select_ratio(c, strength)[:, None], len(dem), axis=1),
             "capped": np.zeros((c.n, len(dem)), dtype=int),
             "capped_effective": np.zeros((c.n, len(dem)), dtype=int)}
    target = c.cases.target_psi.to_numpy(dtype=float)
    for d, dm in dem.iterrows():
        amin = floor ** 2
        fstar = rc.transition_strength(dm.pu_kip, amin, RHO_MIN * amin, floor_governed_ok=True)
        picks["capped"][:, d] = rc.select_capped(c, float(fstar), strength)
        b_eff = governing_minimum(floor, dm.pu_kip, dm.beta_dns, target, s)
        a_eff = b_eff ** 2
        fstar_eff = rc.transition_strength(dm.pu_kip, a_eff, RHO_MIN * a_eff, floor_governed_ok=True)
        picks["capped_effective"][:, d] = rc.select_capped(c, fstar_eff, strength)
    return picks


GOVERNING = ("nominal floor", "slenderness", "axial strength")


def evaluate(c: rc.Candidates, s: Setting = MAIN, strength: np.ndarray | None = None,
             per_case: dict | None = None, extra: dict | None = None) -> pd.DataFrame:
    """Score selectors against the design optimum in every (policy, floor, demand) context.

    If per_case is a dict, it receives the share of contexts in which each selector
    agrees with the design optimum, per scenario, keyed by (policy, floor, selector).
    If extra is a dict, it receives (i) agreement counts, and counts of choices with the
    same GWP as the ratio choice, by the constraint that sets the size of the optimal
    column -- the nominal floor, the slenderness provisions or the axial strength check --
    and (ii) member carbon summed over the 20 levels of each column stack, for the
    stack-level excess of every selector.
    """
    strength = c.strength if strength is None else strength
    dem, _, result = member_quantities(c, strength, s)
    fv = rc.select_fixed_volume(c)
    ratio = rc.select_ratio(c, strength)
    c2 = c.take(ratio, c.gwp) > c.take(fv, c.gwp) + 1e-9
    rows = []
    for (policy, floor), (conc, steel, size) in result.items():
        picks = selector_picks(c, strength, dem, floor, s)
        for d, dm in dem.iterrows():
            carbon = c.gwp * conc[:, :, d] + s.rebar_factor * steel[:, :, d]
            best = rc.select_objective(c, np.where(c.valid, carbon, np.inf))
            c_best = c.take(best, carbon)
            b_best = c.take(best, size[:, :, d])
            fv_optimal = c.take(fv, carbon) <= c_best * (1 + 1e-9)
            if extra is not None:
                # Size set by the floor, by the slenderness provisions (exemption size or
                # delta <= 1.4 at the optimum's own strength) or, above both, by the axial check.
                b_gov = governing_minimum(floor, dm.pu_kip, dm.beta_dns, c.take(best, strength), s)
                category = np.where(b_best <= floor + 1e-9, GOVERNING[0],
                                    np.where(b_best <= b_gov + 1e-9, GOVERNING[1], GOVERNING[2]))
            for name in SELECTORS:
                pick_carbon = c.take(picks[name][:, d], carbon)
                regret = pick_carbon / c_best - 1.0
                agree = regret <= 1e-9
                if per_case is not None:
                    key = (policy, floor, name)
                    per_case[key] = per_case.get(key, 0.0) + agree / len(dem)
                if extra is not None:
                    gov = extra.setdefault("governing", {})
                    same_ratio = c.take(picks[name][:, d], c.gwp) == c.take(picks["ratio"][:, d], c.gwp)
                    for cat in GOVERNING:
                        m = category == cat
                        acc = gov.setdefault((policy, cat, name), [0, 0, 0])
                        acc[0] += int(m.sum())
                        acc[1] += int((agree & m).sum())
                        acc[2] += int((same_ratio & m).sum())
                    stack = extra.setdefault("stack", {})
                    key = (policy, floor, float(dm.tributary_ft2), name)
                    pick_sum, best_sum = stack.get(key, (np.zeros(c.n), np.zeros(c.n)))
                    stack[key] = (pick_sum + pick_carbon, best_sum + c_best)
                rows.append({"label": s.label, "storey_in": s.storey_in, "size_step_in": s.size_step_in,
                             "slenderness": s.slenderness, "policy": policy, "floor_in": floor,
                             "tributary_ft2": dm.tributary_ft2, "levels_supported": dm.levels_supported,
                             "pu_kip": dm.pu_kip, "selector": name, "cases": int(c.n),
                             "agreement": float(agree.mean()),
                             "agreement_c2": float(agree[c2].mean()) if c2.any() else np.nan,
                             "mean_regret": float(regret.mean()),
                             "p95_regret": float(np.quantile(regret, 0.95)),
                             "max_regret": float(regret.max()),
                             "share_regret_gt_5pct": float((regret > 0.05).mean()),
                             "mean_regret_c2": float(regret[c2].mean()) if c2.any() else np.nan,
                             "optimum_at_nominal_floor": float((b_best <= floor + 1e-9).mean()),
                             "fixed_volume_is_optimal": float(fv_optimal.mean())})
    return pd.DataFrame(rows)


def governing_table(extra: dict) -> pd.DataFrame:
    rows = []
    for (policy, cat, name), (n, k, same) in extra["governing"].items():
        rows.append({"policy": policy, "governing": cat, "selector": name, "case_contexts": n,
                     "agreement": k / n if n else np.nan,
                     "same_gwp_as_ratio": same / n if n else np.nan})
    out = pd.DataFrame(rows)
    totals = out.groupby(["policy", "selector"]).case_contexts.transform("sum")
    return out.assign(share_of_case_contexts=out.case_contexts / totals)


def stack_table(extra: dict) -> pd.DataFrame:
    rows = []
    for (policy, floor, bay, name), (pick_sum, best_sum) in extra["stack"].items():
        excess = pick_sum / best_sum - 1.0
        rows.append({"policy": policy, "floor_in": floor, "tributary_ft2": bay, "selector": name,
                     "mean_stack_excess": float(excess.mean()), "p95_stack_excess": float(np.quantile(excess, 0.95)),
                     "share_stacks_optimal": float((excess <= 1e-9).mean())})
    return pd.DataFrame(rows)


def aggregate(contexts: pd.DataFrame) -> pd.DataFrame:
    """Equal weight on every (demand, floor) context inside a policy."""
    metrics = dict(contexts=("agreement", "size"), agreement=("agreement", "mean"),
                   agreement_c2=("agreement_c2", "mean"), mean_regret=("mean_regret", "mean"),
                   mean_regret_c2=("mean_regret_c2", "mean"), max_regret=("max_regret", "max"),
                   share_regret_gt_5pct=("share_regret_gt_5pct", "mean"),
                   optimum_at_nominal_floor=("optimum_at_nominal_floor", "mean"),
                   fixed_volume_is_optimal=("fixed_volume_is_optimal", "mean"))
    keys = ["label", "policy", "selector"]
    by_floor = contexts.groupby(keys + ["floor_in"]).agg(**metrics).reset_index()
    total = contexts.groupby(keys).agg(**metrics).reset_index().assign(floor_in="all")
    return pd.concat([by_floor, total], ignore_index=True)


def band(levels: pd.Series) -> pd.Series:
    return pd.cut(levels, [0, 2, 5, 10, 15, 20], labels=["1-2", "3-5", "6-10", "11-15", "16-20"])


SENSITIVITIES = (
    Setting(label="slenderness_always", slenderness="always"),
    Setting(label="storey_10ft", storey_in=120.0),
    Setting(label="size_step_0.5in", size_step_in=0.5),
    Setting(label="rebar_0.5", rebar_factor=0.5 * STEEL_DENSITY),
    Setting(label="rebar_2.0", rebar_factor=2.0 * STEEL_DENSITY),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="main setting only")
    args = parser.parse_args()
    c = rc.build_candidates()
    rc.verify_against_submission(c)
    demands(MAIN).to_csv(rc.OUT / "r1_design_demands.csv", index=False)
    # Main results use a weak-dominance tie-break for every selector (and for the design
    # optimum): among equal objective values the stronger of two equal-GWP mixes is taken.
    cp = rc.pareto_order(c)
    per_case: dict = {}
    extra: dict = {}
    contexts = [evaluate(cp, MAIN, per_case=per_case, extra=extra)]
    pc = c.cases[["menu_id", "target_psi", "company", "plant_issue_id"]].copy()
    for (policy, floor, name), share in per_case.items():
        pc[f"{policy}|{floor:g}|{name}"] = share
    pc.to_csv(rc.OUT / "r1_design_case_agreement.csv", index=False)
    governing_table(extra).to_csv(rc.OUT / "r1_design_governing.csv", index=False)
    stack_table(extra).to_csv(rc.OUT / "r1_design_stack.csv", index=False)
    if not args.quick:
        contexts.append(evaluate(c, replace(MAIN, label="tie_break_submitted")))
        for setting in SENSITIVITIES:
            contexts.append(evaluate(cp, setting))
    contexts = pd.concat(contexts, ignore_index=True)
    contexts.to_csv(rc.OUT / "r1_design_validation_contexts.csv", index=False)
    agg = aggregate(contexts)
    agg.to_csv(rc.OUT / "r1_design_validation_summary.csv", index=False)
    banded = contexts.assign(band=band(contexts.levels_supported)).groupby(
        ["label", "policy", "floor_in", "band", "selector"], observed=True).agg(
        agreement=("agreement", "mean"), mean_regret=("mean_regret", "mean"),
        optimum_at_nominal_floor=("optimum_at_nominal_floor", "mean"),
        fixed_volume_is_optimal=("fixed_volume_is_optimal", "mean")).reset_index()
    banded.to_csv(rc.OUT / "r1_design_validation_bands.csv", index=False)
    print(agg[agg.floor_in.eq("all")].to_string(index=False))


if __name__ == "__main__":
    main()
