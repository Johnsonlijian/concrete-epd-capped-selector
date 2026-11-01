"""Shared candidate sets and selectors for the first-revision (R1) analyses.

Every scenario is one primary description-series case (menu, target strength)
from ``choices_description_series_1000.csv``.  Candidate sets are rebuilt from
``analysis_records.csv`` with the same rules as
``quantity_model.evaluate_catalogues`` so that every new analysis keeps the
submitted 1,538-case denominator.  Candidates are stored in padded
(scenario x candidate) arrays sorted by (GWP, row_id), so "first index among
tied minima" reproduces the submitted ``best()`` tie-break exactly.

Selectors (all evaluated on the same candidate set W of a scenario):
  FV  fixed-volume selector, minimises g
  R   ordinary strength-normalised selector, minimises g / f
  Q   quantity-constrained selector, minimises g / min(f, f*)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("R1_DATA_DIR", ROOT / "derived"))
OUT = Path(os.environ.get("R1_OUT_DIR", ROOT / "derived"))
RAW_NAME = "Compiled_Concrete_EPD_Data_Version_4c_Final_Published.csv"
RAW = Path(os.environ.get("R1_RAW", next(
    (p for p in (ROOT / "raw" / RAW_NAME, ROOT.parent / "raw" / RAW_NAME) if p.exists()),
    ROOT.parent / "raw" / RAW_NAME)))
MENU_KEYS = ["company", "plant", "address", "issued", "operator", "source"]
WINDOW_PSI = 1000.0
NRM = "national ready mix"

# Tolerances used by the submitted scripts: common_cohort() used an absolute
# 1e-12 tie window for g/f; quantity_model.py used rtol 1e-12, atol 1e-10.
RATIO_TOL = dict(rtol=0.0, atol=1e-12)
QUANTITY_TOL = dict(rtol=1e-12, atol=1e-10)


@dataclass
class Candidates:
    """Padded candidate arrays; padding has valid == False and NaN values."""

    cases: pd.DataFrame
    row_id: np.ndarray
    gwp: np.ndarray
    strength: np.ndarray
    valid: np.ndarray
    exact: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.gwp.shape[0]

    def take(self, index: np.ndarray, values: np.ndarray) -> np.ndarray:
        return values[np.arange(self.n), index]

    def subset(self, mask: np.ndarray) -> "Candidates":
        """Keep scenarios with mask True (rows) – candidate columns unchanged."""
        return Candidates(
            cases=self.cases.loc[mask].reset_index(drop=True),
            row_id=self.row_id[mask], gwp=self.gwp[mask], strength=self.strength[mask],
            valid=self.valid[mask], exact=self.exact[mask],
            extra={k: v[mask] for k, v in self.extra.items()},
        )

    def restrict(self, keep: np.ndarray) -> "Candidates":
        """Drop individual candidates (keep False) while preserving sort order."""
        valid = self.valid & keep
        return Candidates(
            cases=self.cases, row_id=np.where(valid, self.row_id, -1),
            gwp=np.where(valid, self.gwp, np.nan), strength=np.where(valid, self.strength, np.nan),
            valid=valid, exact=self.exact & valid, extra=dict(self.extra),
        )


def load_records() -> pd.DataFrame:
    return pd.read_csv(DATA / "analysis_records.csv", keep_default_na=False, low_memory=False)


def load_cases() -> pd.DataFrame:
    return pd.read_csv(DATA / "choices_description_series_1000.csv", low_memory=False)


def build_candidates(records: pd.DataFrame | None = None, cases: pd.DataFrame | None = None,
                     window: float = WINDOW_PSI, scope: str = "series") -> Candidates:
    """Rebuild candidate sets for every primary case.

    scope = "series" reproduces the primary rule (same menu, 28-day, same
    component list and same description series); "component" drops the
    description-series condition and "plant" also drops the component
    condition.  Baselines and targets are always those of the primary case.
    """
    records = load_records() if records is None else records
    cases = load_cases() if cases is None else cases
    by_id = records.set_index("row_id")
    groups = {k: g for k, g in records.groupby(MENU_KEYS, sort=False)}
    pools = []
    for case in cases.itertuples():
        base = by_id.loc[case.baseline_id]
        group = groups[tuple(base[k] for k in MENU_KEYS)]
        mask = group.age_days.eq(28) & group.strength_psi.between(case.target_psi, case.target_psi + window)
        if scope in ("series", "component"):
            mask &= group.components.eq(base.components)
        if scope == "series":
            mask &= group.description_series.eq(base.description_series)
        pool = group.loc[mask].sort_values(["gwp", "row_id"], kind="stable")
        pools.append(pool)
    width = max(len(p) for p in pools)
    shape = (len(pools), width)
    row_id = np.full(shape, -1, dtype=np.int64)
    gwp = np.full(shape, np.nan)
    strength = np.full(shape, np.nan)
    for i, pool in enumerate(pools):
        k = len(pool)
        row_id[i, :k] = pool.row_id.to_numpy()
        gwp[i, :k] = pool.gwp.to_numpy(dtype=float)
        strength[i, :k] = pool.strength_psi.to_numpy(dtype=float)
    valid = row_id >= 0
    target = cases.target_psi.to_numpy(dtype=float)[:, None]
    exact = valid & np.isclose(strength, target)
    return Candidates(cases=cases.reset_index(drop=True), row_id=row_id, gwp=gwp,
                      strength=strength, valid=valid, exact=exact)


def attach_fields(c: Candidates, table: pd.DataFrame, fields, key: str = "row_id") -> Candidates:
    """Copy per-record fields (looked up by row_id) into padded candidate arrays."""
    lookup = table.drop_duplicates(key).set_index(key)
    flat = c.row_id.ravel()
    ok = flat >= 0
    for name in fields:
        values = np.full(flat.shape, np.nan if np.issubdtype(lookup[name].dtype, np.number) else None,
                         dtype=float if np.issubdtype(lookup[name].dtype, np.number) else object)
        values[ok] = lookup.loc[flat[ok], name].to_numpy()
        c.extra[name] = values.reshape(c.row_id.shape)
    return c


def pareto_order(c: Candidates) -> Candidates:
    """Re-sort candidates by (GWP ascending, strength descending, row_id).

    Selectors take the first tied minimum, so this ordering makes every selector
    prefer the stronger of two candidates with equal objective value and equal
    GWP (weak dominance).  The submitted analyses used (GWP, row_id) order.
    """
    key_g = np.where(c.valid, c.gwp, np.inf)
    key_f = np.where(c.valid, -c.strength, np.inf)
    key_r = np.where(c.valid, c.row_id, np.iinfo(np.int64).max)
    order = np.lexsort((key_r, key_f, key_g), axis=1)
    rows = np.arange(c.n)[:, None]
    return Candidates(cases=c.cases, row_id=c.row_id[rows, order], gwp=c.gwp[rows, order],
                      strength=c.strength[rows, order], valid=c.valid[rows, order],
                      exact=c.exact[rows, order], extra={k: v[rows, order] for k, v in c.extra.items()})


def select(objective: np.ndarray, valid: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    """First candidate (in GWP, row_id order) within tolerance of the minimum."""
    obj = np.where(valid, objective, np.inf)
    best = obj.min(axis=1, keepdims=True)
    if not np.isfinite(best).all():
        raise ValueError("Every scenario needs at least one valid candidate")
    tied = valid & (np.abs(obj - best) <= atol + rtol * np.abs(best))
    return np.argmax(tied, axis=1)


def select_fixed_volume(c: Candidates, subset: np.ndarray | None = None) -> np.ndarray:
    valid = c.valid if subset is None else c.valid & subset
    # Candidates are sorted by (gwp, row_id); the first valid one is best().
    return np.argmax(valid, axis=1)


def select_ratio(c: Candidates, strength: np.ndarray | None = None) -> np.ndarray:
    f = c.strength if strength is None else strength
    return select(c.gwp / f, c.valid, **RATIO_TOL)


def select_capped(c: Candidates, fstar: np.ndarray | float, strength: np.ndarray | None = None) -> np.ndarray:
    f = c.strength if strength is None else strength
    fstar = np.broadcast_to(np.asarray(fstar, dtype=float).reshape(-1, 1) if np.ndim(fstar) else fstar,
                            f.shape)
    return select(c.gwp / np.minimum(f, fstar), c.valid, **QUANTITY_TOL)


def select_objective(c: Candidates, objective: np.ndarray) -> np.ndarray:
    return select(objective, c.valid, **QUANTITY_TOL)


# Fixed-steel axial section model of the submitted manuscript (US units).
PHI_COMPRESSION = 0.65
AXIAL_LIMIT = 0.80
FY_PSI = 60000.0


def k_constant(pu_kip: np.ndarray | float, steel_in2: np.ndarray | float) -> np.ndarray:
    """K in A_c,req = K / f for the fixed-steel axial model (psi*in^2)."""
    return (np.asarray(pu_kip) * 1000.0 / (PHI_COMPRESSION * AXIAL_LIMIT) - FY_PSI * np.asarray(steel_in2)) / 0.85


def transition_strength(pu_kip, min_gross_in2, steel_in2, floor_governed_ok: bool = False) -> np.ndarray:
    """f* = K / (A_min - A_s); positive K and A_min > A_s are required.

    With floor_governed_ok, a non-positive K (the fixed steel alone carries the
    demand, so the geometry floor governs at every strength) returns a vanishing
    f*, for which g / min(f, f*) ranks exactly like fixed volume.
    """
    k = k_constant(pu_kip, steel_in2)
    net = np.asarray(min_gross_in2) - np.asarray(steel_in2)
    if np.any(net <= 0):
        raise ValueError("Transition strength needs A_min > A_s")
    if floor_governed_ok:
        k = np.maximum(k, 1e-6)
    elif np.any(k <= 0):
        raise ValueError("Transition strength needs a positive concrete demand")
    return k / net


def model_area(strength, pu_kip, min_gross_in2, steel_in2) -> np.ndarray:
    """Net concrete area max(K/f, A_min - A_s) of the fixed-steel model."""
    k = k_constant(pu_kip, steel_in2)
    return np.maximum(np.asarray(k)[..., None] / strength if np.ndim(k) else k / strength,
                      (np.asarray(min_gross_in2) - np.asarray(steel_in2))[..., None]
                      if np.ndim(min_gross_in2) else min_gross_in2 - steel_in2)


def verify_against_submission(c: Candidates, data_dir: Path = DATA) -> dict:
    """Reproduce every selection reported in the submitted derived files."""
    cases = c.cases
    fv_exact = select_fixed_volume(c, c.exact)
    fv = select_fixed_volume(c)
    ratio = select_ratio(c)
    checks = {
        "baseline_id": bool((c.take(fv_exact, c.row_id) == cases.baseline_id.to_numpy()).all()),
        "candidate_id": bool((c.take(fv, c.row_id) == cases.candidate_id.to_numpy()).all()),
    }
    common = pd.read_csv(data_dir / "common_baseline_comparisons.csv")
    merged = cases[["menu_id", "target_psi"]].assign(ratio_row=c.take(ratio, c.row_id)).merge(
        common[["menu_id", "target_psi", "ratio_candidate_id"]], on=["menu_id", "target_psi"], validate="one_to_one")
    checks["ratio_candidate_id"] = bool((merged.ratio_row == merged.ratio_candidate_id).all())
    scen = pd.read_csv(data_dir / "quantity_model_scenarios.csv")
    ok = True
    for (width, demand), part in scen.groupby(["minimum_width_in", "demand_ratio"]):
        part = cases[["menu_id", "target_psi"]].merge(part, on=["menu_id", "target_psi"], validate="one_to_one")
        fstar = float(part.saturation_strength_psi.iloc[0])
        capped = c.take(select_capped(c, fstar), c.row_id)
        area = np.maximum(k_constant(part.demand_kip.iloc[0], 8.0) / c.strength, width ** 2 - 8.0)
        explicit = c.take(select_objective(c, c.gwp * area), c.row_id)
        ok &= bool((capped == part.capped_selected_id.to_numpy()).all())
        ok &= bool((explicit == part.quantity_selected_id.to_numpy()).all())
        ok &= bool((c.take(ratio, c.row_id) == part.ordinary_selected_id.to_numpy()).all())
    checks["quantity_model_scenarios"] = ok
    if not all(checks.values()):
        raise AssertionError(f"Vectorised selectors do not reproduce the submission: {checks}")
    return checks
