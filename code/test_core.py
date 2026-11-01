"""Regression tests for the first-revision analyses (synthetic checks run without data)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import column_design as cd
import core as rc
from field_provenance import fc_from_required_average


def synthetic(n_cases: int = 400, width: int = 6, seed: int = 7) -> rc.Candidates:
    rng = np.random.default_rng(seed)
    strength = 2500.0 + 500.0 * rng.integers(0, 6, size=(n_cases, width))
    gwp = np.round(200.0 + 0.04 * strength + rng.normal(0, 30, size=(n_cases, width)))
    order = np.lexsort((np.arange(width)[None, :].repeat(n_cases, 0), gwp), axis=1)
    rows = np.arange(n_cases)[:, None]
    gwp, strength = gwp[rows, order], strength[rows, order]
    row_id = np.arange(n_cases * width).reshape(n_cases, width)[rows, order] + 2
    valid = np.ones_like(gwp, dtype=bool)
    cases = pd.DataFrame({"target_psi": strength.min(axis=1), "company": "x"})
    return rc.Candidates(cases=cases, row_id=row_id, gwp=gwp, strength=strength, valid=valid,
                         exact=np.isclose(strength, strength.min(axis=1, keepdims=True)))


@pytest.mark.parametrize("fstar", [1500.0, 3000.0, 4200.0, 5000.0, 6800.0, 20000.0])
def test_capped_selector_equals_explicit_quantity_minimum(fstar):
    c = synthetic()
    q = rc.select_capped(c, fstar)
    area = np.maximum(fstar / c.strength, 1.0)            # K/f vs A_min with K = f* A_min
    explicit = rc.select_objective(c, c.gwp * area)
    carbon = c.gwp * area
    assert np.allclose(c.take(q, carbon), c.take(explicit, carbon))


def test_capped_choice_lies_between_conventions_and_is_not_dominated():
    c = synthetic()
    fv, ratio = rc.select_fixed_volume(c), rc.select_ratio(c)
    for fstar in np.linspace(1000, 12000, 45):
        q = rc.select_capped(c, fstar)
        g_q, f_q = c.take(q, c.gwp), c.take(q, c.strength)
        assert np.all(g_q >= c.take(fv, c.gwp) - 1e-9)
        assert np.all(g_q <= c.take(ratio, c.gwp) + 1e-9)
        assert not np.any(c.valid & (c.strength >= f_q[:, None]) & (c.gwp < g_q[:, None] - 1e-9))
        agree = np.isclose(c.take(fv, c.gwp), c.take(ratio, c.gwp))
        assert np.allclose(g_q[agree], c.take(fv, c.gwp)[agree])


def test_limits_of_capped_selector_are_the_two_conventions():
    c = synthetic()
    assert np.allclose(c.take(rc.select_capped(c, 1e-6), c.gwp), c.take(rc.select_fixed_volume(c), c.gwp))
    hi = rc.select_capped(c, 1e9)
    lo = rc.select_ratio(c)
    assert np.allclose(c.take(hi, c.gwp) / c.take(hi, c.strength), c.take(lo, c.gwp) / c.take(lo, c.strength))


def test_section_engine_reproduces_structurepoint_example():
    y, area = np.array([2.5, 13.5]), np.array([4.0, 4.0])
    phi_p, phi_m, cap = cd.section_curve(5000.0, 16.0, y, area, n_points=4000)
    assert abs(cap - 797.7) < 0.1
    for p, m in [(622.3, 169.86), (421.9, 220.05), (270.9, 250.77), (171.6, 286.75), (0.0, 213.96)]:
        assert abs(cd.moment_capacity(phi_p, phi_m, np.array([p]))[0] / 12.0 - m) < 0.1


def test_demand_hand_calculation():
    d = cd.demands(cd.MAIN).set_index(["tributary_ft2", "levels_supported"])
    sw = 0.150 * (20 / 12) ** 2 * 13.0
    dead = 900 * (142.5 + 4 * 132.5) / 1000 + 5 * sw
    pu = 1.2 * dead + 1.6 * 0.4 * 4 * 65 * 900 / 1000 + 0.5 * 900 * 12 / 1000
    assert math.isclose(d.loc[(900.0, 5), "pu_kip"], pu, rel_tol=1e-12)


def test_min_size_design_is_minimal_and_feasible():
    dem = cd.demands(cd.MAIN)
    table = cd.feasibility_table([4000.0], dem, cd.MAIN)
    q = cd.designs(table, {4000.0: [300.0]}, cd.REBAR_KG_PER_M3)
    _, _, size = q[("min_size", 12.0, 4000.0)]
    feas = np.stack(table.feasible.to_numpy())
    for d in range(len(dem)):
        b = size[d, 0]
        assert feas[(table.b_in == b).to_numpy(), d].any()
        smaller = (table.b_in < b - 1e-9) & (table.b_in >= 12.0)
        assert not feas[smaller.to_numpy(), d].any()


def test_slenderness_exemption_size():
    assert math.isclose(cd.MAIN.exempt_size(), 147.0 / (0.3 * 34.0))
    assert math.isinf(cd.Setting(slenderness="always").exempt_size())


@pytest.mark.parametrize("fc", [2500.0, 2900.0, 3000.0, 4000.0, 5000.0, 5500.0, 7000.0])
def test_required_average_strength_inverse(fc):
    fcr = fc + 1000 if fc < 3000 else (fc + 1200 if fc <= 5000 else 1.10 * fc + 700)
    assert math.isclose(float(fc_from_required_average(np.array([fcr]))[0]), fc, rel_tol=1e-12)


def test_pareto_order_prefers_stronger_equal_gwp():
    c = rc.Candidates(cases=pd.DataFrame({"target_psi": [4000.0]}), row_id=np.array([[5, 9]]),
                      gwp=np.array([[300.0, 300.0]]), strength=np.array([[4000.0, 5000.0]]),
                      valid=np.array([[True, True]]), exact=np.array([[True, False]]))
    p = rc.pareto_order(c)
    assert p.take(rc.select_fixed_volume(p), p.strength)[0] == 5000.0
    assert c.take(rc.select_fixed_volume(c), c.strength)[0] == 4000.0


@pytest.mark.skipif(not (rc.DATA / "analysis_records.csv").exists(), reason="derived data not present")
def test_vectorised_selectors_reproduce_reference():
    checks = rc.verify_against_reference(rc.build_candidates())
    assert all(checks.values())


def test_governing_and_stack_bookkeeping():
    c = rc.pareto_order(synthetic(n_cases=30, width=4, seed=3))
    extra: dict = {}
    contexts = cd.evaluate(c, cd.MAIN, extra=extra)
    gov = cd.governing_table(extra)
    assert np.allclose(gov.groupby(["policy", "selector"]).share_of_case_contexts.sum(), 1.0)
    assert np.allclose(gov[gov.selector == "ratio"].same_gwp_as_ratio, 1.0)
    for (policy, sel), part in gov.groupby(["policy", "selector"]):
        overall = contexts[(contexts.policy == policy) & (contexts.selector == sel)].agreement.mean()
        weighted = (part.agreement * part.case_contexts).sum() / part.case_contexts.sum()
        assert math.isclose(weighted, overall, rel_tol=1e-9)
    stack = cd.stack_table(extra)
    assert (stack.mean_stack_excess >= -1e-12).all()
    assert len(stack) == (len(cd.POLICIES) * len(cd.FLOORS_IN) * len(cd.SELECTORS)
                          * cd.demands(cd.MAIN).tributary_ft2.nunique())


def test_paired_bootstrap_of_a_constant_difference(tmp_path, monkeypatch):
    import extended_checks as ec
    rng = np.random.default_rng(1)
    shares = pd.DataFrame({"company": rng.choice(list("abcdef"), 60)})
    base = rng.uniform(0.3, 0.8, 60)
    offsets = {"fixed_volume": -0.2, "ratio": 0.0, "capped": 0.1, "capped_effective": 0.15}
    for policy in cd.POLICIES:
        for floor in cd.FLOORS_IN:
            for sel, off in offsets.items():
                shares[f"{policy}|{floor:g}|{sel}"] = base + off
    shares.to_csv(tmp_path / "r1_design_case_agreement.csv", index=False)
    monkeypatch.setattr(rc, "OUT", tmp_path)
    out = ec.paired_bootstrap(n_boot=50).set_index(["policy", "comparison"])
    row = out.loc[("min_carbon", "capped - ratio")]
    assert all(math.isclose(row[k], 0.1) for k in ("difference", "lo", "hi"))
    assert row.share_cases_a_ge_b == 1.0
    assert math.isclose(out.loc[("min_size", "capped_effective - capped"), "difference"], 0.05)


def test_percentages_never_round_up_to_100():
    from make_outputs import pct
    assert pct(0.99996) == r"$>$99.9"
    assert pct(1.0) == "100.0" and pct(0.5) == "50.0"


@pytest.mark.skipif(not rc.RAW.exists() or not (rc.DATA / "analysis_records.csv").exists(),
                    reason="raw or derived data not present")
def test_column_admissible_subset_size():
    import extended_checks as ec
    keep = ec.admissible_mask(rc.load_cases())
    assert int((~keep).sum()) == 18
