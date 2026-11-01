"""Bounded alias tests; the frozen-fixture regression is not PDF validation."""

import json
from pathlib import Path

import pandas as pd
import pytest
from reconcile_aliases import (
    SMYRNA_CLUSTER,
    allowlist,
    reconcile_aliases,
    verification_rows,
    write_audits,
)

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def original():
    return pd.read_csv(PROJECT / "derived/alias_input_records.csv", low_memory=False)


@pytest.fixture(scope="module")
def reconciled(original):
    return reconcile_aliases(original)


def test_allowlist_is_exactly_thirteen_blocks():
    entries = allowlist()
    assert len(entries) == len({r["block_id"] for r in entries}) == 13


def test_frozen_identity_and_conflict_counts(original, reconciled):
    frame, meta = reconciled
    counts = meta["summary"]
    assert len(original) == 39745
    assert len(frame) == 39412
    assert counts["cross_label_identities"] == 234
    assert counts["cross_label_rows"] == 468
    assert counts["quarantined_identities"] == 99
    assert counts["quarantined_rows"] == 198
    assert counts["components_conflict_identities"] == 99
    assert counts["wc_conflict_identities"] == 13
    assert counts["other_payload_conflict_identities"] == 0
    assert counts["reconciled_identities"] == 135
    assert counts["duplicate_rows_removed"] == 135
    assert counts["matched_blocks"] == 13


def test_quivas_pair_is_reconciled_with_both_sources(reconciled):
    frame, meta = reconciled
    assert {34797, 34854}.issubset(set(frame.row_id))
    assert not {44021, 44085}.intersection(frame.row_id)
    rows = frame.set_index("row_id").loc[[34797, 34854]]
    for col in ("company", "plant", "address", "issued", "operator", "source"):
        assert rows[col].nunique() == 1
    assert rows.company.iloc[0] == SMYRNA_CLUSTER
    assert rows.source.iloc[0] == "urn:epd-alias:quivas:2023-03-02"
    audit = {r["row_id"]: r for r in meta["record_audit"]}
    assert audit[44021]["representative_row_id"] == 34797
    assert audit[44085]["representative_row_id"] == 34854
    assert audit[44021]["original_source"] != audit[34797]["original_source"]
    assert audit[44021]["action"] == "duplicate_removed"


def test_input_and_unaffected_rows_unchanged(original, reconciled):
    before = original.copy(deep=True)
    result, meta = reconcile_aliases(original)
    pd.testing.assert_frame_equal(before, original)
    affected = {r["row_id"] for r in meta["record_audit"]}
    expected = original[~original.row_id.isin(affected)]
    actual = result[~result.row_id.isin(affected)]
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(result, reconciled[0])
    assert result.columns.tolist() == original.columns.tolist()


def test_nonallowlisted_date_and_company_are_untouched(original):
    pair = original[original.row_id.isin([34797, 44021])].copy()
    pair["issued"] = "3/2/2024"
    result, meta = reconcile_aliases(pair)
    pd.testing.assert_frame_equal(pair, result)
    assert not meta["record_audit"]
    pair["issued"] = "3/2/2023"
    pair["company"] = "unrelated supplier at the same address"
    result, meta = reconcile_aliases(pair)
    pd.testing.assert_frame_equal(pair, result)
    assert not meta["record_audit"]


def test_unknown_plant_at_allowlisted_address_is_not_merged(original):
    pair = original[original.row_id.isin([34797, 44021])].copy()
    pair["plant"] = "different colocated plant"
    result, _ = reconcile_aliases(pair)
    pd.testing.assert_frame_equal(pair, result)


def test_any_new_analytical_conflict_is_quarantined(original):
    pair = original[original.row_id.isin([34797, 44021])].copy()
    pair.loc[pair.row_id.eq(44021), "gwp"] = 318
    result, meta = reconcile_aliases(pair)
    assert result.empty
    assert meta["summary"]["other_payload_conflict_identities"] == 1
    assert meta["identity_audit"][0]["conflict_fields"] == ["gwp"]


def test_order_independence_and_idempotence(original, reconciled):
    result, _ = reconcile_aliases(original.sample(frac=1, random_state=20260909))
    pd.testing.assert_frame_equal(
        result.sort_values("row_id"), reconciled[0].sort_values("row_id")
    )
    again, _ = reconcile_aliases(reconciled[0])
    pd.testing.assert_frame_equal(again, reconciled[0])


def test_bad_row_identity_fails_closed(original):
    pair = original.iloc[:2].copy()
    pair["row_id"] = 10
    with pytest.raises(ValueError, match="unique"):
        reconcile_aliases(pair)


def test_single_label_collision_is_not_silently_deduplicated(original):
    pair = original[original.row_id.isin([34797, 44021])].copy()
    pair["company"] = pair.company.iloc[0]
    result, meta = reconcile_aliases(pair)
    assert set(result.row_id) == {34797, 44021}
    assert meta["summary"]["cross_label_identities"] == 0
    assert meta["summary"]["duplicate_rows_removed"] == 0


def test_audit_exports_preserve_original_sources(reconciled, tmp_path):
    _, meta = reconciled
    write_audits(meta, tmp_path)
    assert all(path.name.startswith("alias_") for path in tmp_path.iterdir())
    report = json.loads((tmp_path / "alias_summary.json").read_text())
    assert report["summary"]["quarantined_rows"] == 198
    audit = pd.read_csv(tmp_path / "alias_record_audit.csv").set_index("row_id")
    assert audit.loc[44021, "representative_row_id"] == 34797
    assert "smyrna-ready-mix" in audit.loc[44021, "original_source"]
    assert "epd-martin-marietta-lookup" in audit.loc[34797, "original_source"]


def test_metadata_is_json_serializable_and_accounts_for_every_affected_row(reconciled):
    frame, meta = reconciled
    json.dumps(meta, allow_nan=False)
    audit = pd.DataFrame(meta["record_audit"])
    assert audit.row_id.is_unique
    assert len(audit) == meta["summary"]["affected_input_rows"]
    removed = audit[audit.action.ne("retained")]
    assert len(removed) == 333
    assert not set(removed.row_id).intersection(frame.row_id)
    assert "NOT verified legal suppliers" in meta["source_cluster_interpretation"]


def test_recomputed_primary_and_other_supplier_results(reconciled):
    frame, meta = reconciled
    rows = verification_rows(frame, meta, PROJECT / "code/analyze_epd_choices.py")
    all_cases, other = rows
    assert (
        all_cases["cases"],
        all_cases["objective_positive"],
        all_cases["window_positive"],
    ) == (1538, 995, 357)
    assert (other["cases"], other["objective_positive"], other["window_positive"]) == (
        789,
        684,
        34,
    )
    assert all_cases["source_label_clusters"] == 28
    assert other["source_label_clusters"] == 27
    assert all_cases["median_volume_threshold_affected"] == pytest.approx(
        0.08732394366197182
    )
    assert all_cases["window_robust_5pct"] == 198
    assert all_cases["window_robust_10pct"] == 92
