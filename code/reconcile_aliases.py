"""Conservative, allowlisted EPD extraction-alias reconciliation.

Pure API: ``result, metadata = reconcile_aliases(screened_frame)``. Pass the
screened frame AFTER its within-label screening and local deduplication, BEFORE
choices(). No files are written by that function and the input is not mutated.
Only the 13 listed blocks can change. Canonical source URNs are analytical
identifiers, NOT downloadable sources; original URLs remain in record_audit.
Use metadata['source_cluster_map'] for source-cluster weighting, never as a
verified legal-supplier registry. Export audits explicitly with write_audits().
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

POLICY_VERSION = "allowlisted-alias-quarantine-v1"
NRMCA = "national ready mixed concrete association"
ASTM = "astm international"
MARTIN = "martin marietta - smyrna ready mix concrete, llc."
SMYRNA = "smyrna ready mix"
US_CONCRETE = "us concrete mix / smyrna ready mix"
AMRIZE = "amrize"
HOLCIM = "holcim - aggregate industries"
SMYRNA_CLUSTER = "source-cluster:smyrna"
AMRIZE_CLUSTER = "source-cluster:amrize-holcim"
SOURCE_CLUSTER_MAP = {
    MARTIN: SMYRNA_CLUSTER,
    SMYRNA: SMYRNA_CLUSTER,
    US_CONCRETE: SMYRNA_CLUSTER,
    AMRIZE: AMRIZE_CLUSTER,
    HOLCIM: AMRIZE_CLUSTER,
    SMYRNA_CLUSTER: SMYRNA_CLUSTER,
    AMRIZE_CLUSTER: AMRIZE_CLUSTER,
}
MENU_KEYS = ["company", "plant", "address", "issued", "operator", "source"]
PAYLOAD_FIELDS = [
    "strength_psi",
    "gwp",
    "a1",
    "a2",
    "a3",
    "description",
    "components",
    "wc_raw",
    "expires",
]
REQUIRED = ["row_id", *MENU_KEYS, "mix", "age_days", "unit", *PAYLOAD_FIELDS]


@dataclass(frozen=True)
class Site:
    key: str
    address: str
    address_aliases: tuple[str, ...]
    company_plants: tuple[tuple[str, str], ...]
    plant: str
    cluster: str
    operator: str
    dates: tuple[str, ...]


# Exact local aliases, not an address normalizer or a global ownership crosswalk.
SITES = (
    Site(
        "quivas",
        "1145 quivas st denver co 80204",
        ("1145 quivas st. denver co 80204", "1145 quivas street, denver, co, 80204"),
        ((MARTIN, "quivas plant"), (SMYRNA, "quivas plant")),
        "quivas plant",
        SMYRNA_CLUSTER,
        NRMCA,
        (
            "2021-10-02",
            "2021-10-03",
            "2023-03-02",
            "2023-03-16",
            "2023-03-23",
            "2023-04-28",
        ),
    ),
    Site(
        "chambers",
        "2650 chambers rd aurora co 80011",
        ("2650 chambers rd. aurora co 80011", "2650 chambers rd., aurora, co 80011"),
        ((MARTIN, "chambers plant"), (SMYRNA, "chambers plant")),
        "chambers plant",
        SMYRNA_CLUSTER,
        NRMCA,
        ("2021-10-02", "2021-10-03", "2023-03-02"),
    ),
    Site(
        "fenton",
        "5400 fenton st arvada co 80002",
        ("5400 fenton st. arvada co 80002",),
        ((MARTIN, "arvada plant"), (US_CONCRETE, "golden plant")),
        "fenton source-alias site",
        SMYRNA_CLUSTER,
        NRMCA,
        ("2021-10-02", "2021-10-03"),
    ),
    Site(
        "main-costilla",
        "444 e costilla st colorado springs co 80903",
        (
            "444 e costilla st, colorado springs, co 80903",
            "444 e costilla st colorado springs co 80903",
        ),
        ((AMRIZE, "main costilla plant"), (HOLCIM, "main costilla plant")),
        "main costilla plant",
        AMRIZE_CLUSTER,
        ASTM,
        ("2023-08-18", "2023-11-16"),
    ),
)


def _text(value: object) -> str:
    return "" if pd.isna(value) else " ".join(str(value).lower().split())


def _date(value: object) -> str:
    """Parse declaration date labels, not timezone-bearing instants."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()  # noqa: DTZ007
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date in alias block: {value!r}")


def _json_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else value


def _payload_value(field: str, value: object) -> object:
    if field == "expires":
        return _date(value)
    if field == "wc_raw":
        try:
            number = float(value)
            return None if pd.isna(number) else number
        except (ValueError, TypeError):
            return _text(value)
    if field in ("components", "description"):
        return _text(value)
    return _json_scalar(value)


def _block_id(site: Site, issue: str) -> str:
    return f"{site.key}:{issue}"


def allowlist() -> list[dict]:
    return [
        {
            "block_id": _block_id(site, issue),
            "issued_iso": issue,
            "operator": site.operator,
            "canonical_company": site.cluster,
            "canonical_plant": site.plant,
            "canonical_address": site.address,
            "canonical_source": f"urn:epd-alias:{_block_id(site, issue)}",
            "address_aliases": list(site.address_aliases),
            "company_plant_aliases": [list(pair) for pair in site.company_plants],
        }
        for site in SITES
        for issue in site.dates
    ]


def reconcile_aliases(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a new frame and JSON-serializable audit metadata, without I/O.

    Cross-label identities: allowlisted site/issue/operator + mix/age/unit.
    Any disagreement in the analytical payload quarantines every variant.
    Concordant identities retain the minimum original row_id, never minimum
    GWP. Single-label identities are NOT deduplicated here. All records in an
    affected block receive canonical menu identifiers, including unique mixes.
    Every input column is preserved; unaffected records are unchanged.
    """
    if not frame.columns.is_unique:
        raise ValueError("Duplicate input columns")
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing screened-frame columns: {missing}")
    if frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError("row_id must be nonmissing and unique")
    ids = pd.to_numeric(frame.row_id, errors="raise")
    if not ((ids >= 2) & (ids % 1 == 0)).all():
        raise ValueError("row_id must contain original integer CSV data positions")

    work = frame.copy(deep=True).reset_index(drop=True)
    matched: dict[int, tuple[Site, str]] = {}
    for site in SITES:
        eligible_pairs = set(site.company_plants) | {(site.cluster, site.plant)}
        mask = (
            work.address.map(_text).isin((*site.address_aliases, site.address))
            & work.operator.map(_text).eq(site.operator)
            & pd.Series(
                [
                    (c, p) in eligible_pairs
                    for c, p in zip(work.company.map(_text), work.plant.map(_text))
                ],
                index=work.index,
            )
        )
        for idx in work.index[mask]:
            issue = _date(work.at[idx, "issued"])
            if issue in site.dates:
                if idx in matched:
                    raise ValueError("Overlapping alias allowlist entries")
                matched[idx] = (site, issue)

    identities: dict[tuple, list[int]] = {}
    for idx, (site, issue) in matched.items():
        row = work.loc[idx]
        key = (
            _block_id(site, issue),
            _text(row["mix"]),
            _json_scalar(row["age_days"]),
            _text(row["unit"]),
        )
        if not key[1] or not key[3] or key[2] is None:
            raise ValueError(f"Incomplete identity in alias block: row {row.row_id}")
        identities.setdefault(key, []).append(idx)

    identity_audit, record_audit = [], []
    removed: set[int] = set()
    counts = {
        "cross_label_identities": 0,
        "cross_label_rows": 0,
        "quarantined_identities": 0,
        "quarantined_rows": 0,
        "components_conflict_identities": 0,
        "wc_conflict_identities": 0,
        "other_payload_conflict_identities": 0,
        "reconciled_identities": 0,
        "duplicate_rows_removed": 0,
    }
    for key, indices in sorted(identities.items()):
        part = work.loc[indices].sort_values("row_id")
        labels = sorted(part.company.map(_text).unique())
        cross_label = len(labels) > 1
        conflicts = (
            [
                field
                for field in PAYLOAD_FIELDS
                if len({_payload_value(field, value) for value in part[field]}) > 1
            ]
            if cross_label
            else []
        )
        if cross_label:
            counts["cross_label_identities"] += 1
            counts["cross_label_rows"] += len(part)
        if conflicts:
            removed.update(indices)
            counts["quarantined_identities"] += 1
            counts["quarantined_rows"] += len(part)
            counts["components_conflict_identities"] += int("components" in conflicts)
            counts["wc_conflict_identities"] += int("wc_raw" in conflicts)
            counts["other_payload_conflict_identities"] += int(
                bool(set(conflicts) - {"components", "wc_raw"})
            )
            representative = None
        elif cross_label:
            representative = int(part.row_id.iloc[0])
            removed.update(part.index[1:])
            counts["reconciled_identities"] += 1
            counts["duplicate_rows_removed"] += len(part) - 1
        else:
            representative = None
        identity_id = hashlib.sha256(json.dumps(key).encode("ascii")).hexdigest()[:20]
        identity_audit.append(
            {
                "identity_id": identity_id,
                "block_id": key[0],
                "mix": key[1],
                "age_days": key[2],
                "unit": key[3],
                "cross_label": cross_label,
                "row_ids": [int(v) for v in part.row_id],
                "company_aliases": labels,
                "source_urls": sorted(part.source.astype(str).unique()),
                "conflict_fields": conflicts,
                "representative_row_id": representative,
                "action": "quarantine"
                if conflicts
                else ("reconcile" if cross_label else "retain_unique_identity"),
            }
        )
        for idx, row in part.iterrows():
            site, issue = matched[idx]
            canonical = {
                "company": site.cluster,
                "plant": site.plant,
                "address": site.address,
                "issued": issue,
                "operator": site.operator,
                "source": f"urn:epd-alias:{_block_id(site, issue)}",
            }
            action = (
                "quarantined"
                if conflicts
                else ("duplicate_removed" if idx in removed else "retained")
            )
            record_audit.append(
                {
                    "row_id": int(row.row_id),
                    "identity_id": identity_id,
                    "block_id": key[0],
                    "action": action,
                    "representative_row_id": representative
                    if cross_label
                    else int(row.row_id),
                    "conflict_fields": conflicts,
                    **{
                        f"original_{col}": _json_scalar(row[col])
                        for col in frame.columns
                    },
                    **{f"canonical_{col}": value for col, value in canonical.items()},
                }
            )
            if idx not in removed:
                for col, value in canonical.items():
                    work.at[idx, col] = value

    kept_positions = [i for i in work.index if i not in removed]
    result = work.loc[kept_positions].copy()
    result.index = frame.index.take(kept_positions)
    unaffected = [i for i in work.index if i not in matched]
    pd.testing.assert_frame_equal(
        frame.iloc[unaffected].reset_index(drop=True),
        work.loc[unaffected].reset_index(drop=True),
    )
    assert (
        len(result)
        == len(frame) - counts["quarantined_rows"] - counts["duplicate_rows_removed"]
    )
    metadata = {
        "policy_version": POLICY_VERSION,
        "summary": {
            "input_rows": len(frame),
            "output_rows": len(result),
            "allowlisted_blocks": 13,
            "matched_blocks": len(
                {(_block_id(s, issue)) for s, issue in matched.values()}
            ),
            "affected_input_rows": len(matched),
            "unaffected_input_rows": len(unaffected),
            "unaffected_records_equal": True,
            **counts,
        },
        "source_cluster_map": SOURCE_CLUSTER_MAP.copy(),
        "source_cluster_interpretation": "linked source-label clusters; NOT verified legal suppliers",
        "canonical_source_interpretation": "analytical URNs, not retrieval URLs; use original_source in record_audit",
        "source_verification": "Original-PDF and omitted raw-field conflicts remain unresolved; no independent source validation is claimed.",
        "allowlist": allowlist(),
        "identity_audit": identity_audit,
        "record_audit": sorted(record_audit, key=lambda r: r["row_id"]),
    }
    return result, metadata


def write_audits(metadata: dict, output_dir: Path) -> None:
    """Explicitly export only alias_* audit files; never write parent results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("allowlist", "identity_audit", "record_audit"):
        records = [
            {
                k: json.dumps(v, ensure_ascii=True)
                if isinstance(v, (list, dict))
                else v
                for k, v in row.items()
            }
            for row in metadata[key]
        ]
        pd.DataFrame(records).to_csv(output_dir / f"alias_{key}.csv", index=False)
    pd.DataFrame(
        [
            {"company_label": k, "source_cluster": v, "legal_supplier_verified": False}
            for k, v in metadata["source_cluster_map"].items()
        ]
    ).to_csv(output_dir / "alias_source_clusters.csv", index=False)
    concise = {
        k: v
        for k, v in metadata.items()
        if k not in ("allowlist", "identity_audit", "record_audit")
    }
    (output_dir / "alias_summary.json").write_text(
        json.dumps(concise, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def verification_rows(
    frame: pd.DataFrame, metadata: dict, selector_path: Path
) -> list[dict]:
    """Run existing pure selector functions, without prepare(), main(), or writes."""
    spec = importlib.util.spec_from_file_location("alias_selector_check", selector_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load selector module: {selector_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    choices = module.choices(frame, "description_series", 1000)
    common = module.common_cohort(frame, choices)
    joined = choices.merge(
        common[["menu_id", "target_psi", "ratio_gwp", "series_gwp"]],
        on=["menu_id", "target_psi"],
        validate="one_to_one",
    )
    joined["source_cluster"] = joined.company.replace(metadata["source_cluster_map"])
    joined["window_positive"] = joined.gain_fraction.gt(1e-9)
    joined["objective_positive"] = joined.ratio_gwp.gt(joined.series_gwp + 1e-9)
    joined["objective_excess"] = joined.ratio_gwp / joined.series_gwp - 1
    joined["volume_threshold"] = 1 - joined.series_gwp / joined.ratio_gwp
    rows = []
    for label, part in (
        ("all", joined),
        ("exclude_nrm", joined[joined.company.ne("national ready mix")]),
    ):
        if part.empty:
            rows.append({"cohort": label, "cases": 0})
            continue
        affected = part[part.objective_positive]
        clusters = part.groupby("source_cluster")[
            ["window_positive", "objective_positive"]
        ].mean()
        rows.append(
            {
                "cohort": label,
                "cases": len(part),
                "company_field_labels": int(part.company.nunique()),
                "source_label_clusters": len(clusters),
                "legal_supplier_count": "UNVERIFIED",
                "window_positive": int(part.window_positive.sum()),
                "objective_positive": int(part.objective_positive.sum()),
                "window_positive_rate": float(part.window_positive.mean()),
                "objective_positive_rate": float(part.objective_positive.mean()),
                "equal_cluster_window_rate": float(clusters.window_positive.mean()),
                "equal_cluster_objective_rate": float(
                    clusters.objective_positive.mean()
                ),
                "median_objective_excess_affected": float(
                    affected.objective_excess.median()
                ),
                "median_volume_threshold_affected": float(
                    affected.volume_threshold.median()
                ),
                "window_robust_5pct": int(
                    part.break_even_relative_perturbation.gt(0.05).sum()
                ),
                "window_robust_10pct": int(
                    part.break_even_relative_perturbation.gt(0.10).sum()
                ),
            }
        )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "derived")
    parser.add_argument(
        "--verify-with",
        type=Path,
        help="Existing analyze_epd_choices.py providing pure selectors",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    result, metadata = reconcile_aliases(frame)
    metadata["input_file"] = str(args.input.resolve())
    metadata["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    metadata["module_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.verify_with:
        records = verification_rows(result, metadata, args.verify_with)
        metadata["selector_module_sha256"] = hashlib.sha256(
            args.verify_with.read_bytes()
        ).hexdigest()
        metadata["verification"] = records
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(
            args.output_dir / "alias_verification.csv", index=False
        )
    write_audits(metadata, args.output_dir)
    print(json.dumps(metadata["summary"], indent=2))
    if args.verify_with:
        print(json.dumps(metadata["verification"], indent=2))


if __name__ == "__main__":
    main()
