# Runbook

## 1. Inputs

1. Download the public Mendeley Data v5 CSV:
   <https://doi.org/10.17632/r4jgxk2mhn.5>
2. Put it at `raw/Compiled_Concrete_EPD_Data_Version_4c_Final_Published.csv`.
   Alternatively set `R1_RAW` to the full path.
3. The historical Knife River EPD and the StructurePoint worked example are public documents;
   their URLs are in `DATASETS_AND_LINKS.csv`. The scripts that need them use the cached
   derived tables in `derived/`; a fresh re-extraction requires the original PDFs.

## 2. Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python --version              # 3.11.x
```

The archived run used Python 3.11.15, pandas 2.3.3, NumPy 2.2.6, Matplotlib 3.11.0,
PyMuPDF 1.28.2, pytest 9.1.1.

## 3. Full rerun

Run from the archive root, in this order:

```bash
python code/analyze_epd_choices.py
python code/quantity_model.py
python code/make_figures_and_tables.py
python code/r1_decomposition.py
python code/r1_column_design.py
python code/r1_strict_subset.py
python code/r1_field_provenance.py
python code/r1_robustness.py
python code/r1_extended_checks.py
python code/r1_make_outputs.py
python -m pytest -q code
```

Expected test result with the raw input and full derived tables: **51 passed**. From this
archive alone: **49 passed, 2 skipped** (two tests need the raw record-level table).

`python code/r1_robustness.py --bootstrap-only` regenerates only
`derived/r1_bootstrap_supplier_clusters.csv`; the full run also repeats the Monte Carlo analyses.

## 4. Quick verification without the raw CSV

The archived `derived/` tables are sufficient for most figure and table regeneration, but
`r1_make_outputs.py` still expects the derived files listed in its `read_csv`/`read_json`
calls. If you only want to inspect the archived numbers, read:

- `derived/r1_decomposition_summary.json` — C1/C2/C3 ranges and the 81-point check.
- `derived/r1_decomposition_grid.csv` — the 12 analyst-defined scenarios.
- `derived/r1_comparability_ladder.csv` — scope ladder and C3 grid.
- `derived/r1_strict_subset_summary.json` — S1–S4 pass rates and strict case counts.
- `derived/r1_design_validation_summary.csv` — design agreement/regret.
- `derived/r1_robustness_montecarlo_summary.csv` — Monte Carlo stability.
- `derived/r1_decomposition_cases.csv` — case-level C1/C2 data.

## 5. Common issues

- `r1_core.py` looks for the raw CSV at `raw/` or `../raw/` relative to the archive root.
  Set `R1_RAW` if it is elsewhere.
- The code uses U.S. customary units internally (psi, in, kip), matching the source data and
  ACI 318-19; the manuscript reports SI first.
- Monte Carlo and bootstrap runs use seed `20261013`; small differences across library
  versions are possible.
- The raw compilation, EPD PDFs and most record-level intermediate tables are not included.
  `derived/alias_input_records.csv` is included because the alias-reconciliation test needs it.

## 6. What not to expect

This package does not contain the raw EPD PDFs, the full Mendeley CSV, the submitted
manuscript PDFs or any reviewer/editorial material. It does not prove scientific validity;
it documents the computations reported in the manuscript.
