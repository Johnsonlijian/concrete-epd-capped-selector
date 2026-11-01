# Reproducibility package: capped strength-credit selector for concrete EPD comparisons

This repository accompanies the manuscript

> **Quantity constraints and strength credit in concrete EPD comparisons: a capped selector
> tested against code-compliant column designs**

It contains the analysis and design code, the derived tables needed to regenerate the
manuscript numbers and figures, the four vector figures, checksums, tests and a runbook.
The manuscript and submission files are not redistributed.

This is release **v1.0.1** (2026-11-01). It contains the C3/C3-prime selector-displacement
definitions, the strict observable-data terminology and the final analysis outputs.

## What is included

```
code/         analysis, design, figure, provenance and test scripts
derived/      derived CSV/JSON outputs used by the manuscript and figures
figures/      Fig1鈥揊ig4 as vector PDFs
README.md     this file
RUNBOOK.md    exact commands and environment
requirements.txt
DATASETS_AND_LINKS.csv
CITATION.cff
LICENSE
SHA256SUMS.txt
```

The archive does **not** redistribute the raw Mendeley compilation or third-party EPD PDFs.
It does include one record-level derived table, `derived/alias_input_records.csv`, needed to
run the alias-reconciliation test; it is derived from the public CC BY 4.0 input and contains
no EPD PDFs. The public input is listed in `DATASETS_AND_LINKS.csv`.

## Data

The primary input is the public Mendeley Data compilation:

- Broyles, J. *Compiled Dataset of Concrete Mixture Environmental Product Declarations in the
  U.S.A.*, version 5.
- DOI: <https://doi.org/10.17632/r4jgxk2mhn.5>
- License: CC BY 4.0.

Download the CSV and place it at:

```
raw/Compiled_Concrete_EPD_Data_Version_4c_Final_Published.csv
```

or set the environment variable `R1_RAW` to its full path. The code also accepts
`R1_DATA_DIR` and `R1_OUT_DIR` to point at alternate `derived/` folders.

## Environment

Python 3.11.15 with pandas 2.3.3, NumPy 2.2.6, Matplotlib 3.11.0, PyMuPDF 1.28.2 and
pytest 9.1.1. A TeX installation is needed only to rebuild the manuscript PDFs; the
manuscript source is not part of this repository.

## Reproduce

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# after placing the public CSV in raw/ or setting R1_RAW
python code/analyze_epd_choices.py
python code/quantity_model.py
python code/make_figures_and_tables.py
python code/decomposition.py
python code/column_design.py
python code/strict_subset.py
python code/field_provenance.py
python code/robustness.py
python code/extended_checks.py
python code/make_outputs.py

python -m pytest -q code
```

`python code/robustness.py --bootstrap-only` regenerates only the supplier-cluster bootstrap
(`derived/r1_bootstrap_supplier_clusters.csv`) without the Monte Carlo runs.

The optional marked-manuscript utility compares two LaTeX sources; every sentence, caption,
display equation and table cell that differs is shown in blue. The manuscript sources are
not redistributed, so this utility is useful only when the two source versions are supplied
separately:

```bash
python code/make_highlighted.py PREVIOUS/main.tex CORRECTED/main.tex main_marked.tex
```

The scripts are deterministic except for the declared Monte Carlo and bootstrap analyses,
which use the fixed seed `20261013`.

With the full public input and the complete derived tables, `pytest -q code` returns
**51 passed**. From this archive alone, two raw-data-dependent tests skip and the result is
**49 passed, 2 skipped**; the archived derived tables are still sufficient to inspect every
reported number and figure.

## Key conventions

- **C2** is the set of matched cases in which the ratio selector `R` chooses a higher declared
  GWP than fixed-volume selection `F`: 995 of 1,538 cases (64.7%).
- **C3** is `C2 鈭?{Q 鈮?R}`; **C3鈥?* is `C2 鈭?{Q 鈮?F}`. Both numerators are restricted to C2,
  but both rates use all 1,538 cases. The two counts are not complementary: an intermediate
  resolution is counted in both.
- The **strict observable-data subset** is a sensitivity subset built from the recorded PCR
  fields. Equal A3 GWP and matching populated fields are compatibility proxies, not proof of
  identical allocation or full PCR comparability.
- The **governing minimum** is computed before candidate-specific design from the exemption
  size and the magnifier-limited size at the lowest retained strength; the manuscript gives
  the closed-form equations.

## Outputs

`make_outputs.py` writes regenerated figures to `figures/` and table/number inputs to a
`manuscript/` directory by default. Set `R1_OUT_DIR` to write those inputs elsewhere. The
manuscript directory is not tracked in this repository; the archived `derived/` and
`figures/` trees are the reference outputs.

## License

Code: MIT. Derived tables and figures produced by the author: CC BY 4.0. The input dataset
remains under its own CC BY 4.0 terms. See `LICENSE`.

## Contact

Lijian REN, Inner Mongolia University of Technology, `renlijian@imut.edu.cn`.
