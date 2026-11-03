# DBM301 Assignment 2 - Groceries

Group 3 project workspace for the Groceries data-mining assignment.

## Environment setup

Requirements: Git, VS Code, and [uv](https://docs.astral.sh/uv/). The project uses Python 3.13, selected by `.python-version`.

```powershell
git clone https://github.com/TruongMinhThang24/dbm301_asm2_group3_de2.git
Set-Location .\dbm301_asm2_group3_de2
uv sync
```

In VS Code, open this repository folder. Install the recommended Python, Pylance, and Jupyter extensions if prompted. The workspace settings select `.venv\Scripts\python.exe`; choose that interpreter/kernel if VS Code asks.

To check the environment:

```powershell
uv run python -c "import pandas, numpy, sklearn, mlxtend, matplotlib, seaborn, jupyter, ipykernel; print('Environment OK')"
uv run python -m unittest discover -s tests
```

After changing dependencies, regenerate the submission requirements file:

```powershell
uv export --format requirements-txt --no-hashes --no-emit-project --output-file requirements.txt
```

Commit `pyproject.toml` and `uv.lock` together so team members resolve the same environment. `requirements.txt` is an exported install list for submission; `uv.lock` is the reproducible project lock.

## Project layout

```text
data/
  raw/                 Instructor-provided source data (not modified)
  processed/           Reproducible intermediate data
src/groceries_group3/
  preparation/         Data loading and preparation
  association/         Apriori, FP-Growth, and association rules
  classification/      Classification utilities
  clustering/          Basket clustering utilities
  anomaly/             Outlier detection utilities
  common/              Shared paths and random seed
notebooks/             Analysis experiments and final notebook.ipynb
outputs/               Tables, figures, predictions.csv, anomaly_scores.csv
reports/               Written report and presentation
tests/                 Small checks for reusable code
docs/                  Team workflow and project notes
```

Rubric criteria and the evidence expected for each submission item are mapped in [docs/rubric_evidence_matrix.md](./docs/rubric_evidence_matrix.md).

The reusable project modules under `src/` support development and tests. The submission notebook is intentionally self-contained and does not import those modules; its fixed seed is 42.

## Data preparation — Thang's scope (WBS 2.0–2.7)

Place the instructor-provided `groceries_train.csv` and `groceries_test.csv` in `data/raw/`. The raw directory is Git-ignored. Run the notebook with the project kernel. Its completed code is limited to Thang's data preparation and EDA; Parts 2–6 remain scaffolds for the other assignment work.

The notebook embeds the reviewed item_group_rules mapping for 169 items and alternative assignments, so neither mapping file nor `src/` is required to run it. It writes the item mapping and disputed alternatives under `data/mappings/`. Measuring their effect on association rules belongs to task 3.5, not this section.

Test labels are hidden; the absence of `other vegetables` from the visible test items must never be treated as a negative label. Prediction and anomaly-score files are not produced by Thang's data-preparation section.

## Data and experiment rules

- Put the instructor-approved source file in `data/raw/`. Do not edit the original.
- Raw and processed data are ignored by Git by default. Share data only if the instructor permits it; otherwise tell teammates how to obtain the approved file.
- The notebook resolves the provided data folder relative to the current working directory; do not add personal absolute paths.
- Fit preprocessing only on training data and reuse the fitted transformation for validation/test data to avoid leakage.
- Record important parameters and seeds in the notebook. Keep the final notebook runnable from top to bottom and within the assignment's time limit.
- Package only the six instructor-required deliverables in `Nhom3_De2.zip`; do not include private or restricted source data.

## Team Git workflow

The repository owner can invite Hưng, Phú, and Quốc Anh from GitHub: open the repository's **Settings → Collaborators → Add people**, search each member's exact GitHub username, and grant repository access. Confirm that each invite is accepted before relying on private-repository access.

Use `main` for reviewed, working code. Each member creates a short-lived branch from an updated `main`, for example `feature/data-preparation` or `feature/clustering`, and opens a pull request before merging.

```powershell
git switch main
git pull
git switch -c feature/short-task-name
# make and test changes
git add <files>
git commit -m "Describe the change"
git push -u origin feature/short-task-name
```

Keep commits focused, pull before starting, and do not force-push shared branches. If a merge conflict occurs, coordinate with the file owner, resolve the marked sections, run the relevant checks, then commit the resolution. Hand off the branch name, changed files, test results, and remaining work.
