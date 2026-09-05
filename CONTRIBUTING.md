# Contributing

Simple rules for collaborating on this project.

## Branches

- Create a **feature branch** from the default branch (`main` / `master`).
- Name it after the work, for example `fix/cic-ip-columns` or `docs/setup`.
- Do not commit directly to the default branch for non-trivial work.

## Do not commit data or artifacts

- **Do not commit datasets** (`datasets/`, CSVs, PCAPs, `.binetflow`, archives).
- **Do not commit checkpoints** (`checkpoints/`, `*.pt`, `*.joblib`, scalers).
- **Do not commit generated outputs** (`outputs/`, logs).
- **Do not commit secrets** (`.env`, Streamlit `secrets.toml`, API keys).

Download datasets locally and keep them gitignored.

## Tests before push

From the project root, with the virtual environment activated:

```powershell
python -m pytest -q
```

Fix failures before you push. Do not run full training just to land a code change.

## Commit messages

Write a **meaningful** commit message that explains *why* the change exists, not only which files moved. Example:

```
Select attack threshold on validation only so test FPR is honest.
```

## Pull requests

- Open a **PR for major changes** (architecture, evaluation protocol, dataset adapters, dashboard behavior).
- Keep PRs reviewable: one concern per PR when practical.
- Describe how you tested (pytest, smoke train, Streamlit) without pasting fabricated metrics.
