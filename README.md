# Cybersecurity Temporal World Model (SOC Dashboard)

Offline research prototype that turns network-flow telemetry into **graphs**, encodes them with a **GNN**, and forecasts future attack risk with a **GRU World Model** and recursive **K-step rollout**. A Streamlit SOC dashboard runs inference locally. A logistic-regression baseline is trained on the same splits for comparison.

This repository contains **source code, tests, configs, and documentation only**. **Datasets are not included** and must be downloaded separately.

## Architecture

```
Flow telemetry (CSV / PCAP)
        │
        ▼
 Dataset adapters  →  unified 11-feature schema
        │
        ▼
 Chronological windows (no shuffle)
        │
        ▼
 Graph construction (IP nodes, flow edges)
        │
        ▼
 GNN encoder  →  latent z(t)
        │
        ▼
 GRU World Model  →  recursive K-step rollout  →  z(t+K)
        │
        ▼
 Decoder heads: future attack  |  network state  |  MITRE tactic
        │
        ▼
 Validation-selected threshold  +  feature attribution
        │
        ▼
 Streamlit SOC dashboard (offline)
```

The World Model is **not** ordinary single-step classification: it predicts the **future** graph state at horizon **K** (default K=3). Training also includes MITRE tactic prediction where a mapping is evidence-backed, plus a logistic-regression baseline on the same 11 flow features.

See [docs/architecture.md](docs/architecture.md) for a fuller description.

## Supported datasets

Download these yourself. They are **not** shipped on GitHub.

| Dataset | Typical local path | Notes |
| --- | --- | --- |
| **CIC-IDS2018** | `datasets/CIC-IDS2018/` | CICFlowMeter CSV (and optional PCAP if you have it) |
| **CTU-13** | `datasets/CTU-13-Dataset/` | NetFlow / `.binetflow` plus optional PCAP |
| **UNSW-NB15** | `datasets/UNSW-NB15/` | Official CSV splits / flow files |

Obtain files from the original publishers (CIC / CSE-CIC-IDS2018, CTU-13, UNSW-NB15). Place them under `datasets/` as shown in **Expected directory structure** below. Do not commit CSVs, PCAPs, or archives.

## Repository structure

```
app.py                 Streamlit SOC dashboard
config.yaml            Paths and hyperparameters
mitre_mapping.yaml     Evidence-based MITRE label mapping
requirements.txt       Python dependencies
src/                   Training, evaluation, GNN, World Model, adapters
tests/                 Unit / pipeline tests
docs/                  Architecture and dataset notes
.streamlit/            Streamlit theme (no secrets)
```

Generated and ignored locally (do not commit):

```
datasets/              Downloaded corpora
checkpoints/           Model weights, scaler, threshold, schema
outputs/               Evaluation reports
```

## Expected directory structure

Create these folders in the project root (they are gitignored):

```
datasets/
  CIC-IDS2018/
    csv/                 CIC-IDS2018 flow CSVs
  CTU-13-Dataset/
    1/                   scenario folders with .binetflow / PCAP
    ...
  UNSW-NB15/             UNSW-NB15 CSVs

checkpoints/             written by smoke/full training
outputs/                 written by evaluation
```

`config.yaml` uses `paths.dataset_dir: "datasets"` (relative to the project root). Change it only if your data lives elsewhere on your machine.

## Windows setup

From the project root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Python **3.10–3.12** is recommended. PyTorch / PyTorch Geometric wheels may lag on newer interpreters.

## Tests

```powershell
python -m pytest -q
```

## Smoke training (do not start with full training)

Requires datasets already present under `datasets/`.

```powershell
python -m src.train --mode smoke --epochs 1
```

Smoke mode uses a small chronological sample. Full training is a separate, explicit step (`--mode full`) and is not required to collaborate on code.

## Streamlit dashboard

```powershell
python -m streamlit run app.py
```

The app is **offline**. If `checkpoints/` is empty, the UI should warn that the model is untrained rather than inventing predictions.

## Team workflow

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: feature branches, never commit datasets or checkpoints, run tests before push, open a PR for major changes.
