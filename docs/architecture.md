# Architecture

This document describes the **implemented** pipeline. It is not a proposal to replace the World Model with ordinary classification.

## Goal

Given a chronological sequence of network graphs \(G_t, G_{t+1}, \ldots\), the model encodes \(G_t\) into a latent state \(z(t)\) and **rolls out** \(K\) steps to predict quantities at **\(t+K\)**:

- future **attack** probability (binary head)
- future **macro network state**
- **MITRE ATT&CK tactic** logits when a mapping is evidence-backed

Horizon \(K\) is configured in `config.yaml` (default **3**). Targets are aligned to the **future** window, not \(y_t\). Sequences do not roll across file/scenario boundaries.

## Components (all retained)

| Piece | Role |
| --- | --- |
| Dataset adapters | CIC-IDS2018, CTU-13, UNSW-NB15 → unified flow schema |
| Preprocessing | Fit scaler on **train** only; persist for Streamlit |
| Temporal windowing | Chronological windows per file/scenario |
| Graph builder | Nodes = source/destination identities (IPs when valid); 11 edge features |
| GNN encoder | Spatial encoding → \(z(t)\) |
| GRU World Model | Temporal dynamics + **recursive** K-step rollout |
| State decoder | Attack / net-state / MITRE heads |
| LR baseline | Same split, same 11 features, not a substitute World Model |
| Evaluation | Validation-only threshold; held-out test evaluated once |
| Explainability | Gradient × input on the 11 edge features (not SHAP / GNNExplainer unless that code exists) |
| Streamlit `app.py` | Offline SOC UI |

## Unified edge features (11)

`duration`, `packets_forward`, `packets_backward`, `bytes_forward`, `bytes_backward`, `rate_fwd`, `rate_bwd`, `byte_rate_fwd`, `byte_rate_bwd`, `avg_pkt_size_fwd`, `avg_pkt_size_bwd`

## Data split

Per file/scenario, a **chronological** 70 / 10 / 20 train / validation / test split. No shuffling of time. The test set is not used to pick \(K\) or the classification threshold.

## Training vs inference

- **Smoke:** `python -m src.train --mode smoke --epochs 1` — small sample, pipeline check.
- **Full:** `python -m src.train --mode full` — only after the pipeline is trusted.
- **Dashboard:** `python -m streamlit run app.py` — loads `checkpoints/` if present.

## Datasets

Corpora live under `datasets/` on each machine and are **not** stored in git. See the README for download expectations (CIC-IDS2018, CTU-13, UNSW-NB15).
