"""
Offline Cybersecurity SOC Dashboard — Streamlit Application

Inference pipeline when telemetry is uploaded:
  Upload (CSV / PCAP)
  → detect format → parse → apply SAVED scaler
  → chronological windows → build graphs
  → GNN encode → z(t)
  → recursive K-step rollout → z(t+1..t+K)
  → decode future attack probs / network state / MITRE
  → display on dashboard

Model Honesty:
  - If checkpoint is missing → UNTRAINED warning, no predictions shown.
  - If scaler is missing → MISSING PREPROCESSING warning, no inference.
  - MITRE tactics only shown when evidence-backed.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.config import load_config
from src.data_discovery import discover_datasets
from src.dataset_adapters import CTU13Adapter, CICIDSAdapter, UNSWAdapter
from src.explainability import explain_edge_features, format_top_features
from src.graph_builder import GraphBuilder
from src.mitre_mapper import MitreMapper
from src.pcap_parser import extract_pcap_features
from src.preprocessing import FlowFeatureScaler
from src.temporal_windowing import create_time_windows
from src.world_model import StateDecoder, TemporalWorldModel

st.set_page_config(
    page_title="Cybersecurity SOC Dashboard",
    layout="wide",
    page_icon="🛡️",
)

# ─────────────────────────────────────────────
# Checkpoint/artifact paths
# ─────────────────────────────────────────────

CKPT_DIR = Path("checkpoints")
CKPT_PATH = CKPT_DIR / "world_model.pt"
LR_PATH = CKPT_DIR / "lr_baseline.joblib"
SCALER_PATH = CKPT_DIR / "scaler.joblib"
SCHEMA_PATH = CKPT_DIR / "feature_schema.json"

# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

@st.cache_resource
def load_inference_bundle():
    """
    Attempt to load all required inference artifacts.
    Returns (gnn, wm, decoder, lr_model, scaler, selected_k, is_ready, status_msg).
    is_ready = True only when ALL components loaded successfully.
    """
    cfg = load_config()
    hp = cfg["hyperparameters"]
    mitre = MitreMapper(cfg["mitre"]["mapping_file"])

    gnn = GNNEncoder(
        node_in_dim=hp["node_feature_dim"],
        edge_in_dim=hp["edge_feature_dim"],
        hidden_dim=hp["gnn_hidden_dim"],
        out_dim=hp["gnn_out_dim"],
    )
    wm = TemporalWorldModel(
        z_dim=hp["gnn_out_dim"],
        hidden_dim=hp["wm_hidden_dim"],
        num_layers=hp["wm_num_layers"],
    )
    decoder = StateDecoder(z_dim=hp["gnn_out_dim"], num_mitre_tactics=hp["num_mitre_tactics"])

    issues = []

    selected_k = hp["k_rollout_steps"]  # default; overridden if checkpoint loads
    selected_threshold = 0.5

    if not CKPT_PATH.exists():
        issues.append(f"❌ Model checkpoint not found: `{CKPT_PATH}`")
    else:
        try:
            ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
            gnn.load_state_dict(ckpt["gnn_state_dict"])
            wm.load_state_dict(ckpt["wm_state_dict"])
            decoder.load_state_dict(ckpt["decoder_state_dict"])
            selected_k = ckpt.get("selected_k", hp["k_rollout_steps"])
            selected_threshold = float(ckpt.get("selected_threshold", 0.5))
        except Exception as e:
            issues.append(f"❌ Failed to load checkpoint: {e}")

    cfg_snap = CKPT_DIR / "training_config.json"
    if cfg_snap.exists():
        try:
            snap = json.loads(cfg_snap.read_text(encoding="utf-8"))
            selected_threshold = float(snap.get("selected_threshold", selected_threshold))
        except Exception:
            pass

    if not SCALER_PATH.exists() or not SCHEMA_PATH.exists():
        issues.append(f"❌ Preprocessing scaler missing: `{CKPT_DIR}`")
        scaler = None
    else:
        try:
            scaler = FlowFeatureScaler.load(CKPT_DIR)
        except Exception as e:
            issues.append(f"❌ Failed to load scaler: {e}")
            scaler = None

    if not LR_PATH.exists():
        issues.append(f"❌ LR baseline not found: `{LR_PATH}`")
        lr_model = None
    else:
        try:
            lr_model = joblib.load(LR_PATH)
        except Exception as e:
            issues.append(f"❌ Failed to load LR baseline: {e}")
            lr_model = None

    is_ready = len(issues) == 0
    gnn.eval(); wm.eval(); decoder.eval()
    return gnn, wm, decoder, lr_model, scaler, mitre, selected_k, selected_threshold, is_ready, issues


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _detect_adapter(filename: str):
    f = filename.lower()
    if "unsw" in f:
        return UNSWAdapter(), "UNSW-NB15"
    if "ctu" in f or "binetflow" in f or "capture" in f:
        return CTU13Adapter(), "CTU-13"
    return CICIDSAdapter(), "CIC-IDS2018"


def _build_graphs_from_df(df: pd.DataFrame, window_ms: int):
    windows = create_time_windows(df, window_ms=window_ms)
    builder = GraphBuilder()
    graphs = [builder.build_window_graph(w) for w in windows.values()]
    return graphs, windows


def _run_inference(gnn, wm, decoder, graph, k):
    with torch.no_grad():
        z_t = gnn(graph.x, graph.edge_index, graph.edge_attr)
        z_seq = z_t.unsqueeze(1)
        z_pred_t1, h_n = wm(z_seq)
        future_zs = [z_pred_t1] + wm.rollout(z_pred_t1, h_n, k=k - 1)
        curr_attack_prob, curr_net_state, _ = decoder(z_t)
        future_probs = []
        future_net_states = []
        future_mitre_logits = []
        for fz in future_zs:
            ap, ns, ml = decoder(fz)
            future_probs.append(ap.item())
            future_net_states.append(ns.detach().numpy()[0])
            future_mitre_logits.append(ml.detach().numpy()[0])
    return curr_attack_prob.item(), curr_net_state.detach().numpy()[0], future_probs, future_net_states, future_mitre_logits


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    st.title("🛡️ Predictive SOC Dashboard — Temporal World Model")
    cfg = load_config()
    hp = cfg["hyperparameters"]

    gnn, wm, decoder, lr_model, scaler, mitre, selected_k, selected_threshold, is_ready, issues = load_inference_bundle()

    # ── Model status banner ──────────────────────────────────────────
    if is_ready:
        st.success("✅ **Trained model checkpoint loaded.** Inference is active.")
    else:
        st.error(
            "⚠️ **UNTRAINED / NOT VALIDATED**\n\n"
            "One or more required artifacts are missing. "
            "Predictions will NOT be generated until training is complete.\n\n"
            + "\n".join(issues)
        )
        st.info(
            "Run training first:\n```\npython -m src.train --mode smoke --epochs 1\n```"
        )

    # ── Sidebar ─────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configuration")
    k_steps = st.sidebar.slider(
        "Forecasting Horizon (K-steps)",
        1, 10,
        selected_k,
        help="K for recursive rollout. Best K was selected by validation F1.",
    )
    st.sidebar.caption(f"Checkpoint selected K: **{selected_k}**")
    st.sidebar.caption(f"Frozen decision threshold: **{selected_threshold:.2f}** (validation Youden J)")

    st.sidebar.subheader("📂 Data Ingestion")
    ingestion_mode = st.sidebar.radio("Mode", ["Upload File", "Discover from datasets/"])

    df_raw = None
    file_label = ""

    if ingestion_mode == "Upload File":
        uploaded = st.sidebar.file_uploader(
            "Select CSV or PCAP", type=["csv", "pcap"]
        )
        if uploaded:
            file_label = uploaded.name
            st.sidebar.caption(
                f"📄 **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)"
            )
            if uploaded.name.endswith(".pcap"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                with st.spinner("Parsing PCAP…"):
                    df_raw = extract_pcap_features(tmp_path, max_packets=50000)
                os.remove(tmp_path)
                if df_raw.empty:
                    st.sidebar.error("No IP packets found in PCAP.")
                else:
                    st.sidebar.success(f"Parsed into {len(df_raw)} flows.")
            else:
                raw_df = pd.read_csv(uploaded)
                adapter, ds_name = _detect_adapter(uploaded.name)
                df_raw = adapter.load_flow_data(raw_df)
                st.sidebar.success(f"Loaded {len(df_raw)} rows ({ds_name})")

    else:
        with st.spinner("Scanning datasets/…"):
            inv = discover_datasets(cfg["paths"]["dataset_dir"])
        all_files = inv["csv"] + inv["binetflow"]
        if all_files:
            sel = st.sidebar.selectbox(
                "Dataset file", all_files, format_func=lambda x: str(Path(x).name)
            )
            file_label = str(Path(sel).name)
            if st.sidebar.button("Load selected file"):
                adapter, _ = _detect_adapter(str(sel))
                with st.spinner(f"Loading {Path(sel).name}…"):
                    df_raw = adapter.load_flow_data(sel)
                st.sidebar.success(f"Loaded {len(df_raw)} rows")
        else:
            st.sidebar.warning("No CSV/binetflow files found in datasets/")

    process_btn = st.sidebar.button("▶  Process Telemetry", disabled=(df_raw is None))

    if process_btn and df_raw is not None:
        with st.spinner("Processing telemetry…"):
            graphs, windows = _build_graphs_from_df(df_raw, hp["time_window_ms"])
        st.session_state["df"] = df_raw
        st.session_state["graphs"] = graphs
        st.session_state["windows"] = windows
        st.session_state["file_label"] = file_label
        st.sidebar.success(f"✔ {len(graphs)} time windows built.")

    # ── Guard: no data yet ──────────────────────────────────────────
    if "graphs" not in st.session_state or len(st.session_state["graphs"]) < 2:
        st.info(
            "Upload or select a telemetry file and click **Process Telemetry** to begin.\n\n"
            "_The file must produce at least 2 time windows._"
        )
        return

    df = st.session_state["df"]
    graphs = st.session_state["graphs"]
    windows = st.session_state["windows"]
    t = max(0, len(graphs) - 2)
    current_graph = graphs[t]
    current_window_df = list(windows.values())[t]

    # ── Guard: no inference without trained artifacts ────────────────
    if not is_ready:
        st.warning(
            "Dashboard populated with graph/feature data below. "
            "**Inference and forecasts are suppressed until trained weights are present.**"
        )
        st.subheader("1. Current Network State (preprocessing only)")
        st.metric("Total Flows (this window)", len(current_window_df))
        st.metric("Active Nodes", current_graph.x.size(0))
        return

    # ── Run inference ────────────────────────────────────────────────
    curr_prob, curr_ns, future_probs, future_ns, future_mitre_logits = _run_inference(
        gnn, wm, decoder, current_graph, k_steps
    )

    # Scaler-normalised flow features for LR
    lr_prob = None
    if lr_model is not None and scaler is not None:
        try:
            X_window = scaler.transform(current_window_df)
            X_agg = X_window.mean(axis=0, keepdims=True)
            lr_prob = lr_model.predict_proba(X_agg)[:, 1][0]
        except Exception:
            lr_prob = None

    # ── Layout ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Current Network State")
        st.metric("Flows in window", len(current_window_df))
        st.metric("Active nodes", current_graph.x.size(0))
        st.metric("Active edges", current_graph.num_edges)

    with col2:
        st.subheader("3. Current Attack Risk")
        st.metric("Attack probability (t)", f"{curr_prob:.1%}")

    with col3:
        st.subheader("12. Forecasting Config")
        st.info(
            f"**Selected K:** {selected_k}  \n"
            f"**Active K (UI):** {k_steps}  \n"
            f"**Decision threshold:** {selected_threshold:.2f}  \n"
            f"**Algorithm:** Recursive GNN-GRU  \n"
            f"**Source:** {st.session_state.get('file_label', 'N/A')}"
        )

    st.divider()
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("4 & 5. Infiltration Probability — Recursive K-step Forecast")
        times_hist = list(windows.keys())[:t + 1]
        probs_hist = []
        with torch.no_grad():
            for g in graphs[:t + 1]:
                z = gnn(g.x, g.edge_index, g.edge_attr)
                ap, _, _ = decoder(z)
                probs_hist.append(ap.item())

        times_future = [times_hist[-1] + (i + 1) for i in range(k_steps)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times_hist, y=probs_hist,
            mode="lines+markers", name="Observed",
            line=dict(color="#3b82f6"),
        ))
        fig.add_trace(go.Scatter(
            x=times_future, y=future_probs[:k_steps],
            mode="lines+markers", name="Recursive Forecast",
            line=dict(color="#ef4444", dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark", yaxis_title="Attack Probability",
            yaxis_range=[0, 1], margin=dict(t=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.subheader("7. Predicted MITRE ATT&CK Tactic")
        # Decode the most likely class from mitre logits at t+K
        last_mitre = future_mitre_logits[-1]
        pred_idx = int(np.argmax(last_mitre))
        from src.train import IDX_TO_TACTIC
        pred_tactic = IDX_TO_TACTIC.get(pred_idx, "Unknown")
        st.warning(f"**Predicted tactic at t+{k_steps}:** `{pred_tactic}`")
        st.caption(
            "Evidence: derived from temporal latent state z(t+K) via decoder.  "
            "Only tactics with evidence in `mitre_mapping.yaml` are reported. "
            "Others are shown as Unknown / Unsupported."
        )

    st.divider()
    col6, col7 = st.columns(2)

    with col6:
        st.subheader("10. World Model vs Logistic Regression")
        rows = [{"Model": "World Model (t+1)", "Attack Probability": future_probs[0]}]
        if lr_prob is not None:
            rows.append({"Model": "Logistic Regression", "Attack Probability": lr_prob})
        comp_df = pd.DataFrame(rows)
        fig2 = px.bar(
            comp_df, x="Model", y="Attack Probability", color="Model",
            template="plotly_dark", range_y=[0, 1],
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col7:
        st.subheader("11. Current vs Predicted Network State")
        ns_labels = ["Duration", "Fwd Pkts", "Bwd Pkts", "Fwd Bytes", "Bwd Bytes"]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Current z(t)", x=ns_labels, y=curr_ns.tolist()))
        fig3.add_trace(go.Bar(name="Predicted z(t+1)", x=ns_labels, y=future_ns[0].tolist()))
        fig3.update_layout(barmode="group", template="plotly_dark", margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    col8, col9 = st.columns(2)

    with col8:
        st.subheader("9. Suspicious Flows")
        benign_check = {"benign", "background", "normal"}
        suspicious = current_window_df[
            ~current_window_df["label"].str.lower().isin(benign_check)
        ] if "label" in current_window_df.columns else pd.DataFrame()
        if suspicious.empty:
            st.success("No suspicious flows in current window.")
        else:
            st.dataframe(suspicious.head(20), use_container_width=True)

    with col9:
        st.subheader("8. Top Contributing Features (gradient × input on 11 edge features)")
        expl = explain_edge_features(gnn, wm, decoder, current_graph, k=k_steps, top_n=5)
        if expl.get("ranked"):
            st.caption(f"Method: `{expl['method']}` — not SHAP / not GNNExplainer.")
            rows = [
                {
                    "Rank": i,
                    "Feature": item["feature"],
                    "Direction": item["direction"],
                    "Score": item["score"],
                }
                for i, item in enumerate(expl["ranked"], start=1)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            fig4 = px.bar(
                pd.DataFrame(rows),
                x="Feature",
                y="Score",
                template="plotly_dark",
                labels={"Score": "signed gradient×input"},
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No edge features available for attribution.")

    st.divider()
    st.subheader("13. Model Confidence")
    entropy = -sum(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9) for p in future_probs) / k_steps
    confidence = max(0.0, 1.0 - entropy / np.log(2))
    st.metric("Mean forecast confidence", f"{confidence:.1%}")
    st.caption("Confidence = 1 − normalised binary entropy across K-step predictions.")


if __name__ == "__main__":
    main()
