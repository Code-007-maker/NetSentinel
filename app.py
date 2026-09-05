"""Offline Streamlit SOC dashboard backed by the local world-model checkpoint."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.inference import analyze, build_window_graphs, load_bundle, prepare_input, validate_upload
from src.train import IDX_TO_TACTIC

st.set_page_config(page_title="Predictive SOC", page_icon=":material/security:", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data(max_entries=3)
def get_traffic(filename: str, content: bytes):
    return prepare_input(filename, content)


def risk_level(probability: float, threshold: float) -> str:
    if probability >= min(.95, threshold + .25):
        return "CRITICAL"
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold * .65:
        return "MEDIUM"
    return "LOW"


def main():
    st.title("Predictive SOC")
    st.caption("Offline GNN + GRU world-model forecasting. Research smoke model; not production or real-time detection.")
    try:
        bundle = get_bundle()
        model_status = "available"
    except Exception as exc:
        bundle, model_status = None, str(exc)
        st.error(f"MODEL NOT AVAILABLE — {exc}")
    cfg = load_config()
    with st.sidebar:
        st.header("Analysis input")
        upload = st.file_uploader("Traffic file", type=["csv", "pcap", "pcapng", "binetflow"])
        if upload:
            st.caption(f"Type: {upload.name.rsplit('.', 1)[-1].upper()} · {upload.size / 1024:.1f} KB")
        seconds = st.number_input("Analysis window (seconds)", 1, 3600, int(cfg["hyperparameters"]["time_window_ms"] / 1000))
        horizon = st.slider("Forecast horizon K", 1, 10, min(bundle.selected_k if bundle else 3, 10))
        st.caption("Dataset-independent local inference")
        st.caption(f"Model status: {model_status}")
        submit = st.button("Analyze traffic", type="primary", disabled=bundle is None or upload is None)
    if not submit:
        st.info("Upload CSV, PCAP, PCAPNG, or BinetFlow traffic and select Analyze traffic.")
        return
    if error := validate_upload(upload.name, upload.getvalue()):
        st.error(error)
        return
    try:
        with st.spinner("Extracting traffic, constructing graphs, and recursively forecasting…"):
            traffic = get_traffic(upload.name, upload.getvalue())
            windows = build_window_graphs(traffic, int(seconds * 1000), bundle.scaler)
            result = analyze(bundle, windows, horizon)
    except Exception as exc:
        st.error(f"Analysis could not complete: {exc}")
        return

    graph, frame = result["graph"], result["frame"]
    current, forecast = result["current_probability"], result["forecast_probabilities"][-1]
    with st.container(horizontal=True):
        st.metric("Current risk", risk_level(current, bundle.threshold), border=True)
        st.metric("Forecast risk", risk_level(forecast, bundle.threshold), border=True)
        st.metric("Active hosts", int(graph.x.size(0)), border=True)
        st.metric("Network flows", len(frame), border=True)
        st.metric("Current attack probability", f"{current:.1%}", border=True)
        st.metric("Forecast horizon", f"t+{horizon}", border=True)

    st.subheader("Network status")
    overview, evidence = st.columns(2)
    with overview:
        st.caption(f"Actual current graph: {graph.x.size(0)} nodes · {graph.edge_index.size(1)} edges")
        st.markdown("**Top communicating hosts**")
        st.dataframe(result["hosts"], hide_index=True, width="stretch")
    with evidence:
        st.markdown("**Traffic evidence: current-window flow means**")
        st.dataframe(pd.DataFrame(result["flow_means"].items(), columns=["indicator", "mean"]), hide_index=True, width="stretch")

    st.subheader("Attack forecast")
    timeline = pd.DataFrame({
        "state": ["current"] + [f"t+{step}" for step in range(1, horizon + 1)],
        "attack probability": [current] + result["forecast_probabilities"],
    })
    st.line_chart(timeline, x="state", y="attack probability", width="stretch")
    st.caption(f"Risk labels use the saved validation threshold ({bundle.threshold:.2f}). No calibrated uncertainty is available.")

    st.subheader("Future state")
    state_summary = [{"state": "current", "latent norm": float(np.linalg.norm(result["current_state"])), "latent mean": float(np.mean(result["current_state"]))}]
    state_summary.extend({"state": f"t+{step}", "latent norm": float(np.linalg.norm(state)), "latent mean": float(np.mean(state))}
                         for step, state in enumerate(result["future_states"], 1))
    st.dataframe(pd.DataFrame(state_summary), hide_index=True, width="stretch")

    st.subheader("MITRE ATT&CK")
    tactic = IDX_TO_TACTIC.get(int(np.argmax(result["mitre_logits"])), "Unknown / Unsupported")
    if tactic in {"None", "Unknown / Unsupported"}:
        st.info("No evidence-backed ATT&CK tactic is available for the forecast state.")
    else:
        st.warning(f"Forecast tactic at t+{horizon}: {tactic}. Decoder outputs are not calibrated confidence values.")

    st.subheader("Explainability")
    ranked = result["explanation"].get("ranked", [])
    if ranked:
        st.dataframe(pd.DataFrame(ranked), hide_index=True, width="stretch")
    else:
        st.info("No edge-feature attribution is available for this traffic window.")

    st.subheader("Analysis summary")
    indicators = ", ".join(item["feature"] for item in ranked[:3]) or "available traffic evidence"
    st.write(f"The latest window contains {len(frame)} flows across {graph.x.size(0)} hosts. The actual recursive t+{horizon} forecast is {risk_level(forecast, bundle.threshold)} ({forecast:.1%}). Analyst attention: review {indicators} and the communicating hosts listed above.")


if __name__ == "__main__":
    main()
