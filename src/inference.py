"""Offline, artifact-backed inference service for the SOC dashboard."""
from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.dataset_adapters import CICIDSAdapter, CTU13Adapter, UNSWAdapter, compute_derived_features
from src.explainability import explain_edge_features
from src.feature_schema import EDGE_FEATURE_NAMES
from src.gnn_encoder import GNNEncoder
from src.graph_builder import GraphBuilder
from src.mitre_mapper import MitreMapper
from src.pcap_parser import extract_pcap_features
from src.preprocessing import FlowFeatureScaler
from src.temporal_windowing import create_time_windows
from src.world_model import StateDecoder, TemporalWorldModel

SUPPORTED_SUFFIXES = {".csv", ".pcap", ".pcapng", ".binetflow"}


@dataclass
class InferenceBundle:
    gnn: GNNEncoder
    wm: TemporalWorldModel
    decoder: StateDecoder
    scaler: FlowFeatureScaler
    mapper: MitreMapper
    selected_k: int
    threshold: float


def validate_upload(filename: str, content: bytes) -> str | None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return f"Unsupported file type '{suffix or 'none'}'. Use CSV, PCAP, PCAPNG, or BinetFlow."
    if not content:
        return "The uploaded file is empty."
    return None


def load_bundle(checkpoint_dir: str | Path = "checkpoints") -> InferenceBundle:
    """Load the actual local checkpoint and fitted scaler; never create substitutes."""
    root = Path(checkpoint_dir)
    checkpoint_path = root / "world_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("MODEL NOT AVAILABLE: world_model.pt is missing.")
    if not (root / "scaler.joblib").exists() or not (root / "feature_schema.json").exists():
        raise FileNotFoundError("MODEL NOT AVAILABLE: fitted scaler/schema is missing.")
    cfg = load_config()
    hp = cfg["hyperparameters"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # The validated UI integration configuration is residual, delta scale 1.0.
    # Saved parameter tensors are loaded from the real checkpoint below.
    transition = checkpoint.get("transition", "residual")
    delta_scale = float(checkpoint.get("delta_scale", 1.0))
    gnn = GNNEncoder(hp["node_feature_dim"], hp["edge_feature_dim"], hp["gnn_hidden_dim"], hp["gnn_out_dim"])
    wm = TemporalWorldModel(hp["gnn_out_dim"], hp["wm_hidden_dim"], hp["wm_num_layers"], transition, delta_scale)
    decoder = StateDecoder(hp["gnn_out_dim"], hp["num_mitre_tactics"])
    gnn.load_state_dict(checkpoint["gnn_state_dict"])
    wm.load_state_dict(checkpoint["wm_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    for model in (gnn, wm, decoder):
        model.eval()
    return InferenceBundle(
        gnn, wm, decoder, FlowFeatureScaler.load(root), MitreMapper(cfg["mitre"]["mapping_file"]),
        int(checkpoint.get("selected_k", hp["k_rollout_steps"])), float(checkpoint.get("selected_threshold", .5)),
    )


def _adapter(filename: str):
    name = filename.lower()
    if name.endswith(".binetflow") or "ctu" in name:
        return CTU13Adapter()
    if "unsw" in name:
        return UNSWAdapter()
    return CICIDSAdapter()


def prepare_input(filename: str, content: bytes) -> pd.DataFrame:
    """Parse a user file to unified timestamped traffic without inventing fields."""
    if error := validate_upload(filename, content):
        raise ValueError(error)
    suffix = Path(filename).suffix.lower()
    if suffix in {".pcap", ".pcapng"}:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        try:
            frame = extract_pcap_features(str(temporary_path), max_packets=50_000)
        finally:
            temporary_path.unlink(missing_ok=True)
        if frame.empty:
            raise ValueError("No usable IP flow records were extracted from this capture.")
        frame = compute_derived_features(frame)
    elif suffix == ".binetflow":
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        try:
            frame = _adapter(filename).load_flow_data(temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    else:
        try:
            frame = _adapter(filename).load_flow_data(pd.read_csv(io.BytesIO(content), low_memory=False))
        except Exception as exc:
            raise ValueError(f"CSV could not be parsed into the supported flow schema: {exc}") from exc
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        raise ValueError("Unsupported schema: no timestamped usable flow records.")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        raise ValueError("No valid timestamps remain after parsing.")
    return frame


def build_window_graphs(frame: pd.DataFrame, window_ms: int, scaler: FlowFeatureScaler) -> list[tuple[pd.DataFrame, Any]]:
    builder, windows, built = GraphBuilder(), create_time_windows(frame, window_ms), []
    for window in windows.values():
        graph = builder.build_window_graph(window)
        if graph is not None and graph.x.numel() and graph.edge_attr.numel():
            scaler.scale_graph(graph)
            built.append((window, graph))
    if not built:
        raise ValueError("No usable windows/graphs were built from the input traffic.")
    return built


def analyze(bundle: InferenceBundle, windows: list[tuple[pd.DataFrame, Any]], k: int) -> dict:
    """Run actual GNN encoding, recursive rollout, decoding, and attribution."""
    frame, graph = windows[-1]
    with torch.no_grad():
        z_t = bundle.gnn(graph.x, graph.edge_index, graph.edge_attr)
        z_1, hidden = bundle.wm(z_t.unsqueeze(1))
        states = [z_1] + bundle.wm.rollout(z_1, hidden, max(k - 1, 0))
        current_probability = float(bundle.decoder(z_t)[0].item())
        forecasts, mitre_logits = [], []
        for state in states:
            probability, _, logits = bundle.decoder(state)
            forecasts.append(float(probability.item()))
            mitre_logits.append(logits.squeeze(0).cpu().numpy())
    hosts = pd.concat([frame.get("src_node", frame.get("src_ip")), frame.get("dst_node", frame.get("dst_ip"))]).value_counts().head(8)
    explanation = explain_edge_features(bundle.gnn, bundle.wm, bundle.decoder, graph, k=k, top_n=5)
    return {
        "graph": graph, "frame": frame, "current_state": z_t.squeeze().cpu().numpy(),
        "future_states": [state.squeeze().cpu().numpy() for state in states],
        "current_probability": current_probability, "forecast_probabilities": forecasts,
        "mitre_logits": mitre_logits[-1], "explanation": explanation,
        "hosts": hosts.rename_axis("host").reset_index(name="flows"),
        "flow_means": frame.reindex(columns=EDGE_FEATURE_NAMES, fill_value=0).mean().to_dict(),
    }
