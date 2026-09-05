"""
Canonical feature schema for the Cybersecurity World Model.

All modules must import from here. Do NOT define feature lists anywhere else.

EDGE features (per-flow, 11 total):
  5 base  : duration, pkt_fwd, pkt_bwd, bytes_fwd, bytes_bwd
  6 derived: rate_fwd, rate_bwd, byte_rate_fwd, byte_rate_bwd,
             avg_pkt_size_fwd, avg_pkt_size_bwd

NODE features (per-host/endpoint, 2):
  out_degree, in_degree

NOTE ON IP AVAILABILITY:
  Some CIC-IDS2018 CSVs (notably Tuesday-20-02-2018) include Src IP / Dst IP.
  Those strings are used as graph node identities when present and non-empty.
  Other CICFlowMeter days have no host IPs; those files use protocol/port
  pseudo-nodes.  IPs are never invented.  UNSW-NB15 aggregated train/test
  sets also lack IPs and use the same fallback.
"""
from __future__ import annotations

# ── Edge feature schema (GraphBuilder edge_attr columns) ──────────────────────
EDGE_FEATURE_NAMES: list[str] = [
    "duration",
    "packets_forward",
    "packets_backward",
    "bytes_forward",
    "bytes_backward",
    "rate_fwd",
    "rate_bwd",
    "byte_rate_fwd",
    "byte_rate_bwd",
    "avg_pkt_size_fwd",
    "avg_pkt_size_bwd",
]

EDGE_FEATURE_DIM: int = len(EDGE_FEATURE_NAMES)   # 11

# ── Node feature schema ────────────────────────────────────────────────────────
NODE_FEATURE_NAMES: list[str] = ["out_degree", "in_degree"]
NODE_FEATURE_DIM: int = len(NODE_FEATURE_NAMES)    # 2

# ── Backward-compat alias (old name) ─────────────────────────────────────────
UNIFIED_FEATURE_NAMES = EDGE_FEATURE_NAMES
FLOW_FEATURE_COLS = EDGE_FEATURE_NAMES   # used by FlowFeatureScaler
# LR baseline uses the same 11 names (window-mean). See src/baseline_model.py.
