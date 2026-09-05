"""
GraphBuilder: converts a flow-level DataFrame (one time window) into a PyG Data graph.

Node identity:
  Prefer adapter-provided src_node / dst_node (real IPs when recovered).
  Fallback columns: src_ip / dst_ip.
  Pseudo-nodes are used only when the adapter could not recover an identity.

All edge features come from EDGE_FEATURE_NAMES in src/feature_schema.py.
Node features: [out_degree, in_degree].
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.feature_schema import EDGE_FEATURE_NAMES, EDGE_FEATURE_DIM, NODE_FEATURE_DIM

logger = logging.getLogger(__name__)

# IPv4, plus a light IPv6 check (contains ':') for anonymised/v6 strings.
_IPV4_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
_MISSING = {"", "nan", "none", "null", "nat", "-", "<na>", "n/a"}


def _looks_like_real_ip(s: str) -> bool:
    s = s.strip()
    if not s or s.lower() in _MISSING:
        return False
    if _IPV4_RE.match(s):
        return True
    # Preserve IPv6 / anonymised colon-separated identities if present.
    if ":" in s and any(ch.isalnum() for ch in s):
        return True
    return False


def _is_invalid_identity(s: str) -> bool:
    return (not s) or s.strip().lower() in _MISSING


class GraphBuilder:
    """Builds per-window PyG graphs from unified flow DataFrames."""

    def __init__(self) -> None:
        self._node_to_id: Dict[str, int] = {}
        self.invalid_ips: int = 0
        self.self_loops_total: int = 0
        self._logged_first: bool = False

    def _get_node_id(self, node: str) -> int:
        if node not in self._node_to_id:
            self._node_to_id[node] = len(self._node_to_id)
        return self._node_to_id[node]

    def reset(self) -> None:
        """Reset node registry between independent sequences."""
        self._node_to_id = {}
        self.invalid_ips = 0
        self.self_loops_total = 0
        self._logged_first = False

    def identity_audit(self) -> Dict[str, int]:
        all_nodes = list(self._node_to_id.keys())
        valid_ips = [n for n in all_nodes if _looks_like_real_ip(n)]
        pseudo_nodes = [n for n in all_nodes if not _looks_like_real_ip(n)]
        return {
            "real_ips": len(valid_ips),
            "pseudo_nodes": len(pseudo_nodes),
            "invalid_ips": int(self.invalid_ips),
            "self_loops": int(self.self_loops_total),
        }

    @staticmethod
    def _empty_graph() -> Data:
        return Data(
            x=torch.empty((0, NODE_FEATURE_DIM), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float),
            y=torch.tensor([0], dtype=torch.float),
        )

    def build_window_graph(
        self, df: pd.DataFrame, smoke_test_log: bool = False
    ) -> Optional[Data]:
        """
        Build a PyG graph from one time-window DataFrame.

        Columns used:
          src_node, dst_node       — node identifiers
          EDGE_FEATURE_NAMES[i]    — 11 edge features
          label                    — window-level attack flag

        Returns None if the graph is structurally corrupt (see validation).
        """
        if df.empty:
            return self._empty_graph()

        src_col = "src_node" if "src_node" in df.columns else "src_ip"
        dst_col = "dst_node" if "dst_node" in df.columns else "dst_ip"

        if src_col not in df.columns or dst_col not in df.columns:
            logger.warning("GraphBuilder: no src/dst columns found in df. Returning empty graph.")
            return self._empty_graph()

        edge_indices = []
        edge_feats = []

        for _, row in df.iterrows():
            src = str(row[src_col]).strip()
            dst = str(row[dst_col]).strip()
            if _is_invalid_identity(src) or _is_invalid_identity(dst):
                self.invalid_ips += int(_is_invalid_identity(src)) + int(_is_invalid_identity(dst))
                continue

            src_id = self._get_node_id(src)
            dst_id = self._get_node_id(dst)
            edge_indices.append([src_id, dst_id])

            feats = []
            for feat_name in EDGE_FEATURE_NAMES:
                v = row.get(feat_name, 0.0)
                v = float(v) if np.isfinite(float(v)) else 0.0
                feats.append(v)
            edge_feats.append(feats)

        if not edge_indices:
            return self._empty_graph()

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

        num_nodes = len(self._node_to_id)
        node_feats = torch.zeros((num_nodes, NODE_FEATURE_DIM), dtype=torch.float)
        if edge_index.numel() > 0:
            out_deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
            in_deg = torch.bincount(edge_index[1], minlength=num_nodes).float()
            node_feats[:, 0] = out_deg
            node_feats[:, 1] = in_deg

        is_attack_window = 1 if any(
            str(lbl).strip().lower() not in {"benign", "background", "normal", "0", ""}
            for lbl in df["label"]
        ) else 0
        y = torch.tensor([is_attack_window], dtype=torch.float)

        all_nodes = list(self._node_to_id.keys())
        valid_ips = [n for n in all_nodes if _looks_like_real_ip(n)]
        pseudo_nodes = [n for n in all_nodes if not _looks_like_real_ip(n)]
        self_loops = int((edge_index[0] == edge_index[1]).sum())
        self.self_loops_total += self_loops
        avg_degree = edge_index.shape[1] / max(1, num_nodes)

        is_real_ip_dataset = len(valid_ips) > 0 and len(pseudo_nodes) == 0
        corrupt = False
        if is_real_ip_dataset and num_nodes <= 1:
            logger.warning(
                "[DATA_MAPPING_ERROR] Graph has only %d unique node(s) in a real-IP dataset "
                "(all flows map to same IP). File will be skipped.", num_nodes)
            corrupt = True

        if smoke_test_log and not self._logged_first:
            logger.info(
                "Graph [first window]: nodes=%d, edges=%d, feat_dim=%d, "
                "avg_deg=%.2f, self_loops=%d, real_ips=%d, pseudo_nodes=%d, invalid_ips=%d",
                num_nodes, edge_index.shape[1], EDGE_FEATURE_DIM,
                avg_degree, self_loops, len(valid_ips), len(pseudo_nodes), self.invalid_ips,
            )
            self._logged_first = True

        if corrupt:
            return None

        if not torch.isfinite(edge_attr).all():
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, y=y)
