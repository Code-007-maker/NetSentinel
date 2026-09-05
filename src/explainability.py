"""
Lightweight feature attribution for the 11 flow-level edge features.

Method: input-gradient × value on edge_attr, evaluated at the K-step
World Model attack head (predicted S_{t+K}).  This is NOT SHAP and NOT
GNNExplainer.

Positive signed score: increasing the feature raises predicted attack probability.
Negative signed score: increasing the feature lowers predicted attack probability.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.feature_schema import EDGE_FEATURE_NAMES


def _predict_attack_at_k(encoder, world_model, decoder, x, edge_index, edge_attr, k: int):
    z_t = encoder(x, edge_index, edge_attr)
    z_seq = z_t.unsqueeze(1)
    z_pred_t1, h_n = world_model(z_seq)
    extra = max(k - 1, 0)
    future = world_model.rollout(z_pred_t1, h_n, k=extra) if extra > 0 else []
    z_k = future[-1] if future else z_pred_t1
    attack_prob, _, _ = decoder(z_k)
    return attack_prob


def explain_edge_features(
    encoder: nn.Module,
    world_model: nn.Module,
    decoder: nn.Module,
    data: Data,
    k: int = 3,
    top_n: int = 5,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Gradient × input attribution over the 11 EDGE_FEATURE_NAMES.

    Returns ranked features with signed direction relative to attack probability
    at the t+K rollout step.
    """
    names = list(feature_names or EDGE_FEATURE_NAMES)
    if data.edge_attr is None or data.edge_attr.numel() == 0:
        return {
            "method": "gradient_x_input_edge_features",
            "ranked": [],
            "scores": {n: 0.0 for n in names},
            "directions": {n: "n/a" for n in names},
        }

    encoder.eval()
    world_model.eval()
    decoder.eval()

    x = data.x.detach()
    edge_index = data.edge_index
    edge_attr = data.edge_attr.clone().detach().requires_grad_(True)

    encoder.zero_grad(set_to_none=True)
    world_model.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)

    attack_prob = _predict_attack_at_k(
        encoder, world_model, decoder, x, edge_index, edge_attr, k
    )
    attack_prob.backward()

    if edge_attr.grad is None:
        signed = np.zeros(len(names), dtype=float)
    else:
        # Mean over edges of (grad * value); sign = direction of influence.
        signed = (edge_attr.grad * edge_attr).mean(dim=0).detach().cpu().numpy()

    scores = {names[i]: float(signed[i]) for i in range(min(len(names), signed.shape[0]))}
    ranked_idx = np.argsort(-np.abs(signed))[:top_n]
    ranked = []
    directions = {}
    for i in ranked_idx:
        name = names[int(i)]
        s = float(signed[int(i)])
        if s > 0:
            direction = "increases attack probability"
        elif s < 0:
            direction = "decreases attack probability"
        else:
            direction = "negligible"
        directions[name] = direction
        ranked.append({"feature": name, "score": s, "direction": direction})

    return {
        "method": "gradient_x_input_edge_features",
        "attack_prob": float(attack_prob.detach().cpu().item()),
        "ranked": ranked,
        "scores": scores,
        "directions": directions,
    }


def format_top_features(explanation: Dict, top_n: int = 5) -> str:
    lines = ["Top driving features:"]
    for i, item in enumerate(explanation.get("ranked", [])[:top_n], start=1):
        lines.append(
            f"{i}. {item['feature']}  ({item['direction']}, score={item['score']:+.4g})"
        )
    if len(lines) == 1:
        lines.append("(no edge features available)")
    return "\n".join(lines)
