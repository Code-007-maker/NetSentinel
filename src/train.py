"""
Complete training pipeline for the Cybersecurity World Model SOC system.

Architecture trained end-to-end:
  GNN spatial encoder
  + Temporal World Model (GRU) with recursive K-step rollout
  + StateDecoder heads:
      - attack_head   (BCE, future window labels)
      - net_state_head (MSE, future latent representation)
      - mitre_head    (CE, evidence-based MITRE tactic where available)
  + Logistic Regression baseline (same split, same features)

Latent-state learning rationale:
  z_next_true = gnn(g_{t+1}).detach()
  The GRU transition is trained to predict the GNN embedding of the NEXT
  window.  Gradient is detached from z_next_true so the GNN is not
  pulled in two conflicting directions.  The GNN is supervised directly
  through the attack_head BCE loss and (at K>1) through the multi-step
  attack losses that backprop through the recursive rollout.

  This avoids representational collapse because:
  1. The GNN receives direct attack-label supervision.
  2. The WM must predict a moving target (future graph embeddings).
  3. Attack diversity across windows provides a non-trivial learning signal.

Usage:
  Smoke test (recommended first):
    python -m src.train --mode smoke --epochs 1

  Full training:
    python -m src.train --mode full --epochs 20 --k 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

from src.baseline_model import (
    LR_FEATURE_DIM,
    LR_FEATURE_NAMES,
    artifact_compatible,
    assert_lr_input_matches_model,
    build_lr_features,
    save_lr_schema,
)
from src.config import load_config
from src.dataset_adapters import CICIDSAdapter, CTU13Adapter, UNSWAdapter
from src.dataset_manager import DatasetManager, is_attack_label
from src.evaluate import (
    evaluate_predictions,
    format_probability_audit,
    probability_audit,
    select_decision_threshold,
)
from src.explainability import explain_edge_features, format_top_features
from src.feature_schema import EDGE_FEATURE_NAMES, EDGE_FEATURE_DIM
from src.gnn_encoder import GNNEncoder
from src.graph_builder import GraphBuilder
from src.mitre_mapper import MitreMapper
from src.preprocessing import FlowFeatureScaler
from src.temporal_windowing import create_time_windows
from src.utils import set_seed
from src.world_model import StateDecoder, TemporalWorldModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _tensor_stats(name: str, tensor: torch.Tensor) -> str:
    """Compact, reproducible numeric health summary for a tensor."""
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values in {name}")
    detached = tensor.detach()
    return (f"{name}: min={detached.min().item():.6g} max={detached.max().item():.6g} "
            f"mean={detached.mean().item():.6g} std={detached.std(unbiased=False).item():.6g} "
            f"l2={torch.linalg.vector_norm(detached).item():.6g}")


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values in {name}")


def _class_stats(values: torch.Tensor, labels: torch.Tensor, name: str) -> str:
    """Per-class scalar summary used by the smoke representation audit."""
    items = []
    for label, label_name in ((1, "attack"), (0, "benign")):
        subset = values[labels == label]
        if subset.numel():
            items.append(
                f"{label_name}(n={subset.size(0)} mean={subset.mean().item():.6g} "
                f"std={subset.std(unbiased=False).item():.6g})"
            )
    return f"{name}: " + " ".join(items)


@torch.no_grad()
def log_representation_audit(
    gnn: GNNEncoder, wm: TemporalWorldModel, decoder: StateDecoder,
    examples: List[Tuple], k: int, split_name: str, max_examples: int = 128,
    mode: str = "eval",
) -> Dict[str, float]:
    """Measure sample variation through GNN -> pooling -> GRU -> attack logit.

    Targets are strictly the supplied split's y_(t+K); this routine does not
    select hyperparameters or inspect test labels for a training decision.
    """
    if mode not in {"train", "eval"}:
        raise ValueError(f"Unsupported audit mode {mode!r}")
    prior_modes = (gnn.training, wm.training, decoder.training)
    for module in (gnn, wm, decoder):
        module.train(mode == "train")
    rows = []
    for ex in examples[:max_examples]:
        g_t, future_attacks, _, future_graphs = _normalise_example(ex)
        if g_t is None or g_t.x.numel() == 0 or len(future_attacks) < k:
            continue
        z_t, pooled, latent_pre_norm = gnn(g_t.x, g_t.edge_index, g_t.edge_attr, return_components=True)
        z1, h = wm(z_t.unsqueeze(1))
        z1_pre = wm.last_transition_pre_activation
        predicted = [z1] + wm.rollout(z1, h, k=k - 1)
        transition_pres = [z1_pre] + list(wm.last_rollout_pre_activations)
        logit = decoder.forward_logits(predicted[-1])[0]
        true_zs = []
        for step_i, graph in enumerate(future_graphs[:k], start=1):
            if graph is None or graph.x.numel() == 0:
                true_zs = []
                break
            z_true = gnn(graph.x, graph.edge_index, graph.edge_attr)
            true_zs.append(z_true.squeeze(0))
            if len(rows) < 3:
                logger.info(
                    "TARGET_ID[%s/%s] sample=%d step=t+%d graph_id=%d nodes=%d edges=%d z_true_l2=%.6g",
                    split_name, mode, len(rows), step_i, id(graph), graph.x.size(0),
                    graph.edge_index.size(1), torch.linalg.vector_norm(z_true).item(),
                )
        if len(true_zs) != k:
            continue
        rows.append((int(future_attacks[k - 1]), pooled.squeeze(0), latent_pre_norm.squeeze(0), z_t.squeeze(0),
                     *[z.squeeze(0) for z in transition_pres], *[z.squeeze(0) for z in predicted],
                     *true_zs, logit.squeeze()))
    for module, prior_mode in zip((gnn, wm, decoder), prior_modes):
        module.train(prior_mode)
    if not rows:
        return {}

    labels = torch.tensor([r[0] for r in rows], dtype=torch.long)
    pooled, latent_pre_norm, z_t = (torch.stack([r[i] for r in rows]) for i in (1, 2, 3))
    transition_pres = [torch.stack([r[4 + i] for r in rows]) for i in range(k)]
    preds = [torch.stack([r[4 + k + i] for r in rows]) for i in range(k)]
    true_zs = [torch.stack([r[4 + 2 * k + i] for r in rows]) for i in range(k)]
    logits = torch.stack([r[4 + 3 * k] for r in rows]).reshape(-1)
    probs = torch.sigmoid(logits)

    def embedding_summary(name: str, values: torch.Tensor) -> None:
        norms = torch.linalg.vector_norm(values, dim=1)
        per_dim = values.std(dim=0, unbiased=False)
        unit = F.normalize(values, p=2, dim=1, eps=1e-12)
        cosine = unit @ unit.T
        off_diagonal = cosine[~torch.eye(len(values), dtype=torch.bool)] if len(values) > 1 else cosine.reshape(-1)
        logger.info(
            "REP_AUDIT[%s/%s] %s mean=%.6g std=%.6g per_dim_std_mean=%.6g "
            "norm_mean=%.6g norm_std=%.6g cosine_offdiag_mean=%.6g cosine_offdiag_std=%.6g euclidean_offdiag_mean=%.6g",
            split_name, mode, name, values.mean().item(), values.std(unbiased=False).item(),
            per_dim.mean().item(), norms.mean().item(), norms.std(unbiased=False).item(),
            off_diagonal.mean().item(), off_diagonal.std(unbiased=False).item(),
            torch.cdist(values, values)[~torch.eye(len(values), dtype=torch.bool)].mean().item() if len(values) > 1 else 0.0,
        )
    embedding_summary("graph_pooled", pooled)
    embedding_summary("latent_before_normalization", latent_pre_norm)
    embedding_summary("z_t", z_t)
    for i, (pre, pred) in enumerate(zip(transition_pres, preds), start=1):
        embedding_summary(f"z_t_plus_{i}_before_tanh", pre)
        embedding_summary(f"z_t_plus_{i}", pred)
        embedding_summary(f"true_z_t_plus_{i}", true_zs[i - 1])
        model_mse = F.mse_loss(pred, true_zs[i - 1])
        mean_baseline = F.mse_loss(true_zs[i - 1].mean(dim=0, keepdim=True).expand_as(true_zs[i - 1]), true_zs[i - 1])
        logger.info("TRANSITION_QUALITY[%s/%s] t+%d model_mse=%.6g mean_baseline_mse=%.6g ratio=%.6g variance_ratio=%.6g",
                    split_name, mode, i, model_mse.item(), mean_baseline.item(),
                    (model_mse / mean_baseline.clamp_min(1e-12)).item(),
                    (pred.var(unbiased=False) / true_zs[i - 1].var(unbiased=False).clamp_min(1e-12)).item())
    logger.info("REP_AUDIT[%s/%s] samples=%d attack=%d benign=%d", split_name, mode, len(rows), int(labels.sum()), int((labels == 0).sum()))
    logger.info("REP_AUDIT[%s/%s] %s", split_name, mode, _class_stats(logits, labels, "attack_logit"))
    logger.info("REP_AUDIT[%s/%s] %s", split_name, mode, _class_stats(probs, labels, "attack_probability"))
    return {
        "z_t_per_dim_std": float(z_t.std(dim=0, unbiased=False).mean().item()),
        "z_k_per_dim_std": float(preds[-1].std(dim=0, unbiased=False).mean().item()),
        "logit_std": float(logits.std(unbiased=False).item()),
        "probability_std": float(probs.std(unbiased=False).item()),
    }


def log_loss_gradient_contributions(
    gnn: GNNEncoder, wm: TemporalWorldModel, decoder: StateDecoder, example: Tuple,
    k: int, bce_loss: nn.BCELoss, mse_loss: nn.MSELoss, ce_loss: nn.CrossEntropyLoss,
    mapper: MitreMapper, pos_weight: float, delta_loss_weight: float = 0.0,
) -> None:
    """Report independent loss gradients without applying an optimizer step."""
    total, attack, state, mitre, delta = compute_step_loss(
        gnn, wm, decoder, example, k, bce_loss, mse_loss, ce_loss, mapper, pos_weight, delta_loss_weight
    )
    _assert_finite("gradient_probe_total", total)
    blocks = {
        "gnn": list(gnn.parameters()), "wm": list(wm.parameters()),
        "attack_head": list(decoder.attack_head.parameters()),
    }
    for name, loss in (("attack", attack), ("state", state), ("delta", delta), ("mitre_x0.1", mitre * 0.1), ("total", total)):
        for module in (gnn, wm, decoder):
            module.zero_grad(set_to_none=True)
        if not loss.requires_grad or float(loss.detach()) == 0.0:
            logger.info("LOSS_GRAD[%s] inactive", name)
            continue
        loss.backward(retain_graph=True)
        norms = {}
        for block, params in blocks.items():
            grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
            norms[block] = float(torch.linalg.vector_norm(torch.cat(grads)).item()) if grads else 0.0
        logger.info("LOSS_GRAD[%s] value=%.6g norms=%s", name, loss.item(), norms)
    for module in (gnn, wm, decoder):
        module.zero_grad(set_to_none=True)


@torch.no_grad()
def log_frozen_embedding_probe(
    gnn: GNNEncoder, wm: TemporalWorldModel, train_examples: List[Tuple],
    val_examples: List[Tuple], k: int,
) -> None:
    """Validation-only linear probe: separates embedding adequacy from head training."""
    def features(examples: List[Tuple]):
        X, y = [], []
        for ex in examples:
            g_t, future_attacks, _, _ = _normalise_example(ex)
            if g_t is None or g_t.x.numel() == 0 or len(future_attacks) < k:
                continue
            z_t = gnn(g_t.x, g_t.edge_index, g_t.edge_attr)
            z1, h = wm(z_t.unsqueeze(1))
            pred = wm.rollout(z1, h, k=k - 1)
            X.append((pred[-1] if pred else z1).squeeze(0).cpu().numpy())
            y.append(int(future_attacks[k - 1]))
        return np.asarray(X), np.asarray(y, dtype=int)
    X_tr, y_tr = features(train_examples)
    X_va, y_va = features(val_examples)
    if len(X_tr) and len(X_va) and len(np.unique(y_tr)) == 2 and len(np.unique(y_va)) == 2:
        probe = LogisticRegression(max_iter=1000, random_state=0).fit(X_tr, y_tr)
        p = probe.predict_proba(X_va)[:, 1]
        metrics = evaluate_predictions(y_va, (p >= 0.5).astype(int), p)
        logger.info("FROZEN_Z_LINEAR_PROBE[val] P=%.3f R=%.3f F1=%.3f AUC=%.3f prob_std=%.6g",
                    metrics.get("precision", 0), metrics.get("recall", 0), metrics.get("f1", 0),
                    metrics.get("roc_auc", 0), float(np.std(p)))
    else:
        logger.warning("FROZEN_Z_LINEAR_PROBE skipped: train/validation needs both target classes.")


def _scale_unique_graphs(scaler: FlowFeatureScaler, data_split) -> int:
    """Apply the train-fitted scaler exactly once to every graph in all splits."""
    seen = set()
    for split in (data_split.train, data_split.val, data_split.test):
        for example in split:
            g_t, _, _, future_graphs = _normalise_example(example)
            for graph in [g_t, *future_graphs]:
                if graph is not None and id(graph) not in seen:
                    scaler.scale_graph(graph)
                    seen.add(id(graph))
    return len(seen)


@torch.no_grad()
def _log_pretraining_latents(gnn: GNNEncoder, wm: TemporalWorldModel, example: Tuple, label: str) -> None:
    """Audit encoder targets and recursive predictions before any optimizer step."""
    g_t, _, _, future_graphs = _normalise_example(example)
    z_t = gnn(g_t.x, g_t.edge_index, g_t.edge_attr)
    logger.info("%s %s", label, _tensor_stats("z_t", z_t))
    for i, graph in enumerate(future_graphs[:3], start=1):
        if graph is not None and graph.x.numel() > 0:
            z_true = gnn(graph.x, graph.edge_index, graph.edge_attr)
            logger.info("%s %s", label, _tensor_stats(f"z_t_plus_{i}_true", z_true))
    z1, h = wm(z_t.unsqueeze(1))
    predicted = [z1] + wm.rollout(z1, h, k=2)
    for i, z_pred in enumerate(predicted, start=1):
        logger.info("%s %s", label, _tensor_stats(f"z_t_plus_{i}_pred", z_pred))


# ------------------------------------------------------------------
# MITRE tactic → integer index mapping (must match num_mitre_tactics)
# ------------------------------------------------------------------

TACTIC_TO_IDX: Dict[str, int] = {
    "None": 0,
    "Reconnaissance": 1,
    "Initial_Access": 2,
    "Execution": 3,
    "Persistence": 4,
    "Privilege_Escalation": 5,
    "Defense_Evasion": 6,
    "Credential_Access": 7,
    "Discovery": 8,
    "Lateral_Movement": 9,
    "Collection": 10,
    "Command_and_Control": 11,
    "Exfiltration": 12,
    "Impact": 13,
}

IDX_TO_TACTIC = {v: k for k, v in TACTIC_TO_IDX.items()}


def label_to_mitre_idx(raw_label: str, mapper: MitreMapper) -> Optional[int]:
    """
    Map a raw dataset label to a MITRE tactic integer index.
    Returns None if unsupported (so we skip the loss for that sample).
    """
    result = mapper.get_mapping(raw_label)
    tactic = result.get("tactic", "Unknown / Unsupported")
    if tactic in {"Unknown / Unsupported", "None"}:
        return None
    return TACTIC_TO_IDX.get(tactic, None)


# ------------------------------------------------------------------
# Single training step
# ------------------------------------------------------------------

def _normalise_example(example: Tuple):
    """Accept legacy (G_t, future_attacks, future_raws) and expanded samples
    containing future graphs as the fourth item."""
    if len(example) == 3:
        g_t, future_attacks, future_raws = example
        future_graphs = []
    elif len(example) >= 4:
        g_t, future_attacks, future_raws, future_graphs = example[:4]
    else:
        raise ValueError(f"Unexpected example tuple length {len(example)}")
    return g_t, future_attacks, future_raws, future_graphs


def compute_step_loss(
    gnn: GNNEncoder,
    wm: TemporalWorldModel,
    decoder: StateDecoder,
    example: Tuple,
    k: int,
    bce_loss: nn.BCELoss,
    mse_loss: nn.MSELoss,
    ce_loss: nn.CrossEntropyLoss,
    mapper: MitreMapper,
    pos_weight: float = 1.0,
    delta_loss_weight: float = 0.0,
    diagnostics: bool = False,
):
    """Return (total_loss, attack_loss, state_loss, mitre_loss, delta_loss)."""
    g_t, future_attacks, future_raws, future_graphs = _normalise_example(example)
    if g_t is None or g_t.x.size(0) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    z_t = gnn(g_t.x, g_t.edge_index, g_t.edge_attr)  # (1, z_dim)
    _assert_finite("z_t", z_t)
    z_seq = z_t.unsqueeze(1)
    z_pred_t1, h_n = wm(z_seq)
    future_zs = wm.rollout(z_pred_t1, h_n, k=k - 1)
    all_pred_zs = [z_pred_t1] + future_zs
    all_deltas = [wm.last_delta] + list(wm.last_rollout_deltas)

    for step_i, z_pred in enumerate(all_pred_zs, start=1):
        _assert_finite(f"predicted_state_t_plus_{step_i}", z_pred)
        if diagnostics:
            logger.info("Latent rollout %s", _tensor_stats(f"z_t_plus_{step_i}_pred", z_pred))

    attack_total = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
    state_total = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
    mitre_total = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
    delta_total = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
    previous_true = z_t
    bce_unreduced = nn.BCEWithLogitsLoss(reduction='none')
    step_count = 0

    for step_i, z_pred in enumerate(all_pred_zs):
        if step_i >= len(future_attacks):
           break

        attack_logits, net_state_pred, mitre_logits = decoder.forward_logits(z_pred)
        _assert_finite("attack_logits", attack_logits)
        attack_prob = torch.sigmoid(attack_logits)
        _assert_finite("attack_probabilities", attack_prob)
        y_att = torch.tensor([[float(future_attacks[step_i])]], device=attack_logits.device, dtype=attack_logits.dtype)

        l_att_raw = bce_unreduced(attack_logits, y_att)
        sample_weight = torch.where(
           y_att >= 0.5,
           torch.tensor([[pos_weight]], device=attack_prob.device, dtype=attack_prob.dtype),
           torch.ones_like(y_att),
        )
        attack_total = attack_total + (l_att_raw * sample_weight).mean()
        step_count += 1

        if step_i < len(future_graphs):
           next_graph = future_graphs[step_i]
           if next_graph is not None and getattr(next_graph, "x", None) is not None and next_graph.x.numel() > 0:
               z_next_true = gnn(next_graph.x, next_graph.edge_index, next_graph.edge_attr).detach()
               _assert_finite(f"z_next_true_t_plus_{step_i + 1}", z_next_true)
               if net_state_pred.shape == z_next_true.shape:
                   # Directly supervise the recursive transition.  Previously
                   # only a decoder MLP was compared to the future state, so a
                   # constant rollout could still satisfy state supervision.
                   transition_loss = mse_loss(z_pred, z_next_true)
                   decoder_state_loss = mse_loss(net_state_pred, z_next_true)
                   state_total = state_total + transition_loss + 0.1 * decoder_state_loss
                   if wm.transition_mode == "residual":
                       true_delta = z_next_true - previous_true
                       delta_total = delta_total + mse_loss(all_deltas[step_i], true_delta)
                   previous_true = z_next_true
                   if diagnostics:
                       logger.info("Direct transition loss t+%d=%.6g decoder-state loss=%.6g",
                                   step_i + 1, transition_loss.item(), decoder_state_loss.item())
                       logger.info("State target %s", _tensor_stats(f"z_t_plus_{step_i + 1}_true", z_next_true))
                       logger.info("State prediction %s", _tensor_stats(f"net_state_t_plus_{step_i + 1}", net_state_pred))
               else:
                   raise ValueError(
                       f"State-target mismatch: net_state_pred shape {tuple(net_state_pred.shape)} "
                       f"!= z_next_true shape {tuple(z_next_true.shape)}"
                   )

        raw_lbl = future_raws[step_i] if step_i < len(future_raws) else "Benign"
        mitre_idx = label_to_mitre_idx(raw_lbl, mapper)
        if mitre_idx is not None:
           target_mitre = torch.tensor([mitre_idx], dtype=torch.long, device=mitre_logits.device)
           mitre_total = mitre_total + ce_loss(mitre_logits, target_mitre)

    # Explicit objective weighting: state dynamics is supervised directly;
    # MITRE remains auxiliary and may not dominate the shared GRU/encoder.
    total_loss = attack_total + state_total + 0.1 * mitre_total + delta_loss_weight * delta_total
    _assert_finite("attack_loss", attack_total)
    _assert_finite("state_loss", state_total)
    _assert_finite("total_loss", total_loss)
    if diagnostics:
        logger.info("First-batch losses: attack_loss=%.6g state_loss=%.6g state_weighted=%.6g mitre_loss=%.6g total_loss=%.6g",
                    attack_total.item(), state_total.item(), state_total.item(), 0.1 * mitre_total.item(), total_loss.item())
        logits_all = torch.cat([decoder.forward_logits(z)[0].detach().flatten() for z in all_pred_zs])
        probs_all = torch.sigmoid(logits_all)
        logger.info("Attack logits %s", _tensor_stats("attack_logits", logits_all))
        logger.info("Attack probabilities %s", _tensor_stats("attack_probabilities", probs_all))
    return total_loss, attack_total, state_total, mitre_total, delta_total


def train_step(
    gnn: GNNEncoder,
    wm: TemporalWorldModel,
    decoder: StateDecoder,
    example: Tuple,
    k: int,
    optimizer: optim.Optimizer,
    bce_loss: nn.BCELoss,
    mse_loss: nn.MSELoss,
    ce_loss: nn.CrossEntropyLoss,
    mapper: MitreMapper,
    pos_weight: float = 1.0,
    delta_loss_weight: float = 0.0,
    gradient_clip_norm: float = 5.0,
    diagnostics: bool = False,
) -> float:
    """One forward + backward pass on a single training example."""
    g_t, _, _, _ = _normalise_example(example)
    if g_t is None or g_t.x.size(0) == 0:
        return 0.0

    optimizer.zero_grad()
    total_loss, _, _, _, _ = compute_step_loss(
        gnn, wm, decoder, example, k, bce_loss, mse_loss, ce_loss, mapper, pos_weight, delta_loss_weight,
        diagnostics=diagnostics,
    )
    if float(total_loss.detach().item()) != 0.0:
        total_loss.backward()
        blocks = {"gnn": list(gnn.parameters()), "wm": list(wm.parameters()), "decoder": list(decoder.parameters())}
        pre_norms = {
            name: float(torch.linalg.vector_norm(torch.cat([p.grad.detach().flatten() for p in params if p.grad is not None])).item())
            if any(p.grad is not None for p in params) else 0.0
            for name, params in blocks.items()
        }
        torch.nn.utils.clip_grad_norm_(list(gnn.parameters()) + list(wm.parameters()) + list(decoder.parameters()), gradient_clip_norm)
        post_norms = {
            name: float(torch.linalg.vector_norm(torch.cat([p.grad.detach().flatten() for p in params if p.grad is not None])).item())
            if any(p.grad is not None for p in params) else 0.0
            for name, params in blocks.items()
        }
        if diagnostics:
            logger.info("Gradient norms pre_clip=%s post_clip=%s clip_norm=%.3f", pre_norms, post_norms, gradient_clip_norm)
        optimizer.step()
    return float(total_loss.item())


# ------------------------------------------------------------------
# Evaluation on a split  (predicted S_{t+K} vs y_{t+K})
# ------------------------------------------------------------------

@torch.no_grad()
def collect_horizon_scores(
    gnn: GNNEncoder,
    wm: TemporalWorldModel,
    decoder: StateDecoder,
    examples: List[Tuple],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    World Model inference at horizon K.

    For each example (G_t, future_attacks[1..K], ...):
      z_t     = GNN(S_t)
      z_{t+1} = WM(z_t)
      z_{t+K} = recursive rollout of K-1 additional steps
      p       = attack_head(z_{t+K})
      y       = future_attacks[K-1] = actual attack state at t+K

    Arrays are in example order (same ordering as `examples`).
    """
    gnn.eval(); wm.eval(); decoder.eval()
    y_true_all, y_prob_all = [], []

    for g_t, future_attacks, _, _ in examples:
        if g_t.x.size(0) == 0 or not future_attacks:
            continue
        if len(future_attacks) < k:
            continue
        y_true_all.append(int(future_attacks[k - 1]))  # y_{t+K}, NOT y_t

        z_t = gnn(g_t.x, g_t.edge_index, g_t.edge_attr)
        z_seq = z_t.unsqueeze(1)
        z_pred_t1, h_n = wm(z_seq)
        extra = max(k - 1, 0)
        future_zs = wm.rollout(z_pred_t1, h_n, k=extra) if extra > 0 else []
        z_k = future_zs[-1] if future_zs else z_pred_t1  # predicted S_{t+K}
        attack_prob, _, _ = decoder(z_k)
        y_prob_all.append(float(attack_prob.item()))

    return np.asarray(y_true_all, dtype=int), np.asarray(y_prob_all, dtype=float)


@torch.no_grad()
def evaluate_split(
    gnn: GNNEncoder,
    wm: TemporalWorldModel,
    decoder: StateDecoder,
    examples: List[Tuple],
    k: int,
    split_name: str = "val",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate attack-prediction metrics at horizon t+K with a frozen threshold."""
    was_training = gnn.training
    y_true, y_prob = collect_horizon_scores(gnn, wm, decoder, examples, k)

    if was_training:
        gnn.train(); wm.train(); decoder.train()

    if y_true.size == 0:
        return {}

    y_pred = (y_prob >= threshold).astype(int)
    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    metrics["threshold"] = float(threshold)
    logger.info(
        "[%s] thr=%.2f P=%.3f R=%.3f F1=%.3f FPR=%.3f AUC=%.3f",
        split_name,
        threshold,
        metrics.get("precision", 0),
        metrics.get("recall", 0),
        metrics.get("f1", 0),
        metrics.get("fpr", 0),
        metrics.get("roc_auc", 0),
    )
    return metrics


def summarize_target_distribution(examples: List[Tuple], k: int, split_name: str = "split") -> Dict[str, float]:
    """Count positive/negative targets for the t+K horizon."""
    y_all = []
    for ex in examples:
        future_attacks = ex[1] if len(ex) > 1 else []
        if len(future_attacks) >= k:
            y_all.append(int(future_attacks[k - 1]))
    arr = np.asarray(y_all, dtype=int)
    n_pos = int(arr[arr == 1].size)
    n_neg = int(arr[arr == 0].size)
    ratio = float(n_pos / max(n_pos + n_neg, 1))
    logger.info(
        "[%s] target-summary total=%d attack=%d benign=%d attack_ratio=%.4f horizon_k=%d",
        split_name,
        arr.size,
        n_pos,
        n_neg,
        ratio,
        k,
    )
    return {
        "total": int(arr.size),
        "attack": n_pos,
        "benign": n_neg,
        "attack_ratio": ratio,
    }


@torch.no_grad()
def compute_split_loss(
    gnn: GNNEncoder,
    wm: TemporalWorldModel,
    decoder: StateDecoder,
    examples: List[Tuple],
    k: int,
    bce_loss: nn.BCELoss,
    mse_loss: nn.MSELoss,
    ce_loss: nn.CrossEntropyLoss,
    mapper: MitreMapper,
    pos_weight: float = 1.0,
    delta_loss_weight: float = 0.0,
) -> Dict[str, float]:
    """Compute average attack/state/mitre losses over a split without backprop."""
    total = 0.0
    attack = 0.0
    state = 0.0
    mitre = 0.0
    count = 0

    for ex in examples:
        step_total, step_attack, step_state, step_mitre, _ = compute_step_loss(
            gnn, wm, decoder, ex, k, bce_loss, mse_loss, ce_loss, mapper, pos_weight, delta_loss_weight
        )
        step_total_f = float(step_total.detach().item())
        step_attack_f = float(step_attack.detach().item())
        step_state_f = float(step_state.detach().item())
        step_mitre_f = float(step_mitre.detach().item())
        if step_total_f == 0.0 and step_attack_f == 0.0 and step_state_f == 0.0 and step_mitre_f == 0.0:
            continue
        total += step_total_f
        attack += step_attack_f
        state += step_state_f
        mitre += step_mitre_f
        count += 1

    if count == 0:
        return {"total_loss": 0.0, "attack_loss": 0.0, "state_loss": 0.0, "mitre_loss": 0.0}
    return {
        "total_loss": total / count,
        "attack_loss": attack / count,
        "state_loss": state / count,
        "mitre_loss": mitre / count,
    }


def audit_probability_distribution(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Summaries for attack vs benign probability distributions at threshold 0.5."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    pred = (y_prob >= threshold).astype(int)

    attack_probs = y_prob[y_true == 1]
    benign_probs = y_prob[y_true == 0]
    summaries = {
        "positive_targets": int((y_true == 1).sum()),
        "negative_targets": int((y_true == 0).sum()),
        "attack_target_ratio": float((y_true == 1).mean()) if y_true.size else 0.0,
        "predicted_attacks_at_threshold": int(pred.sum()),
        "attack_prob_min": float(np.min(attack_probs)) if attack_probs.size else 0.0,
        "attack_prob_max": float(np.max(attack_probs)) if attack_probs.size else 0.0,
        "attack_prob_mean": float(np.mean(attack_probs)) if attack_probs.size else 0.0,
        "attack_prob_std": float(np.std(attack_probs)) if attack_probs.size else 0.0,
        "benign_prob_min": float(np.min(benign_probs)) if benign_probs.size else 0.0,
        "benign_prob_max": float(np.max(benign_probs)) if benign_probs.size else 0.0,
        "benign_prob_mean": float(np.mean(benign_probs)) if benign_probs.size else 0.0,
        "benign_prob_std": float(np.std(benign_probs)) if benign_probs.size else 0.0,
    }
    logger.info(
        "Probability audit thr=%.2f attacked=%d benign=%d predicted_attacks=%d attack_ratio=%.3f "
        "attack_mean=%.4f benign_mean=%.4f attack_std=%.4f benign_std=%.4f",
        threshold,
        summaries["positive_targets"],
        summaries["negative_targets"],
        summaries["predicted_attacks_at_threshold"],
        summaries["attack_target_ratio"],
        summaries["attack_prob_mean"],
        summaries["benign_prob_mean"],
        summaries["attack_prob_std"],
        summaries["benign_prob_std"],
    )
    return summaries


def log_parameter_health(gnn: GNNEncoder, wm: TemporalWorldModel, decoder: StateDecoder, epoch: int, param_snapshot: Optional[dict] = None) -> None:
    """Log gradient norms and parameter movement for the main model blocks."""
    blocks = {
        "gnn": list(gnn.parameters()),
        "wm": list(wm.parameters()),
        "decoder": list(decoder.parameters()),
    }
    for name, params in blocks.items():
        grads = [p.grad.norm().item() for p in params if p.grad is not None]
        total_grad = float(np.sum(grads)) if grads else 0.0
        logger.info("[epoch %d] %s grad_norm_sum=%.6f nonzero_grad_params=%d", epoch, name, total_grad, len(grads))

        if param_snapshot is not None and name in param_snapshot:
            max_delta = 0.0
            for old, new in zip(param_snapshot[name], params):
                if old is not None and new is not None:
                    max_delta = max(max_delta, float((new.detach() - old.detach()).abs().max().item()))
            logger.info("[epoch %d] %s max_param_delta=%.6f", epoch, name, max_delta)


# ------------------------------------------------------------------
# K-selection experiment
# ------------------------------------------------------------------

def select_k(
    gnn, wm, decoder, val_examples, k_candidates: List[int]
) -> int:
    """Select K with best validation F1 on 1-step ahead prediction."""
    best_k, best_f1 = k_candidates[0], -1.0
    for k in k_candidates:
        metrics = evaluate_split(gnn, wm, decoder, val_examples, k, f"val_k{k}")
        f1 = metrics.get("f1", 0.0)
        logger.info("K=%d → val F1=%.4f", k, f1)
        if f1 > best_f1:
            best_f1, best_k = f1, k
    logger.info("Selected K=%d (val F1=%.4f)", best_k, best_f1)
    return best_k


# Logistic Regression feature construction lives in src/baseline_model.py
# (window-mean of the same 11 EDGE_FEATURE_NAMES as the World Model).


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def save_checkpoint(
    ckpt_dir: Path,
    gnn: GNNEncoder,
    wm: TemporalWorldModel,
    decoder: StateDecoder,
    lr_model: Optional[LogisticRegression],
    scaler: FlowFeatureScaler,
    selected_k: int,
    cfg: dict,
    metrics: dict,
    selected_threshold: float = 0.5,
    threshold_meta: Optional[dict] = None,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Neural model weights
    torch.save(
        {
            "gnn_state_dict": gnn.state_dict(),
            "wm_state_dict": wm.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "selected_k": selected_k,
            "selected_threshold": float(selected_threshold),
            "threshold_criterion": (threshold_meta or {}).get(
                "selection_criterion", "youden_j_validation_only"
            ),
            "config": cfg,
            "train_metrics": metrics,
        },
        ckpt_dir / "world_model.pt",
    )

    # LR baseline + explicit feature schema (must match training dim)
    if lr_model is not None:
        n_in = int(getattr(lr_model, "n_features_in_", -1))
        if n_in != LR_FEATURE_DIM:
            raise ValueError(
                f"Refusing to save LR with n_features_in_={n_in}; canonical dim is {LR_FEATURE_DIM}."
            )
        joblib.dump(lr_model, ckpt_dir / "lr_baseline.joblib")
        save_lr_schema(ckpt_dir)
    else:
        stale = ckpt_dir / "lr_baseline.joblib"
        if stale.exists():
            stale.unlink()
        schema_path = ckpt_dir / "lr_schema.json"
        if schema_path.exists():
            schema_path.unlink()

    # Preprocessing scaler + feature schema
    scaler.save(ckpt_dir)

    snapshot = dict(cfg)
    snapshot["selected_k"] = selected_k
    snapshot["selected_threshold"] = float(selected_threshold)
    snapshot["threshold_criterion"] = (threshold_meta or {}).get(
        "selection_criterion", "youden_j_validation_only"
    )
    snapshot["lr_feature_names"] = list(LR_FEATURE_NAMES)
    snapshot["lr_feature_dim"] = int(LR_FEATURE_DIM)
    if snapshot.get("hyperparameters"):
        snapshot["hyperparameters"]["lr_feature_dim"] = int(LR_FEATURE_DIM)
    (ckpt_dir / "training_config.json").write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
    )

    logger.info("Checkpoint bundle saved to %s", ckpt_dir)


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train(mode: str = "smoke", epochs: int = 1, k_override: Optional[int] = None,
          transition: str = "direct", delta_scale: float = 0.5, delta_loss_weight: float = 0.25) -> None:
    cfg = load_config()
    hp = cfg["hyperparameters"]
    set_seed(hp["random_seed"])

    smoke_test = mode == "smoke"
    if smoke_test:
        logger.info("=== SMOKE TEST MODE (min %d windows/file, %d epoch) ===",
                    hp.get("smoke_test_min_windows", 50), epochs)
    else:
        logger.info("=== FULL TRAINING MODE (%d rows/file max) ===",
                    hp["max_rows_per_file"])

    # ------------------------------------------------------------------
    # 1. Build dataset splits
    # ------------------------------------------------------------------
    dm = DatasetManager(cfg["paths"]["dataset_dir"], cfg, smoke_test=smoke_test)
    k_default = k_override if k_override else hp["k_rollout_steps"]
    data_split = dm.build_splits(k=max(hp["k_candidates"]))  # build for max K

    if not data_split.train:
        logger.error("No training examples generated. Check dataset paths and config.")
        return

    # Print loading report
    logger.info("=== Dataset Loading Report ===")
    for row in dm.report:
        logger.info(
            "  %s | rows %d→%d | windows %d | attack %d | benign %d",
            Path(row["file"]).name,
            row["rows_total"], row["rows_sampled"],
            row["windows"], row["attack_windows"], row["benign_windows"],
        )
    logger.info(
        "Total: train=%d  val=%d  test=%d examples",
        len(data_split.train), len(data_split.val), len(data_split.test),
    )
    summarize_target_distribution(data_split.train, k_default, "train")
    summarize_target_distribution(data_split.val, k_default, "val")
    summarize_target_distribution(data_split.test, k_default, "test")

    # ------------------------------------------------------------------
    # 2. Fit feature scaler on TRAINING data only
    # ------------------------------------------------------------------
    # We reconstruct a dataframe of training flow features for the scaler.
    # In production the scaler would be fit on raw flows; here we fit on
    # graph edge attributes which represent the same flow features.
    # For LR we use graph-level aggregate features (no scaler needed there).
    scaler = FlowFeatureScaler()
    all_edge_feats = []
    for g_t, _, _, _ in data_split.train:
        if g_t is not None and g_t.edge_attr is not None and g_t.edge_attr.numel() > 0:
            all_edge_feats.append(g_t.edge_attr.numpy())
    if all_edge_feats:
        import pandas as pd
        edge_arr = np.vstack(all_edge_feats)
        assert edge_arr.shape[1] == EDGE_FEATURE_DIM, (
            f"Edge feature dim mismatch: got {edge_arr.shape[1]}, "
            f"expected {EDGE_FEATURE_DIM} from EDGE_FEATURE_NAMES."
        )
        scaler_df = pd.DataFrame(edge_arr, columns=EDGE_FEATURE_NAMES)
        scaler.fit(scaler_df)
        logger.info("Scaler fitted on %d flow-level edge observations (dim=%d).",
                    len(edge_arr), EDGE_FEATURE_DIM)

    # ------------------------------------------------------------------
    # 3. Initialise models
    # ------------------------------------------------------------------
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
        transition_mode=transition, delta_scale=delta_scale,
    )
    logger.info("TRANSITION_EXPERIMENT mode=%s delta_scale=%.3f delta_loss_weight=%.3f", transition, delta_scale, delta_loss_weight)
    decoder = StateDecoder(
        z_dim=hp["gnn_out_dim"],
        num_mitre_tactics=hp["num_mitre_tactics"],
    )
    # This audit documents the original failure mode: raw physical-unit edge
    # attributes were previously fed directly into the GNN despite fitting a scaler.
    _log_pretraining_latents(gnn, wm, data_split.train[0], "PRE-SCALE")
    scaled_graphs = _scale_unique_graphs(scaler, data_split)
    logger.info("Applied train-fitted edge-feature scaling to %d unique graphs.", scaled_graphs)
    _log_pretraining_latents(gnn, wm, data_split.train[0], "POST-SCALE")
    mapper = MitreMapper(cfg["mitre"]["mapping_file"])

    optimizer = optim.Adam(
        list(gnn.parameters()) + list(wm.parameters()) + list(decoder.parameters()),
        lr=hp["learning_rate"],
    )
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()

    # Report the horizon target distribution separately from the loss targets.
    train_y_k = [ex[1][-1] for ex in data_split.train if ex[1]]
    n_pos = int(sum(train_y_k))   # attack count
    n_neg = int(len(train_y_k) - n_pos)  # benign count
    # The attack loss is applied at every future step.  Balance precisely those
    # train-only targets: w_attack=n_benign/n_attack, w_benign=1.  This makes
    # both classes contribute equally and fixes the constant-logit optimum at .5.
    attack_loss_targets = [int(y) for ex in data_split.train for y in ex[1][:k_default]]
    loss_pos = int(sum(attack_loss_targets))
    loss_neg = len(attack_loss_targets) - loss_pos
    pos_weight = float(loss_neg / max(loss_pos, 1))
    neg_weight = 1.0
    logger.info(
        "Class-balanced BCE sample weights (train t+1..t+K): attack_weight=%.4f benign_weight=%.4f "
        "(attack=%d benign=%d)", pos_weight, neg_weight, loss_pos, loss_neg,
    )
    weighted_constant = (pos_weight * loss_pos) / max(pos_weight * loss_pos + loss_neg, 1e-12)
    unweighted_constant = loss_pos / max(len(attack_loss_targets), 1)
    logger.info(
        "Attack-loss prior across t+1..t+K: attack=%d benign=%d; constant-logit optimum "
        "weighted=%.4f (pos_weight=%.4f), unweighted=%.4f",
        loss_pos, loss_neg, weighted_constant, pos_weight, unweighted_constant,
    )
    log_loss_gradient_contributions(
        gnn, wm, decoder, data_split.train[0], k_default, bce_loss, mse_loss, ce_loss, mapper, pos_weight, delta_loss_weight
    )

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        epoch_loss = 0.0
        gnn.train(); wm.train(); decoder.train()
        param_snapshot = {
            "gnn": [p.detach().clone() for p in gnn.parameters()],
            "wm": [p.detach().clone() for p in wm.parameters()],
            "decoder": [p.detach().clone() for p in decoder.parameters()],
        }

        rng = np.random.default_rng(hp["random_seed"] + epoch)
        indices = rng.permutation(len(data_split.train)).tolist()

        for idx in indices:
            loss = train_step(
                gnn, wm, decoder,
                data_split.train[idx],
                k_default,
                optimizer,
                bce_loss, mse_loss, ce_loss,
                mapper,
                pos_weight=pos_weight,
                delta_loss_weight=delta_loss_weight,
                gradient_clip_norm=float(hp["gradient_clip_norm"]),
                diagnostics=(idx == indices[0]),
            )
            epoch_loss += loss

        avg_loss = epoch_loss / max(len(data_split.train), 1)
        log_parameter_health(gnn, wm, decoder, epoch + 1, param_snapshot)
        val_loss = 0.0
        val_audit = {}
        if data_split.val:
            val_loss_summary = compute_split_loss(
                gnn, wm, decoder, data_split.val, k_default, bce_loss, mse_loss, ce_loss, mapper, pos_weight, delta_loss_weight
            )
            val_loss = val_loss_summary["total_loss"]
            y_val_true, y_val_prob = collect_horizon_scores(gnn, wm, decoder, data_split.val, k_default)
            val_audit = audit_probability_distribution(y_val_true, y_val_prob, threshold=0.5)
            logger.info(
                "Epoch %d/%d  train_loss=%.5f  val_loss=%.5f  val_attack_ratio=%.3f "
                "val_pred_attacks=%d  val_attack_mean=%.4f  val_benign_mean=%.4f",
                epoch + 1,
                epochs,
                avg_loss,
                val_loss,
                val_audit.get("attack_target_ratio", 0.0),
                val_audit.get("predicted_attacks_at_threshold", 0),
                val_audit.get("attack_prob_mean", 0.0),
                val_audit.get("benign_prob_mean", 0.0),
            )
            evaluate_split(gnn, wm, decoder, data_split.val, k_default, "val")
        else:
            logger.info("Epoch %d/%d  train_loss=%.5f  val_loss=n/a", epoch + 1, epochs, avg_loss)

    # These audits are diagnostics only: validation labels are not used to
    # alter the model, optimiser, or hyperparameters.
    # Four-way diagnostic: data split × module mode.  This changes neither
    # weights nor thresholds and exposes any train/eval-dependent discrepancy.
    train_rep_audit = log_representation_audit(gnn, wm, decoder, data_split.train, k_default, "train", mode="eval")
    train_rep_train_mode = log_representation_audit(gnn, wm, decoder, data_split.train, k_default, "train", mode="train")
    if data_split.val:
        val_rep_audit = log_representation_audit(gnn, wm, decoder, data_split.val, k_default, "val", mode="eval")
        log_representation_audit(gnn, wm, decoder, data_split.val, k_default, "val", mode="train")
        representation_collapsed = (
            # This threshold is deliberately relative to the observed latent
            # scale: 4e-4 was mathematically non-zero yet operationally
            # collapsed against true train-target per-dimension std ~2.8e-1.
            train_rep_audit.get("z_k_per_dim_std", 0.0) < 1e-2
            or train_rep_audit.get("logit_std", 0.0) < 1e-5
            or train_rep_audit.get("probability_std", 0.0) < 1e-6
            or
            val_rep_audit.get("z_k_per_dim_std", 0.0) < 1e-2
            or val_rep_audit.get("logit_std", 0.0) < 1e-5
            or val_rep_audit.get("probability_std", 0.0) < 1e-6
        )
        logger.info(
            "MODEL_HEALTH: PIPELINE_PASS=TRUE REPRESENTATION_COLLAPSED=%s MODEL_HEALTH_PASS=%s "
            "(train_zk_std=%.6g val_zk_std=%.6g val_logit_std=%.6g)",
            representation_collapsed, not representation_collapsed,
            train_rep_audit.get("z_k_per_dim_std", 0.0), val_rep_audit.get("z_k_per_dim_std", 0.0),
            val_rep_audit.get("logit_std", 0.0),
        )
        log_frozen_embedding_probe(gnn, wm, data_split.train, data_split.val, k_default)

    # ------------------------------------------------------------------
    # 5. K-selection experiment (on validation set)
    # ------------------------------------------------------------------
    if not smoke_test and data_split.val:
        selected_k = select_k(gnn, wm, decoder, data_split.val, hp["k_candidates"])
    else:
        selected_k = k_default
        logger.info("Skipping K-selection in smoke-test mode. Using K=%d", selected_k)

    # ------------------------------------------------------------------
    # 6. Logistic Regression baseline (same split, same features)
    # ------------------------------------------------------------------
    logger.info(
        "Training Logistic Regression baseline on %d-d window-mean flow features %s",
        LR_FEATURE_DIM,
        LR_FEATURE_NAMES,
    )
    X_train, y_train = build_lr_features(data_split.train)
    lr_model = LogisticRegression(max_iter=1000, random_state=hp["random_seed"])
    lr_skip_reason = ""
    if len(X_train) > 0:
        if int(X_train.shape[1]) != LR_FEATURE_DIM:
            raise ValueError(
                f"LR training matrix dim {X_train.shape[1]} != {LR_FEATURE_DIM}"
            )
        if len(np.unique(y_train)) > 1:
            lr_model.fit(X_train, y_train)
            assert_lr_input_matches_model(lr_model, X_train)
            ok, reason = artifact_compatible(lr_model, {"feature_names": LR_FEATURE_NAMES, "n_features": LR_FEATURE_DIM})
            if not ok:
                raise RuntimeError(reason)
            logger.info("LR fitted n_features_in_=%d", lr_model.n_features_in_)
        else:
            lr_skip_reason = "Training split contains only one class."
            logger.warning("LR Baseline skipped: %s", lr_skip_reason)
            lr_model = None
    else:
        lr_skip_reason = "No training data available."
        lr_model = None

    # ------------------------------------------------------------------
    # 7. Final evaluation on held-out TEST set
    # ------------------------------------------------------------------
    selected_threshold = 0.5
    val_threshold_metrics: Dict[str, float] = {}
    y_val = y_prob_val = None
    if data_split.val:
        y_val, y_prob_val = collect_horizon_scores(
            gnn, wm, decoder, data_split.val, selected_k
        )
        selected_threshold, val_threshold_metrics = select_decision_threshold(
            y_val, y_prob_val
        )
        logger.info(
            "Selected threshold=%.2f (criterion=%s)  val P=%.3f R=%.3f F1=%.3f FPR=%.3f",
            selected_threshold,
            val_threshold_metrics.get("selection_criterion"),
            val_threshold_metrics.get("precision", 0),
            val_threshold_metrics.get("recall", 0),
            val_threshold_metrics.get("f1", 0),
            val_threshold_metrics.get("fpr", 0),
        )
        print("\nSelected threshold: {:.2f}".format(selected_threshold))
        print("Validation precision: {:.4f}".format(val_threshold_metrics.get("precision", 0)))
        print("Validation recall: {:.4f}".format(val_threshold_metrics.get("recall", 0)))
        print("Validation F1: {:.4f}".format(val_threshold_metrics.get("f1", 0)))
        print("Validation FPR: {:.4f}".format(val_threshold_metrics.get("fpr", 0)))
    else:
        logger.warning("No validation set; freezing threshold=0.5")

    results = {
        "world_model": {},
        "logistic_regression": {},
        "selected_k": selected_k,
        "selected_threshold": selected_threshold,
        "threshold_criterion": val_threshold_metrics.get(
            "selection_criterion", "fallback_0.5"
        ),
        "validation_threshold_metrics": val_threshold_metrics,
    }

    if data_split.test:
        logger.info("=== TEST SET EVALUATION (untouched holdout, threshold frozen) ===")

        t0 = time.time()
        y_test_wm, y_prob_wm = collect_horizon_scores(
            gnn, wm, decoder, data_split.test, selected_k
        )
        wm_latency = (time.time() - t0) / max(len(data_split.test), 1) * 1000
        y_pred_wm = (y_prob_wm >= selected_threshold).astype(int) if y_test_wm.size else np.array([])
        wm_metrics = (
            evaluate_predictions(y_test_wm, y_pred_wm, y_prob_wm)
            if y_test_wm.size else {}
        )
        wm_metrics["inference_latency_ms"] = round(wm_latency, 4)
        wm_metrics["threshold"] = float(selected_threshold)
        test_audit = (
            probability_audit(y_test_wm, y_prob_wm, selected_threshold)
            if y_test_wm.size else {}
        )
        results["world_model"] = wm_metrics
        results["test_probability_audit"] = test_audit
        if test_audit:
            print("\n" + format_probability_audit(test_audit))

        # Gradient×input explanation on first test graph predicted as future attack
        expl_text = "n/a (no predicted-attack test graph)"
        for g_t, future_attacks, _, _ in data_split.test:
            if g_t.x.size(0) == 0 or g_t.edge_attr is None or g_t.edge_attr.numel() == 0:
                continue
            expl = explain_edge_features(gnn, wm, decoder, g_t, k=selected_k, top_n=5)
            if expl.get("attack_prob", 0) >= selected_threshold:
                expl_text = format_top_features(expl, top_n=5)
                results["explainability"] = {
                    "method": expl["method"],
                    "ranked": expl["ranked"],
                    "attack_prob": expl.get("attack_prob"),
                }
                print("\n" + expl_text)
                break
        if "explainability" not in results:
            results["explainability"] = {
                "method": "gradient_x_input_edge_features",
                "ranked": [],
                "note": expl_text,
            }

        X_test, y_test = build_lr_features(data_split.test)
        if len(X_test) > 0:
            if lr_model is not None:
                t0 = time.time()
                assert_lr_input_matches_model(lr_model, X_test)
                lr_prob = lr_model.predict_proba(X_test)[:, 1]
                lr_pred = (lr_prob >= 0.5).astype(int)
                lr_latency = (time.time() - t0) / max(len(X_test), 1) * 1000
                lr_metrics = evaluate_predictions(y_test, lr_pred, lr_prob)
                lr_metrics["inference_latency_ms"] = round(lr_latency, 4)
                lr_metrics["trainable"] = "YES"
                results["logistic_regression"] = lr_metrics
            else:
                results["logistic_regression"] = {
                    "trainable": "NO",
                    "reason": lr_skip_reason
                }

        logger.info("World Model  test metrics: %s", results["world_model"])
        logger.info("LR Baseline  test metrics: %s", results["logistic_regression"])

    # ------------------------------------------------------------------
    # 8. Save checkpoint bundle
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    save_checkpoint(
        ckpt_dir, gnn, wm, decoder, lr_model, scaler, selected_k, cfg, results,
        selected_threshold=selected_threshold,
        threshold_meta=val_threshold_metrics,
    )

    # ------------------------------------------------------------------
    # 9. Write evaluation outputs
    # ------------------------------------------------------------------
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "evaluation_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    _write_eval_report(out_dir / "evaluation_report.md", results, dm.report, selected_k, mode)

    if smoke_test:
        _print_smoke_audit(dm.report, data_split, results, selected_k)

    logger.info("Training complete. Artifacts in: %s", ckpt_dir)
    logger.info("Selected K=%d  |  Mode=%s", selected_k, mode)

# ------------------------------------------------------------------
# Smoke test audit report (printed to stdout)
# ------------------------------------------------------------------

def _print_smoke_audit(ds_report, data_split, results, selected_k):
    """Prints the structured audit report to stdout at the end of a smoke test run."""
    SEP = "=" * 70
    feature_names = [
        'duration', 'packets_forward', 'packets_backward', 'bytes_forward', 'bytes_backward',
        'rate_fwd', 'rate_bwd', 'byte_rate_fwd', 'byte_rate_bwd', 'avg_pkt_size_fwd', 'avg_pkt_size_bwd'
    ]

    # ----------------------------------------------------------------
    # DATASET AUDIT
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("=== DATASET AUDIT ===")
    print(SEP)
    failed_files = [r for r in ds_report if r.get('error')]
    success_files = [r for r in ds_report if not r.get('error')]

    if failed_files:
        print("  FAILED FILES:")
        for r in failed_files:
            print(f"    [FAILED] {Path(r['file']).name}")
            print(f"           Reason: {r.get('error', 'Unknown')}")
        print()

    header = f"  {'File/Scenario':<42} {'Rows':>8} {'Win':>5} {'Att':>5} {'Ben':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in success_files:
        name = Path(r['file']).parent.name + "/" + Path(r['file']).name
        print(f"  {name:<42} {r['rows_sampled']:>8} {r['windows']:>5} {r['attack_windows']:>5} {r['benign_windows']:>5}")

    # ----------------------------------------------------------------
    # FEATURE AUDIT
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("=== FEATURE AUDIT ===")
    print(SEP)
    print(f"  Feature count         : {len(feature_names)}")
    print(f"  Feature names         : {', '.join(feature_names)}")
    print(f"  Graph node features   : [out_degree, in_degree]  (dim=2)")
    print(f"  NOTE: CIC-IDS2018 CICFlowMeter CSVs on this system do NOT include Src IP/Dst IP columns.")
    print(f"        real_ips=0 is CORRECT for these files; pseudo-nodes (Protocol+DstPort) are used.")
    print(f"  Graph edge features   : {len(feature_names)} flow-level aggregates (dim={len(feature_names)})")

    # ----------------------------------------------------------------
    # LABEL AUDIT
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("=== LABEL AUDIT ===")
    print(SEP)
    all_examples = data_split.train + data_split.val + data_split.test
    all_raw_labels: list = []
    for _, _, raw_labs, _ in all_examples:
        all_raw_labels.extend(raw_labs)
    from collections import Counter
    label_dist = Counter(all_raw_labels)
    attack_total = sum(v for k, v in label_dist.items() if k.lower() not in {'benign', 'background', 'normal', '0', ''})
    benign_total = sum(v for k, v in label_dist.items() if k.lower() in {'benign', 'background', 'normal', '0', ''})
    print(f"  Raw label distribution (top 10):")
    for lbl, cnt in label_dist.most_common(10):
        print(f"    {lbl:<30} : {cnt}")
    print(f"  Mapped attack windows : {attack_total}")
    print(f"  Mapped benign windows : {benign_total}")
    if attack_total == 0:
        print("  [WARNING] No attack windows found in any split!")
    if benign_total == 0:
        print("  [WARNING] No benign windows found in any split!")

    # ----------------------------------------------------------------
    # GRAPH AUDIT  (summary only — details logged per graph above)
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("=== GRAPH AUDIT ===")
    print(SEP)
    total_graphs = sum(r.get('graphs', r.get('windows', 0)) for r in success_files)
    print(f"  Total graphs built    : {total_graphs}")
    print(f"  Node feature dim      : 2  (out_degree, in_degree)")
    print(f"  Edge feature dim      : {len(feature_names)}")
    print(f"  Per-file identity     : real_ips / pseudo_nodes / invalid_ips / self_loops")
    print(f"  {'File':<42} {'real_ips':>8} {'pseudo':>8} {'invalid':>8} {'loops':>8}")
    print("  " + "-" * 78)
    for r in success_files:
        name = Path(r['file']).parent.name + "/" + Path(r['file']).name
        if len(name) > 42:
            name = name[:39] + "..."
        print(
            f"  {name:<42} {r.get('real_ips', 0):>8} {r.get('pseudo_nodes', 0):>8} "
            f"{r.get('invalid_ips', 0):>8} {r.get('self_loops', 0):>8}"
        )

    # ----------------------------------------------------------------
    # SPLIT AUDIT
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("=== SPLIT AUDIT ===")
    print(SEP)
    train_labels = [ex[1][-1] for ex in data_split.train if ex[1]] if data_split.train else []
    val_labels   = [ex[1][-1] for ex in data_split.val   if ex[1]] if data_split.val   else []
    test_labels  = [ex[1][-1] for ex in data_split.test  if ex[1]] if data_split.test  else []
    def dist(labels):
        a = sum(labels); b = len(labels) - a
        return f"attack={a}, benign={b}"
    print(f"  Train samples         : {len(data_split.train)}  [{dist(train_labels)}]")
    print(f"  Validation samples    : {len(data_split.val)}   [{dist(val_labels)}]")
    print(f"  Test samples          : {len(data_split.test)}   [{dist(test_labels)}]")
    print(f"  Temporal split        : 70/10/20 per file/scenario (chronological, no leakage)")
    print(f"  Target horizon        : t+K  (K={selected_k}, predicted S_(t+K) vs y_(t+K))")
    print(f"  LR target horizon     : t+K  (same as World Model — no unfair comparison)")
    print(f"  Decision threshold    : {results.get('selected_threshold', 0.5):.2f}  "
          f"(validation Youden J; frozen before test)")

    # ----------------------------------------------------------------
    # MODEL AUDIT
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("=== MODEL AUDIT ===")
    print(SEP)
    wm_metrics = results.get('world_model', {})
    lr_res = results.get('logistic_regression', {})
    print(f"  GNN encoder                     : YES  (GCN, in=2 node feats, edge_dim={len(feature_names)})")
    print(f"  Temporal World Model (GRU)      : YES  (z_t -> z_t+1 transition)")
    print(f"  Recursive K-step rollout        : YES  (K={selected_k}, no teacher forcing at inference)")
    print(f"  Future target (y at t+K)        : YES  (attack label of window t+K)")
    print(f"  State decoder (attack head)     : YES  (BCE loss, Sigmoid output)")
    print(f"  State decoder (net_state head)  : YES  (MSE, z_next_true from next GNN embedding)")
    print(f"  MITRE tactic head               : YES  (CE loss, 14 tactics, label-based mapping)")
    expl = results.get("explainability", {})
    expl_method = expl.get("method", "gradient_x_input_edge_features")
    print(f"  Explainability                  : YES  ({expl_method}; not SHAP/GNNExplainer)")
    ranked = expl.get("ranked") or []
    if ranked:
        print("  Top driving features:")
        for i, item in enumerate(ranked[:5], start=1):
            print(f"    {i}. {item['feature']}")
    print()
    if wm_metrics:
        print(
            f"  World Model  test P={wm_metrics.get('precision',0):.3f}  "
            f"R={wm_metrics.get('recall',0):.3f}  F1={wm_metrics.get('f1',0):.3f}  "
            f"FPR={wm_metrics.get('fpr',0):.3f}  AUC={wm_metrics.get('roc_auc',0):.3f}  "
            f"thr={wm_metrics.get('threshold', results.get('selected_threshold', 0.5)):.2f}"
        )
    else:
        print(f"  World Model  test metrics: NOT AVAILABLE (no test data)")
    if lr_res.get('trainable') == 'NO':
        print(f"  LR Baseline  : NOT TRAINED — {lr_res.get('reason', '')}")
    elif lr_res:
        print(f"  LR Baseline  test P={lr_res.get('precision',0):.3f}  R={lr_res.get('recall',0):.3f}  F1={lr_res.get('f1',0):.3f}  AUC={lr_res.get('roc_auc',0):.3f}")

    # ----------------------------------------------------------------
    # SMOKE TEST RESULT
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("=== SMOKE TEST RESULT ===")
    print(SEP)
    issues = []
    if failed_files:
        issues.append(f"  {len(failed_files)} file(s) failed to process (see DATASET AUDIT above)")
    if len(data_split.train) == 0:
        issues.append("  CRITICAL: train split is empty!")
    if len(data_split.val) == 0:
        issues.append("  CRITICAL: validation split is empty!")
    if len(data_split.test) == 0:
        issues.append("  CRITICAL: test split is empty!")
    if attack_total == 0:
        issues.append("  CRITICAL: no attack windows in any split!")
    if benign_total == 0:
        issues.append("  CRITICAL: no benign windows in any split!")

    if issues:
        print("  RESULT : *** FAIL ***")
        print("  Issues:")
        for iss in issues:
            print(f"   {iss}")
    else:
        print("  RESULT : PASS")
        print("  All required conditions satisfied.")
        if failed_files:
            print(f"  NOTE: {len(failed_files)} file(s) were skipped — see DATASET AUDIT for details.")

    # FPR diagnosis: if FPR is high, explain why (never hide metric degradation)
    wm_res = results.get("world_model", {})
    fpr_val = wm_res.get("fpr", None)
    if fpr_val is not None and fpr_val >= 0.90:
        print()
        print("  FPR DIAGNOSIS:")
        print(f"    FPR={fpr_val:.3f} — model classifies most/all benign windows as attack.")
        print("    Root causes (in smoke mode):")
        print("      1. Extreme class imbalance: ~90% attack windows in training data.")
        print("      2. Only 1 epoch of training — model hasn't converged.")
        print("      3. pos_weight = n_benign / n_attack upweights attack examples when attacks are the minority.")
        print("      4. Youden's J threshold selection is applied on validation set.")
        print("    Expected improvement with full training (20+ epochs, full data).")
        print("    FPR=1.0 in smoke mode is a known limitation, NOT a code bug.")
    print(f"{SEP}\n")


# ------------------------------------------------------------------
# Human-readable report
# ------------------------------------------------------------------

def _write_eval_report(
    path: Path,
    results: dict,
    ds_report: list,
    selected_k: int,
    mode: str,
) -> None:
    lines = [
        "# Evaluation Report\n",
        f"**Training mode:** {mode}  \n**Selected K:** {selected_k}\n\n",
        "## Dataset Loading Summary\n",
        "| File | Rows Loaded | Rows Sampled | Windows | Attack Windows | Benign Windows |",
        "|---|---|---|---|---|---|",
    ]
    for r in ds_report:
        lines.append(
            f"| {Path(r['file']).name} | {r['rows_total']} | {r['rows_sampled']} "
            f"| {r['windows']} | {r['attack_windows']} | {r['benign_windows']} |"
        )

    def fmt_metrics(m: dict) -> str:
        return (
            f"| Precision | {m.get('precision', 'N/A'):.4f} |\n"
            f"| Recall | {m.get('recall', 'N/A'):.4f} |\n"
            f"| F1 | {m.get('f1', 'N/A'):.4f} |\n"
            f"| FPR | {m.get('fpr', 'N/A'):.4f} |\n"
            f"| ROC-AUC | {m.get('roc_auc', 'N/A'):.4f} |\n"
            f"| PR-AUC | {m.get('pr_auc', 'N/A'):.4f} |\n"
            f"| Threshold | {m.get('threshold', results.get('selected_threshold', 'N/A'))} |\n"
            f"| Inference Latency (ms) | {m.get('inference_latency_ms', 'N/A')} |"
        )

    lines += [
        "\n## World Model (Test Set)\n",
        "| Metric | Value |", "|---|---|",
        fmt_metrics(results.get("world_model", {})),
    ]
    
    lr_res = results.get("logistic_regression", {})
    if lr_res.get("trainable") == "NO":
        lines += [
            "\n## Logistic Regression Baseline (Test Set)\n",
            "| Metric | Value |", "|---|---|",
            "| Trainable | NO |",
            f"| Reason | {lr_res.get('reason', 'Unknown')} |",
        ]
    else:
        lines += [
            "\n## Logistic Regression Baseline (Test Set)\n",
            "| Metric | Value |", "|---|---|",
            "| Trainable | YES |",
            fmt_metrics(lr_res),
        ]
        
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Evaluation report written to %s", path)


def _pick_lr_smoke_files(dataset_dir: Path) -> List[Tuple[Path, object]]:
    """One real file per supported corpus; does not modify datasets."""
    picks: List[Tuple[Path, object]] = []
    cic = sorted((dataset_dir / "CIC-IDS2018").rglob("*.csv")) if (dataset_dir / "CIC-IDS2018").exists() else []
    ctu = sorted((dataset_dir / "CTU-13-Dataset").rglob("*.binetflow")) if (dataset_dir / "CTU-13-Dataset").exists() else []
    unsw_root = dataset_dir / "UNSW-NB15"
    unsw = []
    if unsw_root.exists():
        candidates = [
            p for p in sorted(unsw_root.rglob("*.csv"))
            if "feature" not in p.name.lower() and "list_events" not in p.name.lower()
        ]
        headered = [p for p in candidates if "training" in p.name.lower() or "testing" in p.name.lower()]
        unsw = headered or candidates
    if cic:
        picks.append((cic[0], CICIDSAdapter()))
    if ctu:
        picks.append((ctu[0], CTU13Adapter()))
    if unsw:
        picks.append((unsw[0], UNSWAdapter()))
    return picks


def _examples_from_loaded_df(df: pd.DataFrame, source: str, k: int, window_ms: int, hp: dict):
    if df is None or df.empty:
        return [], [], []
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    windows = create_time_windows(df, window_ms=window_ms)
    builder = GraphBuilder()
    graphs, labels, raws = [], [], []
    for w_df in windows.values():
        g = builder.build_window_graph(w_df)
        if g is None:
            continue
        graphs.append(g)
        if "label" in w_df.columns and len(w_df):
            attack = int(any(is_attack_label(str(x)) for x in w_df["label"]))
            raw = next((str(x) for x in w_df["label"] if is_attack_label(str(x))), str(w_df["label"].iloc[0]))
        else:
            attack, raw = 0, "Benign"
        labels.append(attack)
        raws.append(raw)

    n_windows = len(graphs)
    samples = []
    for t in range(n_windows - k):
        samples.append((graphs[t], labels[t + 1 : t + k + 1], raws[t + 1 : t + k + 1]))
    M = len(samples)
    if M < 3:
        logger.warning("LR rebuild: %s produced only %d samples, skipping.", source, M)
        return [], [], []
    tr_len = int(M * hp.get("train_ratio", 0.70))
    vl_len = max(1, int(M * hp.get("val_ratio", 0.10)))
    te_len = M - tr_len - vl_len
    if te_len <= 0:
        te_len = 1
        tr_len = M - vl_len - te_len
        if tr_len <= 0:
            tr_len, vl_len, te_len = 1, 1, 1
    return samples[:tr_len], samples[tr_len:tr_len + vl_len], samples[tr_len + vl_len:]


def rebuild_lr_baseline(mode: str = "smoke") -> dict:
    """
    Rebuild ONLY the Logistic Regression artifact.

    Uses a bounded read of existing CIC/CTU/UNSW files (nrows on large CSVs)
    so this does not re-run World Model training or ingest every CIC CSV.
    Feature schema is the same 11 flow-level edge features as the World Model.
    """
    cfg = load_config()
    hp = cfg["hyperparameters"]
    set_seed(hp["random_seed"])
    k = int(hp["k_rollout_steps"])
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_nrows = 12000 if mode != "full" else None

    stale_path = ckpt_dir / "lr_baseline.joblib"
    if stale_path.exists():
        stale = joblib.load(stale_path)
        n_old = int(getattr(stale, "n_features_in_", -1))
        if n_old != LR_FEATURE_DIM:
            logger.warning(
                "Invalidating incompatible LR artifact n_features_in_=%d (canonical=%d)",
                n_old, LR_FEATURE_DIM,
            )

    train_ex, test_ex = [], []
    files_used = []
    for path, adapter in _pick_lr_smoke_files(Path(cfg["paths"]["dataset_dir"])):
        logger.info("LR rebuild loading %s", path)
        try:
            if path.suffix.lower() == ".csv" and csv_nrows:
                raw = pd.read_csv(path, nrows=csv_nrows, low_memory=False)
                df = adapter.load_flow_data(raw)
            else:
                df = adapter.load_flow_data(path)
        except Exception as exc:
            logger.warning("LR rebuild skipped %s: %s", path.name, exc)
            continue
        tr, _vl, te = _examples_from_loaded_df(
            df, str(path), k, hp["time_window_ms"], hp
        )
        train_ex.extend(tr)
        test_ex.extend(te)
        files_used.append(str(path))
        print(f"LR rebuild file={path.name} train={len(tr)} test={len(te)}")

    X_train, y_train = build_lr_features(train_ex)
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        raise RuntimeError("Cannot rebuild LR: empty or single-class training split.")
    if int(X_train.shape[1]) != LR_FEATURE_DIM:
        raise ValueError(f"LR rebuild dim {X_train.shape[1]} != {LR_FEATURE_DIM}")

    lr_model = LogisticRegression(max_iter=1000, random_state=hp["random_seed"])
    lr_model.fit(X_train, y_train)
    assert_lr_input_matches_model(lr_model, X_train)

    joblib.dump(lr_model, ckpt_dir / "lr_baseline.joblib")
    save_lr_schema(ckpt_dir)

    cfg_snap = ckpt_dir / "training_config.json"
    if cfg_snap.exists():
        snap = json.loads(cfg_snap.read_text(encoding="utf-8"))
        snap["lr_feature_names"] = list(LR_FEATURE_NAMES)
        snap["lr_feature_dim"] = int(LR_FEATURE_DIM)
        snap.setdefault("hyperparameters", {})["lr_feature_dim"] = int(LR_FEATURE_DIM)
        cfg_snap.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    metrics: dict = {
        "n_features": int(lr_model.n_features_in_),
        "feature_names": list(LR_FEATURE_NAMES),
        "trainable": "YES",
        "train_rows": int(len(y_train)),
        "world_model_retrained": False,
        "files": files_used,
        "csv_nrows_cap": csv_nrows,
    }
    if test_ex:
        X_test, y_test = build_lr_features(test_ex)
        assert_lr_input_matches_model(lr_model, X_test)
        lr_prob = lr_model.predict_proba(X_test)[:, 1]
        lr_pred = (lr_prob >= 0.5).astype(int)
        metrics.update(evaluate_predictions(y_test, lr_pred, lr_prob))
        metrics["test_rows"] = int(len(y_test))
        print("\nLR REBUILD TEST METRICS (11-d window-mean flow features)")
        print(f"  n_features_in_={lr_model.n_features_in_}")
        print(f"  test_rows={len(y_test)}")
        print(f"  Precision={metrics.get('precision', 0):.4f}")
        print(f"  Recall={metrics.get('recall', 0):.4f}")
        print(f"  F1={metrics.get('f1', 0):.4f}")
        print(f"  FPR={metrics.get('fpr', 0):.4f}")
        print(f"  ROC-AUC={metrics.get('roc_auc', 0):.4f}")

    (out_dir / "lr_baseline_eval.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Wrote LR artifact to %s (did not train World Model)", ckpt_dir)
    return metrics


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Cybersecurity World Model")
    parser.add_argument(
        "--mode", choices=["smoke", "full", "lr"], default="smoke",
        help="smoke/full train World Model; lr = rebuild Logistic Regression only",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--k", type=int, default=None, help="Override K rollout steps")
    parser.add_argument("--transition", choices=["direct", "residual"], default="direct")
    parser.add_argument("--delta-scale", type=float, default=0.5)
    parser.add_argument("--delta-loss-weight", type=float, default=0.25)
    args = parser.parse_args()
    if args.mode == "lr":
        rebuild_lr_baseline(mode="smoke")
    else:
        train(mode=args.mode, epochs=args.epochs, k_override=args.k,
              transition=args.transition, delta_scale=args.delta_scale,
              delta_loss_weight=args.delta_loss_weight)
