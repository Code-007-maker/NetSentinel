"""
Logistic Regression baseline on the same 11 flow-level edge features
used by the World Model (window mean of EDGE_FEATURE_NAMES).

This is a fair tabular benchmark: no GNN spatial encoding and no GRU
rollout.  Extra graph-structure scalars (degrees, node/edge counts) are
intentionally excluded so LR is not given a different feature family
than the World Model's telemetry.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data

from src.feature_schema import EDGE_FEATURE_DIM, EDGE_FEATURE_NAMES

logger = logging.getLogger(__name__)

LR_FEATURE_NAMES: list[str] = list(EDGE_FEATURE_NAMES)
LR_FEATURE_DIM: int = EDGE_FEATURE_DIM  # 11
LR_SCHEMA_FILENAME = "lr_schema.json"
LR_MODEL_FILENAME = "lr_baseline.joblib"

# Older smoke checkpoints concatenated 2 degree means + 2 counts + 11 edges.
LEGACY_INCOMPATIBLE_DIMS = {15, 5}


def lr_vector_from_edge_matrix(edge_feats: np.ndarray) -> np.ndarray:
    """Window-mean of per-flow 11-d edge features → shape (1, 11)."""
    if edge_feats is None or np.asarray(edge_feats).size == 0:
        return np.zeros((1, LR_FEATURE_DIM), dtype=np.float32)
    x = np.asarray(edge_feats, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != LR_FEATURE_DIM:
        raise ValueError(
            f"Edge feature matrix has dim {x.shape[1]}, expected {LR_FEATURE_DIM} "
            f"({LR_FEATURE_NAMES}). Refusing to pad/truncate."
        )
    return x.mean(axis=0, keepdims=True).astype(np.float32)


def lr_vector_from_graph(graph: Data) -> np.ndarray:
    if graph is None or graph.edge_attr is None or graph.edge_attr.numel() == 0:
        return np.zeros((1, LR_FEATURE_DIM), dtype=np.float32)
    return lr_vector_from_edge_matrix(graph.edge_attr.detach().cpu().numpy())


def lr_vector_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    if df is None or df.empty:
        return np.zeros((1, LR_FEATURE_DIM), dtype=np.float32)
    missing = [c for c in LR_FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing LR/World-Model flow columns: {missing}"
        )
    x = df[LR_FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return lr_vector_from_edge_matrix(x.to_numpy(dtype=np.float32))


def build_lr_features(examples: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-graph 11-d window means. Target = future_attacks[-1] (t+K), same
    horizon as the World Model rollout.
    """
    X_rows, y = [], []
    for g_t, future_attacks, *_ in examples:
        if g_t is None or not future_attacks:
            continue
        X_rows.append(lr_vector_from_graph(g_t)[0])
        y.append(future_attacks[-1])
    if not X_rows:
        return np.zeros((0, LR_FEATURE_DIM), dtype=np.float32), np.array([], dtype=int)
    return np.vstack(X_rows).astype(np.float32), np.array(y, dtype=int)


def assert_lr_input_matches_model(model: LogisticRegression, X: np.ndarray) -> None:
    """Fail loudly on dimension mismatch — never pad/truncate."""
    if X.ndim != 2:
        raise ValueError(f"LR input must be 2-d, got shape {X.shape}")
    actual = int(X.shape[1])
    expected_schema = LR_FEATURE_DIM
    expected_model = int(getattr(model, "n_features_in_", expected_schema))
    if actual != expected_schema:
        raise ValueError(
            f"LR input dim {actual} != canonical schema dim {expected_schema}"
        )
    if actual != expected_model:
        raise ValueError(
            f"LR input dim {actual} != model n_features_in_={expected_model}. "
            "Artifact is incompatible; rebuild the LR baseline."
        )


def lr_schema_dict() -> dict:
    return {
        "feature_names": list(LR_FEATURE_NAMES),
        "n_features": int(LR_FEATURE_DIM),
        "aggregation": "window_mean",
        "scaled": False,
        "representation": "same_11_flow_edge_features_as_world_model",
    }


def save_lr_schema(directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LR_SCHEMA_FILENAME
    path.write_text(json.dumps(lr_schema_dict(), indent=2), encoding="utf-8")
    return path


def load_lr_schema(directory: str | Path) -> Optional[dict]:
    path = Path(directory) / LR_SCHEMA_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_compatible(model: object, schema: Optional[dict]) -> Tuple[bool, str]:
    n_in = int(getattr(model, "n_features_in_", -1))
    if n_in in LEGACY_INCOMPATIBLE_DIMS:
        return False, (
            f"Incompatible LR artifact n_features_in_={n_in} "
            f"(canonical is {LR_FEATURE_DIM}). Rebuild required."
        )
    if n_in != LR_FEATURE_DIM:
        return False, (
            f"LR n_features_in_={n_in} != canonical {LR_FEATURE_DIM}. Rebuild required."
        )
    if schema is None:
        return False, (
            f"Missing {LR_SCHEMA_FILENAME}; refusing to assume feature order. Rebuild required."
        )
    names = schema.get("feature_names") or []
    n_schema = int(schema.get("n_features", -1))
    if n_schema != LR_FEATURE_DIM or list(names) != list(LR_FEATURE_NAMES):
        return False, (
            f"lr_schema.json does not match canonical {LR_FEATURE_NAMES}. Rebuild required."
        )
    return True, "ok"


def load_lr_bundle(directory: str | Path) -> Tuple[Optional[LogisticRegression], Optional[dict], str]:
    """
    Load LR model + schema. Returns (model, schema, status).
    model is None when missing or incompatible (never silently used).
    """
    directory = Path(directory)
    model_path = directory / LR_MODEL_FILENAME
    if not model_path.exists():
        return None, None, f"LR baseline not found: {model_path}"
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        return None, None, f"Failed to load LR baseline: {exc}"
    schema = load_lr_schema(directory)
    ok, reason = artifact_compatible(model, schema)
    if not ok:
        logger.warning("Invalidating LR artifact: %s", reason)
        return None, schema, reason
    return model, schema, "ok"


class LRBaseline:
    """Thin wrapper kept for callers that prefer an object API."""

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.feature_names = list(LR_FEATURE_NAMES)

    def extract_features(self, df_window: pd.DataFrame) -> np.ndarray:
        return lr_vector_from_dataframe(df_window)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if X_train.ndim != 2 or int(X_train.shape[1]) != LR_FEATURE_DIM:
            raise ValueError(
                f"LR train dim {getattr(X_train, 'shape', None)} != {(None, LR_FEATURE_DIM)}"
            )
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert_lr_input_matches_model(self.model, X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert_lr_input_matches_model(self.model, X)
        return self.model.predict_proba(X)[:, 1]
