"""
Feature preprocessing: scaler fitting, saving, and loading.

Design decisions:
- Scalers are fit ONLY on training data.
- The fitted scaler is saved alongside the model checkpoint so inference
  uses exactly the same normalisation.
- The feature schema is also saved so we can catch mismatches at inference time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

from src.feature_schema import FLOW_FEATURE_COLS  # canonical 11-feature list



class FlowFeatureScaler:
    """Wraps StandardScaler and preserves the feature schema."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False
        self.feature_cols: List[str] = FLOW_FEATURE_COLS

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Fit on training data only."""
        X = self._extract(df)
        self._scaler.fit(X)
        self._fitted = True
        logger.info(
            "Scaler fitted on %d rows, features: %s", len(df), self.feature_cols
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted scaler; raises if not yet fitted."""
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        X = self._extract(df)
        return self._scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save scaler and feature schema to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, directory / "scaler.joblib")
        schema = {"feature_cols": self.feature_cols}
        (directory / "feature_schema.json").write_text(
            json.dumps(schema, indent=2), encoding="utf-8"
        )
        logger.info("Scaler saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> "FlowFeatureScaler":
        """Load a previously saved scaler from *directory*."""
        directory = Path(directory)
        scaler_path = directory / "scaler.joblib"
        schema_path = directory / "feature_schema.json"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at {schema_path}")

        obj = cls()
        obj._scaler = joblib.load(scaler_path)
        obj._fitted = True
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        obj.feature_cols = schema["feature_cols"]
        logger.info("Scaler loaded from %s", directory)
        return obj

    def transform_edge_attr(self, edge_attr) -> "object":
        """Scale a graph edge_attr tensor with the train-fitted scaler."""
        import torch
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        if edge_attr is None or getattr(edge_attr, "numel", lambda: 0)() == 0:
            return edge_attr
        x = edge_attr.detach().cpu().numpy()
        df = pd.DataFrame(x, columns=self.feature_cols)
        out = self.transform(df)
        return torch.as_tensor(out, dtype=torch.float32, device=edge_attr.device)

    def scale_graph(self, graph):
        """In-place scale of graph.edge_attr. Returns the same graph object."""
        if graph is None:
            return graph
        if getattr(graph, "edge_attr", None) is None:
            return graph
        graph.edge_attr = self.transform_edge_attr(graph.edge_attr)
        return graph

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and coerce feature columns from dataframe."""
        cols = []
        for col in self.feature_cols:
            if col in df.columns:
                cols.append(pd.to_numeric(df[col], errors="coerce").fillna(0.0))
            else:
                cols.append(pd.Series(np.zeros(len(df)), name=col))
        return np.column_stack(cols).astype(np.float32)
