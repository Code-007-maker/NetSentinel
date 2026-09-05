"""
DatasetManager: multi-dataset loading with per-dataset temporal splits.

Key design decisions:
- Each dataset/file is treated as an independent timeline.
- Temporal windows are created within each timeline.
- Train/val/test split is chronological: earliest 70 % → train,
  next 10 % → val, last 20 % → test.
- Sequences from different datasets are NEVER connected.
  Only samples (individual (G_t, future_labels) pairs) are combined
  for batching / optimisation.
- Sampling is deterministic: fixed random_seed, head-of-file ordering
  for temporal integrity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.dataset_adapters import CTU13Adapter, CICIDSAdapter, UNSWAdapter
from src.temporal_windowing import create_time_windows
from src.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Label helpers
# ------------------------------------------------------------------

BENIGN_LABELS = {"benign", "background", "normal", "0", ""}


def is_attack_label(label: str) -> int:
    """Return 1 if the label represents an attack, 0 otherwise."""
    return 0 if str(label).strip().lower() in BENIGN_LABELS else 1


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

@dataclass
class GraphSequence:
    """
    A chronological sequence of PyG graphs from one dataset file.

    graphs[i]       → G(t)
    attack_labels[i] → binary label for window i
    raw_label[i]    → original string label (for MITRE mapping)
    source          → dataset/file identifier
    """
    graphs: List[Data]
    attack_labels: List[int]
    raw_labels: List[str]
    source: str = ""


@dataclass
class DataSplit:
    """
    Complete train / val / test sets across all datasets.
    Each split is a list of (graph_at_t, future_attacks_K, future_raw_labels_K,
    future_graphs_K) tuples.
    """
    train: List[Tuple[Data, List[int], List[str], List[Data]]] = field(default_factory=list)
    val:   List[Tuple[Data, List[int], List[str], List[Data]]] = field(default_factory=list)
    test:  List[Tuple[Data, List[int], List[str], List[Data]]] = field(default_factory=list)


# ------------------------------------------------------------------
# Main manager class
# ------------------------------------------------------------------

class DatasetManager:
    """
    Loads all discovered datasets, applies deterministic sampling,
    builds per-dataset temporal splits, and returns (G_t, future_labels)
    training examples.
    """

    def __init__(
        self,
        dataset_dir: str,
        config: dict,
        smoke_test: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.cfg = config
        self.smoke_test = smoke_test
        self.hp = config["hyperparameters"]
        self.rng = np.random.default_rng(self.hp["random_seed"])

        # Report lists
        self.report: List[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_splits(self, k: int) -> DataSplit:
        """
        Discover datasets, load, window, and create (G_t, y_future) examples.
        Splits SAMPLES chronologically (70/10/20) per sequence.
        """
        sequences = self._load_all_sequences()
        split = DataSplit()
        
        skipped_short = 0
        
        for seq in sequences:
            # 1. Generate all valid (G_t, y_future) samples for this sequence
            n_windows = len(seq.graphs)
            samples = []
            for t in range(n_windows - k):
                # Input = S_t; targets = y_{t+1} .. y_{t+K} from THIS sequence only.
                # t + k < n_windows is guaranteed by the range, so t+K never
                # crosses a file/scenario boundary.
                assert t + k < n_windows, (
                    f"t+K crosses sequence boundary: t={t} k={k} n={n_windows} src={seq.source}"
                )
                g_t = seq.graphs[t]
                future_attacks = seq.attack_labels[t + 1 : t + k + 1]
                future_raws = seq.raw_labels[t + 1 : t + k + 1]
                assert len(future_attacks) == k, (
                    f"future target length {len(future_attacks)} != K={k} for {seq.source}"
                )
                samples.append(
                    (g_t, future_attacks, future_raws, seq.graphs[t + 1 : t + k + 1])
                )
                
            M = len(samples)
            if M < 3:
                skipped_short += 1
                logger.warning("Sequence %s has only %d valid samples (needs >= 3). Skipping.", seq.source, M)
                continue
                
            # 2. Split samples chronologically
            tr_len = int(M * self.hp.get("train_ratio", 0.70))
            vl_len = max(1, int(M * self.hp.get("val_ratio", 0.10)))
            te_len = M - tr_len - vl_len

            # Guarantee at least 1 sample per split.
            # When te_len <= 0 we shrink tr_len (not expand vl/te)
            # so the total stays exactly M.
            if te_len <= 0:
                te_len = 1
                tr_len = M - vl_len - te_len   # may be 0 for very small M
                if tr_len <= 0:
                    tr_len, vl_len, te_len = 1, 1, 1  # absolute minimum: 1/1/1
                    
            tr_end = tr_len
            vl_end = tr_end + vl_len
            
            tr_samples = samples[:tr_end]
            vl_samples = samples[tr_end:vl_end]
            te_samples = samples[vl_end:]
            
            split.train.extend(tr_samples)
            split.val.extend(vl_samples)
            split.test.extend(te_samples)
            
            # Print detailed split report for this sequence
            print(f"Dataset/File: {Path(seq.source).name}")
            print(f"    total_windows: {n_windows}")
            print(f"    usable_future_windows: {M}")
            print(f"    train: {len(tr_samples)}")
            print(f"    validation: {len(vl_samples)}")
            print(f"    test: {len(te_samples)}")

        print(f"\nTOTAL SPLIT")
        print(f"    train = {len(split.train)}")
        print(f"    val   = {len(split.val)}")
        print(f"    test  = {len(split.test)}")
        print(f"    skipped_short_sequences = {skipped_short}\n")
        
        def dist(samps):
            if not samps:
                return "attack=0 benign=0", 0, 0
            attacks = sum(1 for ex in samps if any(l == 1 for l in ex[1]))
            benigns = len(samps) - attacks
            return f"attack={attacks} benign={benigns}", attacks, benigns
            
        def dist_horizon(samps, k_h: int):
            ys = [int(ex[1][k_h - 1]) for ex in samps if ex[1] and len(ex[1]) >= k_h]
            if not ys:
                return "attack=0 benign=0", 0, 0
            attacks = int(sum(ys))
            benigns = int(len(ys) - attacks)
            return f"attack={attacks} benign={benigns}", attacks, benigns

        print("CLASS DISTRIBUTION (any future step in 1..K is attack)")
        t_dist, t_a, t_b = dist(split.train)
        v_dist, v_a, v_b = dist(split.val)
        te_dist, te_a, te_b = dist(split.test)
        print(f"    train: {t_dist}")
        print(f"    val:   {v_dist}")
        print(f"    test:  {te_dist}")
        k_h = k
        print(f"CLASS DISTRIBUTION (t+K={k_h} target only)")
        print(f"    train: {dist_horizon(split.train, k_h)[0]}")
        print(f"    val:   {dist_horizon(split.val, k_h)[0]}")
        print(f"    test:  {dist_horizon(split.test, k_h)[0]}")
        print()

        _, v_a_k, v_b_k = dist_horizon(split.val, k_h)
        if (v_a_k > 0 and v_b_k == 0) or (v_b_k > 0 and v_a_k == 0):
            logger.warning("VALIDATION_SINGLE_CLASS_WARNING: t+K validation labels contain only one class.")
            
        return split

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_sequences(self) -> List[GraphSequence]:
        sequences: List[GraphSequence] = []
        use_datasets = [d.lower() for d in self.cfg["training"]["use_datasets"]]

        # Walk dataset_dir, pick directories matching use_datasets list
        for ds_dir in sorted(self.dataset_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            if not any(u in ds_dir.name.lower() for u in use_datasets):
                continue

            logger.info("Scanning dataset directory: %s", ds_dir)
            files = (
                list(ds_dir.rglob("*.csv"))
                + list(ds_dir.rglob("*.binetflow"))
            )
            for f in sorted(files):
                seq = self._process_file(f)
                if seq is not None:
                    sequences.append(seq)

        logger.info("Total sequences collected: %d", len(sequences))
        return sequences

    def _process_file(self, file_path: Path) -> Optional[GraphSequence]:
        """Load one file → unified schema → sample → window → build graphs."""
        path_str = str(file_path).lower()

        try:
            logger.info("Loading %s", file_path.name)

            # Select adapter
            if "ctu" in path_str:
                adapter = CTU13Adapter()
            elif "unsw" in path_str:
                adapter = UNSWAdapter()
            else:
                adapter = CICIDSAdapter()

            df_full = adapter.load_flow_data(file_path)
            rows_total = len(df_full)

            if df_full.empty or rows_total == 0:
                logger.warning("Empty dataframe from %s, skipping.", file_path.name)
                return None

            # Convert timestamp to datetime if needed for time slicing
            if not pd.api.types.is_datetime64_any_dtype(df_full['timestamp']):
                df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], errors='coerce')
            df_full = df_full.dropna(subset=['timestamp']).sort_values("timestamp")

            if self.smoke_test:
                # Time-based slicing to guarantee minimum windows and class balance
                min_windows = self.hp.get("smoke_test_min_windows", 50)
                window_ms = self.hp["time_window_ms"]
                duration_needed = pd.Timedelta(milliseconds=min_windows * window_ms)
                
                # Find first class transition (benign -> attack or attack -> benign)
                attack_mask = ~df_full["label"].astype(str).str.lower().isin(BENIGN_LABELS)
                transitions = attack_mask.diff().fillna(False)
                transition_rows = df_full[transitions]
                
                if not transition_rows.empty:
                    # Center the slice around the first transition
                    first_transition_time = transition_rows.iloc[0]["timestamp"]
                    start_time = first_transition_time - (duration_needed / 2)
                    
                    # Ensure start_time doesn't go before dataset start
                    if start_time < df_full["timestamp"].iloc[0]:
                        start_time = df_full["timestamp"].iloc[0]
                        
                    end_time = start_time + duration_needed
                    
                    # Ensure end_time doesn't go beyond dataset end
                    if end_time > df_full["timestamp"].iloc[-1]:
                        end_time = df_full["timestamp"].iloc[-1]
                        start_time = max(df_full["timestamp"].iloc[0], end_time - duration_needed)
                else:
                    # No transitions, just take from the beginning
                    logger.warning("%s contains only one class in the entire file. Slice may lack balance.", file_path.name)
                    start_time = df_full["timestamp"].iloc[0]
                    end_time = start_time + duration_needed
                
                # Filter by the computed time slice
                df_full = df_full[(df_full["timestamp"] >= start_time) & (df_full["timestamp"] <= end_time)]
                logger.info("Smoke test: Sliced time from %s to %s", start_time, end_time)
            else:
                # Standard deterministic row limit for full training
                max_rows = self.hp.get("max_rows_per_file", 50000)
                if max_rows > 0 and len(df_full) > max_rows:
                    df_full = df_full.iloc[:max_rows]

            df_full = df_full.reset_index(drop=True)
            rows_sampled = len(df_full)

            # Temporal windowing
            windows = create_time_windows(
                df_full,
                window_ms=self.hp["time_window_ms"]
            )
            n_windows = len(windows)

            if n_windows < self.hp["min_windows_per_sequence"]:
                logger.warning(
                    "%s produced only %d windows (min=%d), skipping.",
                    file_path.name, n_windows, self.hp["min_windows_per_sequence"]
                )
                return None

            # Build graphs per window — fresh builder per file for clean node IDs
            builder = GraphBuilder()
            graphs = []
            attack_labels = []
            raw_labels = []
            skipped_windows = 0

            for w_df in windows.values():
                g = builder.build_window_graph(w_df, smoke_test_log=self.smoke_test)

                if g is None:
                    # Corrupt graph (e.g., all-same-IP due to mapping failure)
                    skipped_windows += 1
                    continue

                graphs.append(g)

                # Window label = attack if ANY flow in window is attack
                window_labels = w_df["label"].astype(str).tolist()
                w_attack = int(any(is_attack_label(l) for l in window_labels))
                # Representative raw label: first non-benign label or Benign
                rep_raw = next(
                    (l for l in window_labels if is_attack_label(l)), "Benign"
                )
                attack_labels.append(w_attack)
                raw_labels.append(rep_raw)

            if skipped_windows > 0:
                logger.warning("%s: skipped %d corrupt windows out of %d total.",
                               file_path.name, skipped_windows, n_windows)

            ident = builder.identity_audit()
            logger.info(
                "Graph identity [%s]: real_ips=%d pseudo_nodes=%d invalid_ips=%d self_loops=%d",
                file_path.name,
                ident["real_ips"], ident["pseudo_nodes"],
                ident["invalid_ips"], ident["self_loops"],
            )

            # Report
            attack_count = sum(attack_labels)
            self.report.append({
                "file": str(file_path),
                "rows_total": rows_total,
                "rows_sampled": rows_sampled,
                "windows": n_windows,
                "graphs": len(graphs),
                "attack_windows": attack_count,
                "benign_windows": n_windows - attack_count,
                "real_ips": ident["real_ips"],
                "pseudo_nodes": ident["pseudo_nodes"],
                "invalid_ips": ident["invalid_ips"],
                "self_loops": ident["self_loops"],
            })

            logger.info(
                "%s | rows: %d→%d | windows: %d | attack: %d | benign: %d",
                file_path.name, rows_total, rows_sampled,
                n_windows, attack_count, n_windows - attack_count,
            )

            return GraphSequence(
                graphs=graphs,
                attack_labels=attack_labels,
                raw_labels=raw_labels,
                source=str(file_path),
            )
        except Exception as e:
            logger.error("FAILED to process %s: %s", file_path.name, str(e))
            self.report.append({
                "file": str(file_path),
                "rows_total": 0,
                "rows_sampled": 0,
                "windows": 0,
                "graphs": 0,
                "attack_windows": 0,
                "benign_windows": 0,
                "error": str(e),
            })
            return None

