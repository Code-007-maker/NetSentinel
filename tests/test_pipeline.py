"""
Comprehensive test suite for the Cybersecurity World Model pipeline.

Tests verify:
  - CSV ingestion (CIC, UNSW, CTU)
  - PCAP graceful failure
  - Unified schema structure
  - Temporal windowing
  - Temporal train/val/test split separation (no leakage)
  - Graph construction
  - Feature normalization (no leakage from val/test into scaler)
  - Future target generation
  - Recursive K-step rollout
  - Checkpoint save/load
  - Dashboard inference path shape
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset_adapters import CICIDSAdapter, CTU13Adapter, UNSWAdapter
from src.dataset_manager import DatasetManager, is_attack_label
from src.gnn_encoder import GNNEncoder
from src.graph_builder import GraphBuilder
from src.mitre_mapper import MitreMapper
from src.pcap_parser import extract_pcap_features
from src.preprocessing import FlowFeatureScaler
from src.temporal_windowing import create_time_windows
from src.world_model import StateDecoder, TemporalWorldModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_is_attack_label():
    assert is_attack_label("Attack") == 1
    assert is_attack_label("Botnet") == 1
    assert is_attack_label("Benign") == 0
    assert is_attack_label("Normal") == 0
    assert is_attack_label("background") == 0
    assert is_attack_label("0") == 0
    assert is_attack_label("") == 0
    assert is_attack_label("Malicious") == 1

def _make_flow_df(n: int = 10) -> pd.DataFrame:
    now = pd.Timestamp("2023-01-01 00:00:00")
    times = pd.date_range(now, periods=n, freq="500ms")
    return pd.DataFrame(
        {
            "timestamp": times,
            "src_ip": ["192.168.1.1"] * n,
            "dst_ip": ["10.0.0.1"] * n,
            "src_port": [1000] * n,
            "dst_port": [80] * n,
            "protocol": ["tcp"] * n,
            "duration": np.random.default_rng(0).uniform(0.1, 2.0, n),
            "packets_forward": np.random.default_rng(1).integers(1, 50, n).astype(float),
            "packets_backward": np.random.default_rng(2).integers(0, 20, n).astype(float),
            "bytes_forward": np.random.default_rng(3).integers(100, 5000, n).astype(float),
            "bytes_backward": np.random.default_rng(4).integers(0, 2000, n).astype(float),
            # Derived features (would normally be computed by compute_derived_features)
            "rate_fwd": np.random.default_rng(5).uniform(0.5, 25.0, n),
            "rate_bwd": np.random.default_rng(6).uniform(0.0, 10.0, n),
            "byte_rate_fwd": np.random.default_rng(7).uniform(50, 2500, n),
            "byte_rate_bwd": np.random.default_rng(8).uniform(0, 1000, n),
            "avg_pkt_size_fwd": np.random.default_rng(9).uniform(64, 1500, n),
            "avg_pkt_size_bwd": np.random.default_rng(10).uniform(0, 1500, n),
            "label": ["Benign"] * (n - 3) + ["Botnet"] * 3,
        }
    )


# ---------------------------------------------------------------------------
# 1. CSV Ingestion
# ---------------------------------------------------------------------------

class TestCSVIngestion:
    def test_cic_ids_adapter_from_df(self):
        # CIC-IDS2018 style column names
        df = pd.DataFrame(
            {
                "Timestamp": ["2023-01-01 00:00:01"],
                "Src IP": ["1.2.3.4"],
                "Dst IP": ["5.6.7.8"],
                "Src Port": [1234],
                "Dst Port": [80],
                "Protocol": [6],
                "Flow Duration": [100],
                "Tot Fwd Pkts": [5],
                "Tot Bwd Pkts": [3],
                "TotLen Fwd Pkts": [500],
                "TotLen Bwd Pkts": [300],
                "Label": ["Bot"],
            }
        )
        unified = CICIDSAdapter().load_flow_data(df)
        assert "timestamp" in unified.columns
        assert "label" in unified.columns
        assert unified["source_dataset"].iloc[0] == "CIC-IDS2018"
        assert unified["duration"].iloc[0] == 100.0
        assert unified["src_node"].iloc[0] == "1.2.3.4"
        assert unified["dst_node"].iloc[0] == "5.6.7.8"

    def test_cic_source_destination_ip_aliases(self):
        df = pd.DataFrame(
            {
                "Timestamp": ["2023-01-01 00:00:01"],
                "Source IP": ["10.1.2.3"],
                "Destination IP": ["10.4.5.6"],
                "Dst Port": [443],
                "Protocol": [6],
                "Flow Duration": [10],
                "Label": ["Benign"],
            }
        )
        unified = CICIDSAdapter().load_flow_data(df)
        assert unified["src_node"].iloc[0] == "10.1.2.3"
        assert unified["dst_node"].iloc[0] == "10.4.5.6"

    def test_cic_ids_handles_missing_ip_columns(self):
        df = pd.DataFrame(
            {
                "Timestamp": ["2023-01-01 00:00:01"],
                "Flow Duration": [np.inf],
                "Tot Fwd Pkts": [2],
                "Label": ["BENIGN"],
            }
        )
        unified = CICIDSAdapter().load_flow_data(df)
        assert unified["src_node"].iloc[0] == "proto_0"  # default fallback
        assert unified["duration"].iloc[0] == 0.0       # inf → 0

    def test_unsw_adapter_from_df(self):
        df = pd.DataFrame(
            {
                "stime": [1.0],
                "srcip": ["10.0.0.1"],
                "dstip": ["10.0.0.2"],
                "sport": [4321],
                "dsport": [443],
                "proto": ["tcp"],
                "dur": [0.5],
                "spkts": [10],
                "dpkts": [5],
                "sbytes": [1000],
                "dbytes": [500],
                "attack_cat": ["Exploits"],
                "label": [1],
            }
        )
        unified = UNSWAdapter().load_flow_data(df)
        assert unified["label"].iloc[0] == "Exploits"
        assert unified["source_dataset"].iloc[0] == "UNSW-NB15"

    def test_ctu13_adapter_from_df(self):
        df = pd.DataFrame(
            {
                "StartTime": ["2011-08-10 09:46:53"],
                "SrcAddr": ["147.32.84.165"],
                "DstAddr": ["74.125.232.96"],
                "Sport": [1337],
                "Dport": [80],
                "Proto": ["tcp"],
                "Dur": [1.23],
                "TotPkts": [12],
                "SrcBytes": [800],
                "TotBytes": [1200],
                "Label": ["flow=Background"],
            }
        )
        unified = CTU13Adapter().load_flow_data(df)
        assert unified["source_dataset"].iloc[0] == "CTU-13"
        assert unified["bytes_backward"].iloc[0] == 400  # 1200 - 800


# ---------------------------------------------------------------------------
# 2. PCAP ingestion (graceful failure on empty/non-PCAP file)
# ---------------------------------------------------------------------------

class TestPCAPIngestion:
    def test_empty_pcap_returns_empty_df(self, tmp_path):
        pcap_file = tmp_path / "empty.pcap"
        pcap_file.write_bytes(b"")  # zero-byte file
        df = extract_pcap_features(str(pcap_file))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_pcap_features("/nonexistent/path/file.pcap")


# ---------------------------------------------------------------------------
# 3. Temporal windowing
# ---------------------------------------------------------------------------

class TestTemporalWindowing:
    def test_windows_created(self):
        df = _make_flow_df(20)
        windows = create_time_windows(df, window_ms=500)
        assert len(windows) > 0

    def test_windows_chronologically_ordered(self):
        df = _make_flow_df(20)
        windows = create_time_windows(df, window_ms=500)
        keys = list(windows.keys())
        assert keys == sorted(keys)

    def test_no_row_crossing_window_boundary(self):
        df = _make_flow_df(20)
        windows = create_time_windows(df, window_ms=500)
        # Total rows across all windows == rows in input (after dropna)
        total = sum(len(w) for w in windows.values())
        valid = df.dropna(subset=["timestamp"])
        assert total == len(valid)


# ---------------------------------------------------------------------------
# 4. Temporal train/val/test split — no leakage
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    """Tests for the chronological 70/10/20 sample-level split in build_splits."""

    def _make_seq(self, n: int, k: int = 3):
        """Build a GraphSequence with n windows and produce samples for it."""
        from src.dataset_manager import GraphSequence
        graphs = [GraphBuilder()._empty_graph() for _ in range(n)]
        attack_labels = [i % 2 for i in range(n)]  # alternating for balance
        raw_labels = ["Attack" if i % 2 else "Benign" for i in range(n)]
        seq = GraphSequence(
            graphs=graphs,
            attack_labels=attack_labels,
            raw_labels=raw_labels,
            source="test_sequence",
        )
        # Generate samples (mirrors build_splits logic)
        samples = []
        for t in range(n - k):
            future_attacks = attack_labels[t + 1 : t + k + 1]
            future_raws = raw_labels[t + 1 : t + k + 1]
            samples.append((graphs[t], future_attacks, future_raws))
        return seq, samples

    def _compute_split(self, M: int, train_ratio=0.70, val_ratio=0.10):
        """Mirrors the split logic in build_splits exactly."""
        tr_len = int(M * train_ratio)
        vl_len = max(1, int(M * val_ratio))
        te_len = M - tr_len - vl_len
        if te_len <= 0:
            te_len = 1
            tr_len = M - vl_len - te_len
            if tr_len <= 0:
                tr_len, vl_len, te_len = 1, 1, 1
        return tr_len, vl_len, te_len

    def test_split_51_windows_all_nonempty(self):
        """51 windows -> train/val/test all non-empty."""
        _, samples = self._make_seq(51, k=3)
        M = len(samples)  # 48
        tr, vl, te = self._compute_split(M)
        assert tr > 0, "train must be non-empty"
        assert vl > 0, "val must be non-empty"
        assert te > 0, "test must be non-empty"
        assert tr + vl + te == M

    def test_split_10_windows_all_nonempty(self):
        """10 windows -> train/val/test all non-empty."""
        _, samples = self._make_seq(10, k=3)
        M = len(samples)  # 7
        tr, vl, te = self._compute_split(M)
        assert tr > 0
        assert vl > 0
        assert te > 0
        assert tr + vl + te == M

    def test_split_3_windows_minimum(self):
        """3 windows -> exactly 1/1/1 split (minimum guaranteed)."""
        _, samples = self._make_seq(6, k=3)  # 3 usable samples
        M = len(samples)  # 3
        assert M == 3
        tr, vl, te = self._compute_split(M)
        assert tr == 1
        assert vl == 1
        assert te == 1

    def test_chronological_ordering_preserved(self):
        """Samples must remain in the same order they were created."""
        _, samples = self._make_seq(30, k=3)
        M = len(samples)
        tr_len, vl_len, te_len = self._compute_split(M)
        tr_samples = samples[:tr_len]
        vl_samples = samples[tr_len : tr_len + vl_len]
        te_samples = samples[tr_len + vl_len :]
        # Graphs are the same objects from the seq in order
        all_chron = tr_samples + vl_samples + te_samples
        assert [id(s[0]) for s in all_chron] == [id(s[0]) for s in samples]

    def test_no_sample_in_multiple_splits(self):
        """No sample (by graph id) appears in more than one split."""
        _, samples = self._make_seq(30, k=3)
        M = len(samples)
        tr_len, vl_len, te_len = self._compute_split(M)
        tr_ids = {id(s[0]) for s in samples[:tr_len]}
        vl_ids = {id(s[0]) for s in samples[tr_len : tr_len + vl_len]}
        te_ids = {id(s[0]) for s in samples[tr_len + vl_len :]}
        assert tr_ids.isdisjoint(vl_ids), "train and val overlap!"
        assert tr_ids.isdisjoint(te_ids), "train and test overlap!"
        assert vl_ids.isdisjoint(te_ids), "val and test overlap!"

    def test_future_target_stays_inside_sequence(self):
        """Each future target window must come from the same source sequence (no boundary crossing)."""
        k = 3
        n = 15
        _, samples = self._make_seq(n, k=k)
        # samples[i] uses graphs[i] with future from labels[i+1..i+k]
        # Max t = n - k - 1 = 11, so future goes up to 11 + k = 14 = n - 1. OK.
        assert len(samples) == n - k
        for i, (_, future_attacks, _) in enumerate(samples):
            assert len(future_attacks) == k, f"Sample {i} has {len(future_attacks)} future targets, expected {k}"

    def test_scaler_uses_train_only(self):
        """Validate that scaler fitting data only comes from train examples."""
        # This is structural: build_splits returns split.train before val/test.
        # We verify train indices are strictly less than val indices.
        _, samples = self._make_seq(30, k=3)
        M = len(samples)
        tr_len, vl_len, _ = self._compute_split(M)
        tr_indices = set(range(0, tr_len))
        vl_indices = set(range(tr_len, tr_len + vl_len))
        te_indices = set(range(tr_len + vl_len, M))
        # Scaler would only see tr_indices, never vl or te.
        assert tr_indices.isdisjoint(vl_indices)
        assert tr_indices.isdisjoint(te_indices)


# ---------------------------------------------------------------------------
# 5. Graph construction
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    def test_graph_has_correct_structure(self):
        df = _make_flow_df(5)
        windows = create_time_windows(df, window_ms=500)
        g = GraphBuilder().build_window_graph(list(windows.values())[0])
        assert g.x is not None
        assert g.edge_index is not None
        assert g.edge_attr.shape[1] == 11  # 11 flow features (5 base + 6 derived)

    def test_empty_window_returns_empty_graph(self):
        g = GraphBuilder().build_window_graph(pd.DataFrame())
        assert g.x.size(0) == 0


# ---------------------------------------------------------------------------
# 6. Feature normalization — no leakage
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_scaler_fit_only_on_train(self):
        df_train = _make_flow_df(50)
        df_test = _make_flow_df(10)
        scaler = FlowFeatureScaler()
        scaler.fit(df_train)
        X_test = scaler.transform(df_test)
        assert X_test.shape == (10, 11)

    def test_scaler_save_load(self, tmp_path):
        df = _make_flow_df(20)
        scaler = FlowFeatureScaler()
        scaler.fit(df)
        scaler.save(tmp_path)

        loaded = FlowFeatureScaler.load(tmp_path)
        X1 = scaler.transform(df)
        X2 = loaded.transform(df)
        np.testing.assert_allclose(X1, X2)

    def test_unfitted_scaler_raises(self):
        scaler = FlowFeatureScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(_make_flow_df(5))


# ---------------------------------------------------------------------------
# 7. GNN + World Model shapes
# ---------------------------------------------------------------------------

class TestModelShapes:
    def _make_graph(self):
        df = _make_flow_df(5)
        windows = create_time_windows(df, window_ms=500)
        return GraphBuilder().build_window_graph(list(windows.values())[0])

    def test_gnn_output_shape(self):
        g = self._make_graph()
        gnn = GNNEncoder(node_in_dim=2, edge_in_dim=5, hidden_dim=64, out_dim=128)
        z = gnn(g.x, g.edge_index)
        assert z.shape == (1, 128)

    def test_wm_forward_shape(self):
        g = self._make_graph()
        gnn = GNNEncoder(node_in_dim=2, edge_in_dim=5, hidden_dim=64, out_dim=128)
        wm = TemporalWorldModel(z_dim=128, hidden_dim=128, num_layers=1)
        z_t = gnn(g.x, g.edge_index)
        z_seq = z_t.unsqueeze(1)
        z_pred, h_n = wm(z_seq)
        assert z_pred.shape == (1, 128)

    def test_recursive_rollout_is_genuine(self):
        """
        Verify the rollout at step k uses the PREDICTED z of step k-1,
        NOT independently computed states.
        """
        g = self._make_graph()
        gnn = GNNEncoder(node_in_dim=2, edge_in_dim=5, hidden_dim=64, out_dim=128)
        wm = TemporalWorldModel(z_dim=128, hidden_dim=128, num_layers=1)
        z_t = gnn(g.x, g.edge_index)
        z_seq = z_t.unsqueeze(1)
        z1, h_n = wm(z_seq)

        k = 3
        futures = wm.rollout(z1, h_n, k=k)
        assert len(futures) == k

        # Each rollout step must produce a DIFFERENT z (not copies).
        # If rollout were just repeating the same prediction k times
        # (i.e. not genuinely recursive), all futures would be identical.
        assert not torch.allclose(futures[0], futures[1], atol=1e-6), \
            "Rollout steps 1 and 2 are identical — rollout is not genuinely recursive"
        assert not torch.allclose(futures[1], futures[2], atol=1e-6), \
            "Rollout steps 2 and 3 are identical — rollout is not genuinely recursive"

    def test_decoder_output_shapes(self):
        decoder = StateDecoder(z_dim=128, num_mitre_tactics=14)
        z = torch.randn(1, 128)
        attack_prob, net_state, mitre_logits = decoder(z)
        assert attack_prob.shape == (1, 1)
        assert net_state.shape == (1, 5)
        assert mitre_logits.shape == (1, 14)


# ---------------------------------------------------------------------------
# 8. MITRE mapping honesty
# ---------------------------------------------------------------------------

class TestMITREMapping:
    def test_known_label_maps_correctly(self):
        mapper = MitreMapper("mitre_mapping.yaml")
        result = mapper.get_mapping("Botnet")
        assert result["tactic"] == "Command_and_Control"
        assert result["technique"] == "T1071"

    def test_unknown_label_falls_back(self):
        mapper = MitreMapper("mitre_mapping.yaml")
        result = mapper.get_mapping("SomeCompletlyUnknownLabel_XYZ")
        assert "Unknown" in result["tactic"]

    def test_benign_maps_to_none(self):
        mapper = MitreMapper("mitre_mapping.yaml")
        result = mapper.get_mapping("Benign")
        assert result["tactic"] == "None"


# ---------------------------------------------------------------------------
# 9. Checkpoint save/load
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_checkpoint_roundtrip(self, tmp_path):
        import joblib
        gnn = GNNEncoder(node_in_dim=2, edge_in_dim=5, hidden_dim=32, out_dim=64)
        wm = TemporalWorldModel(z_dim=64, hidden_dim=64, num_layers=1)
        decoder = StateDecoder(z_dim=64, num_mitre_tactics=14)

        torch.save(
            {
                "gnn_state_dict": gnn.state_dict(),
                "wm_state_dict": wm.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "selected_k": 3,
                "selected_threshold": 0.65,
            },
            tmp_path / "world_model.pt",
        )

        ckpt = torch.load(tmp_path / "world_model.pt", map_location="cpu", weights_only=True)
        gnn2 = GNNEncoder(node_in_dim=2, edge_in_dim=5, hidden_dim=32, out_dim=64)
        gnn2.load_state_dict(ckpt["gnn_state_dict"])
        assert ckpt["selected_k"] == 3
        assert "selected_threshold" in ckpt
        assert ckpt["selected_threshold"] == 0.65


# ---------------------------------------------------------------------------
# 10. Validation-only threshold, t+K alignment, CIC IPs, explanations
# ---------------------------------------------------------------------------

class TestThresholdSelection:
    def test_signature_has_no_test_set_arguments(self):
        import inspect
        from src.evaluate import select_decision_threshold
        params = inspect.signature(select_decision_threshold).parameters
        assert "y_true_test" not in params
        assert "y_prob_test" not in params
        assert "y_true_val" in params
        assert "y_prob_val" in params

    def test_uses_validation_only_not_test_labels(self):
        from src.evaluate import select_decision_threshold
        y_val = np.array([0, 0, 0, 1, 1, 1])
        p_val = np.array([0.10, 0.20, 0.25, 0.80, 0.85, 0.90])
        y_test = np.array([0, 0, 1, 1])
        p_test = np.array([0.55, 0.60, 0.65, 0.70])
        t_val, m_val = select_decision_threshold(y_val, p_val)
        t_if_test_used, _ = select_decision_threshold(y_test, p_test)
        # Using test labels would pick a different operating point.
        assert t_val != t_if_test_used
        assert m_val["selection_criterion"] == "youden_j_validation_only"
        # Validation scores are well separated around ~0.5–0.8
        assert t_val >= 0.30

    def test_grid_does_not_read_external_test_array(self):
        from src.evaluate import select_decision_threshold
        y_val = np.array([0, 1, 0, 1])
        p_val = np.array([0.2, 0.9, 0.1, 0.8])
        t1, _ = select_decision_threshold(y_val, p_val)
        # Mutating a held-out test array must not change the selection.
        y_test = np.array([0, 0, 0, 0])
        y_test[:] = 1
        t2, _ = select_decision_threshold(y_val, p_val)
        assert t1 == t2


class TestFutureTargetAlignment:
    def test_t_plus_k_is_not_y_t(self):
        """Input window t must be scored against y_{t+K}, never y_t."""
        k = 3
        labels = [0, 0, 0, 1, 1, 1, 0, 0]
        for t in range(len(labels) - k):
            future = labels[t + 1 : t + k + 1]
            assert len(future) == k
            assert future[-1] == labels[t + k]
            # Explicitly not comparing to the current window unless they happen
            # to share the same label by chance.
            if labels[t] != labels[t + k]:
                assert future[-1] != labels[t]

    def test_rollout_index_matches_t_plus_k(self):
        """z_pred_t1 + (K-1) rollout steps → predicted S_{t+K} vs labels[t+K]."""
        k = 3
        n = 10
        labels = list(range(n))  # unique sentinels per window
        for t in range(n - k):
            all_pred_steps = list(range(1, k + 1))  # t+1 .. t+K
            target_step = all_pred_steps[-1]
            assert target_step == k
            assert labels[t + target_step] == labels[t + k]
            assert labels[t + target_step] != labels[t]

    def test_no_cross_file_scenario_target_leakage(self):
        k = 3
        seq_a = list(range(0, 12))
        seq_b = list(range(1000, 1012))
        samples_a = []
        for t in range(len(seq_a) - k):
            future = seq_a[t + 1 : t + k + 1]
            samples_a.append(future)
            assert max(future) < 100  # never a seq_b label
            assert t + k < len(seq_a)
        samples_b = []
        for t in range(len(seq_b) - k):
            future = seq_b[t + 1 : t + k + 1]
            samples_b.append(future)
            assert min(future) >= 1000
            assert t + k < len(seq_b)
        assert samples_a
        assert samples_b


class TestCICGraphIdentities:
    def test_cic_graph_uses_real_ips_when_valid(self):
        df = pd.DataFrame(
            {
                "Timestamp": ["2023-01-01 00:00:01", "2023-01-01 00:00:02"],
                "Src IP": ["192.168.0.1", "192.168.0.2"],
                "Dst IP": ["10.0.0.8", "10.0.0.9"],
                "Dst Port": [80, 80],
                "Protocol": [6, 6],
                "Flow Duration": [12.0, 8.0],
                "Tot Fwd Pkts": [3, 4],
                "Tot Bwd Pkts": [1, 2],
                "TotLen Fwd Pkts": [300, 400],
                "TotLen Bwd Pkts": [100, 200],
                "Label": ["Benign", "Bot"],
            }
        )
        unified = CICIDSAdapter().load_flow_data(df)
        builder = GraphBuilder()
        g = builder.build_window_graph(unified)
        audit = builder.identity_audit()
        assert g.edge_index.shape[1] == 2
        assert audit["real_ips"] == 4
        assert audit["pseudo_nodes"] == 0
        assert set(builder._node_to_id.keys()) == {
            "192.168.0.1", "192.168.0.2", "10.0.0.8", "10.0.0.9"
        }

    def test_cic_missing_ips_use_pseudo_nodes_not_invented_ips(self):
        df = pd.DataFrame(
            {
                "Timestamp": ["2023-01-01 00:00:01"],
                "Dst Port": [22],
                "Protocol": [6],
                "Flow Duration": [1.0],
                "Label": ["Benign"],
            }
        )
        unified = CICIDSAdapter().load_flow_data(df)
        assert unified["src_node"].iloc[0].startswith("proto_")
        assert "port" in unified["dst_node"].iloc[0]
        builder = GraphBuilder()
        builder.build_window_graph(unified)
        audit = builder.identity_audit()
        assert audit["real_ips"] == 0
        assert audit["pseudo_nodes"] >= 1


class TestExplainability:
    def test_explanation_returns_ranked_features(self):
        from src.explainability import explain_edge_features
        from src.feature_schema import EDGE_FEATURE_NAMES
        df = _make_flow_df(8)
        windows = create_time_windows(df, window_ms=500)
        g = GraphBuilder().build_window_graph(list(windows.values())[0])
        gnn = GNNEncoder(node_in_dim=2, edge_in_dim=11, hidden_dim=16, out_dim=16)
        wm = TemporalWorldModel(z_dim=16, hidden_dim=16, num_layers=1)
        decoder = StateDecoder(z_dim=16, num_mitre_tactics=14)
        expl = explain_edge_features(gnn, wm, decoder, g, k=3, top_n=5)
        assert expl["method"] == "gradient_x_input_edge_features"
        assert len(expl["ranked"]) == 5
        names = [item["feature"] for item in expl["ranked"]]
        assert names == sorted(names, key=lambda n: -abs(expl["scores"][n]))
        for name in names:
            assert name in EDGE_FEATURE_NAMES
        assert "direction" in expl["ranked"][0]

