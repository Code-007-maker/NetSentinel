import pandas as pd
import pytest

from src.inference import prepare_input, validate_upload, load_bundle


def test_upload_validation_rejects_empty_and_unsupported():
    assert validate_upload("capture.txt", b"x")
    assert validate_upload("capture.csv", b"")
    assert validate_upload("capture.csv", b"a,b\n1,2\n") is None


def test_csv_ingestion_uses_unified_schema():
    raw = pd.DataFrame({
        "Timestamp": ["2018-01-01 00:00:00", "2018-01-01 00:01:00"],
        "Src IP": ["10.0.0.1", "10.0.0.2"], "Dst IP": ["10.0.0.3", "10.0.0.4"],
        "Protocol": [6, 6], "Flow Duration": [10, 12], "Tot Fwd Pkts": [2, 3],
        "Tot Bwd Pkts": [1, 1], "TotLen Fwd Pkts": [100, 150],
        "TotLen Bwd Pkts": [50, 60], "Label": ["Benign", "Attack"],
    })
    result = prepare_input("uploaded.csv", raw.to_csv(index=False).encode())
    assert {"timestamp", "src_node", "dst_node", "label"}.issubset(result.columns)
    assert len(result) == 2


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="MODEL NOT AVAILABLE"):
        load_bundle(tmp_path)


def test_pcap_ingestion_uses_parser(monkeypatch):
    monkeypatch.setattr("src.inference.extract_pcap_features", lambda *_args, **_kwargs: pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01"]), "src_ip": ["10.0.0.1"],
        "dst_ip": ["10.0.0.2"], "protocol": [6], "duration": [1.0],
        "packets_forward": [1], "packets_backward": [0], "bytes_forward": [100],
        "bytes_backward": [0], "label": ["Unknown"],
    }))
    result = prepare_input("capture.pcap", b"pcap-bytes")
    assert len(result) == 1
    assert "duration" in result.columns
