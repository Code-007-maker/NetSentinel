"""
Dataset adapters: CIC-IDS2018, UNSW-NB15, CTU-13.

Key design decisions documented per adapter:

CIC-IDS2018
  - Most CSE-CIC-IDS2018 CICFlowMeter CSVs have no host IPs
    (Dst Port, Protocol, Timestamp, Flow Duration, Tot Fwd Pkts, ...).
  - The Tuesday-20-02-2018 file (and any CICFlowMeter export that includes
    them) has real host columns: Src IP, Dst IP (also Source IP / Destination IP).
  - When a valid source/destination IP string is present, it is used as the
    graph node identity.  Pseudo-nodes from (Protocol, Dst Port) are used
    ONLY when the IP identity cannot be recovered.  IPs are never invented.

UNSW-NB15  (two schemas)
  - UNSW-NB15_{1..4}.csv: headerless raw CSVs, 49 columns.
    First data row = row 0 (NOT header). Read with header=None + explicit names.
  - UNSW_NB15_training-set.csv / testing-set.csv: headered CSVs, no IP columns.
    Use (proto, service) as pseudo-node IDs.

CTU-13
  - binetflow files have real SrcAddr / DstAddr IP columns.
  - Labels contain "flow=" prefix; stripped for clarity.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.feature_schema import EDGE_FEATURE_NAMES

logger = logging.getLogger(__name__)

# ── label sets ────────────────────────────────────────────────────────────────
BENIGN_STRINGS = {"benign", "background", "normal", "0", ""}

# Values that mean "no recoverable endpoint identity" (not valid IP strings).
_IP_MISSING = {"", "nan", "none", "null", "nat", "-", "<na>", "n/a"}

CIC_SRC_IP_COLUMNS = [
    "Src IP", "Source IP", "src_ip", "SrcIP", "SourceIP", "srcip",
]
CIC_DST_IP_COLUMNS = [
    "Dst IP", "Destination IP", "dst_ip", "DstIP", "DestinationIP", "dstip",
]


def _missing_identity(series: pd.Series) -> pd.Series:
    """True where an IP/identity cell cannot be used as a node id."""
    na = series.isna()
    text = series.astype(str).str.strip()
    return na | text.str.lower().isin(_IP_MISSING)


def resolve_endpoint_nodes(
    ip_series: Optional[pd.Series],
    fallback: pd.Series,
) -> Tuple[pd.Series, int]:
    """
    Use recoverable IP/identity strings; otherwise the provided pseudo-node fallback.

    Does not invent IPs.  Returns (node_ids, n_invalid_or_missing).
    """
    if ip_series is None:
        return fallback.astype(str), int(len(fallback))
    missing = _missing_identity(ip_series)
    nodes = ip_series.astype(str).str.strip()
    nodes = nodes.where(~missing, fallback.astype(str))
    return nodes, int(missing.sum())

# ── UNSW-NB15 raw CSV column schema (49 columns, headerless files) ────────────
UNSW_RAW_COLUMNS: List[str] = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]  # 49 columns

# ── UnifiedTelemetry schema ───────────────────────────────────────────────────

class UnifiedTelemetry:
    """Unified Schema for network telemetry."""
    COLUMNS = [
        "timestamp", "src_node", "dst_node",
        "protocol", "duration",
        "packets_forward", "packets_backward",
        "bytes_forward", "bytes_backward",
        "label", "source_dataset",
    ] + EDGE_FEATURE_NAMES   # derived features appended


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce to float, replace inf with NaN, fill NaN with 0."""
    return (pd.to_numeric(series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float))


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Division safe against zero denominators. No RuntimeWarning emitted."""
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=(den != 0))
    return out


def _proto_to_int(proto: pd.Series) -> pd.Series:
    """Deterministically encode protocol string to integer."""
    PROTO_MAP = {"tcp": 6, "udp": 17, "icmp": 1}
    return proto.astype(str).str.lower().map(PROTO_MAP).fillna(0).astype(float)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 6 derived flow features from the 5 base columns.
    All arithmetic is performed on already-sanitized float arrays.
    Returns df with EDGE_FEATURE_NAMES columns present and finite.
    """
    for col in ["duration", "packets_forward", "packets_backward",
                "bytes_forward", "bytes_backward"]:
        df[col] = _safe_numeric(df.get(col, pd.Series(0.0, index=df.index)))

    dur  = df["duration"].values
    pfwd = df["packets_forward"].values
    pbwd = df["packets_backward"].values
    bfwd = df["bytes_forward"].values
    bbwd = df["bytes_backward"].values

    df["rate_fwd"]         = _safe_div(pfwd, dur)
    df["rate_bwd"]         = _safe_div(pbwd, dur)
    df["byte_rate_fwd"]    = _safe_div(bfwd, dur)
    df["byte_rate_bwd"]    = _safe_div(bbwd, dur)
    df["avg_pkt_size_fwd"] = _safe_div(bfwd, pfwd)
    df["avg_pkt_size_bwd"] = _safe_div(bbwd, pbwd)

    # Final sanitize
    for col in EDGE_FEATURE_NAMES:
        df[col] = (df[col].replace([np.inf, -np.inf], np.nan)
                   .fillna(0.0).astype(float))
    return df


# ── Base adapter ──────────────────────────────────────────────────────────────

class DatasetAdapter(ABC):
    """Base class for all dataset adapters."""

    def _find_column(
        self, df: pd.DataFrame, candidates: List[str],
        required: bool = False, default=None
    ) -> pd.Series:
        for col in candidates:
            if col in df.columns:
                return df[col]
        if required:
            raise ValueError(
                f"Required column not found. Tried: {candidates}. "
                f"Available: {df.columns.tolist()}"
            )
        return pd.Series(default, index=df.index, dtype="object")

    @abstractmethod
    def load_flow_data(self, file_path: Path) -> pd.DataFrame:
        """Loads and converts flow data into the unified schema."""


# ── CIC-IDS2018 adapter ───────────────────────────────────────────────────────

class CICIDSAdapter(DatasetAdapter):
    """
    CIC-IDS2018 / CICFlowMeter adapter.

    Node identity:
      - If Src IP / Dst IP (or Source IP / Destination IP, etc.) exist and the
        cell is a recoverable identity string, use it as the graph node.
      - Otherwise fall back to pseudo-nodes:
          src = "proto_{protocol}"
          dst = "p{protocol}_port{dst_port}"
      IPs are never synthesised.
    """

    def load_flow_data(self, file_path: Union[Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(file_path, pd.DataFrame):
            df = file_path.copy()
            fname = "DataFrame"
        else:
            df = pd.read_csv(file_path, low_memory=False)
            fname = Path(file_path).name

        # Normalize column names: strip whitespace
        df.columns = [c.strip() for c in df.columns]

        unified_df = pd.DataFrame()

        try:
            ts_col = self._find_column(df, ["Timestamp", "Flow Date/Time"], required=True)
            unified_df["timestamp"] = pd.to_datetime(ts_col, format="mixed", errors="coerce")

            proto = self._find_column(df, ["Protocol"], required=False, default=0)
            dst_port = self._find_column(
                df, ["Dst Port", "Destination Port", "DstPort"], required=False, default=0
            )
            proto_s = proto.astype(str).str.strip()
            dst_port_s = dst_port.astype(str).str.strip()
            src_fallback = "proto_" + proto_s
            dst_fallback = "p" + proto_s + "_port" + dst_port_s

            src_ip_col = next((c for c in CIC_SRC_IP_COLUMNS if c in df.columns), None)
            dst_ip_col = next((c for c in CIC_DST_IP_COLUMNS if c in df.columns), None)
            src_ip = df[src_ip_col] if src_ip_col is not None else None
            dst_ip = df[dst_ip_col] if dst_ip_col is not None else None

            src_nodes, n_src_missing = resolve_endpoint_nodes(src_ip, src_fallback)
            dst_nodes, n_dst_missing = resolve_endpoint_nodes(dst_ip, dst_fallback)
            unified_df["src_node"] = src_nodes
            unified_df["dst_node"] = dst_nodes
            unified_df.attrs["cic_ip_columns"] = {
                "src_ip_column": src_ip_col,
                "dst_ip_column": dst_ip_col,
                "src_missing": n_src_missing,
                "dst_missing": n_dst_missing,
            }
            if src_ip_col:
                logger.info(
                    "CIC [%s]: using IP columns src=%s dst=%s (missing src=%d dst=%d / %d rows)",
                    fname, src_ip_col, dst_ip_col, n_src_missing, n_dst_missing, len(df),
                )
            else:
                logger.info(
                    "CIC [%s]: no Src IP/Dst IP columns; using protocol/port pseudo-nodes",
                    fname,
                )

            unified_df["protocol"] = _safe_numeric(proto)
            unified_df["duration"]  = _safe_numeric(
                self._find_column(df, ["Flow Duration"], required=True))
            unified_df["packets_forward"]  = _safe_numeric(
                self._find_column(df, ["Tot Fwd Pkts", "Total Fwd Packets"], required=False, default=0))
            unified_df["packets_backward"] = _safe_numeric(
                self._find_column(df, ["Tot Bwd Pkts", "Total Backward Packets"], required=False, default=0))
            unified_df["bytes_forward"]  = _safe_numeric(
                self._find_column(df, ["TotLen Fwd Pkts", "Fwd Pkt Len Tot", "Total Length of Fwd Packets"], required=False, default=0))
            unified_df["bytes_backward"] = _safe_numeric(
                self._find_column(df, ["TotLen Bwd Pkts", "Bwd Pkt Len Tot", "Total Length of Bwd Packets"], required=False, default=0))

            raw_label = self._find_column(df, ["Label"], required=True)
            unified_df["label"] = raw_label.apply(
                lambda x: "Benign" if str(x).strip().lower() in BENIGN_STRINGS else str(x).strip())

        except ValueError as e:
            raise ValueError(f"CIC-IDS2018 Mapping Error [{fname}]: {e}") from e

        unified_df["source_dataset"] = "CIC-IDS2018"
        unified_df = compute_derived_features(unified_df)
        return unified_df


# ── UNSW-NB15 adapter ─────────────────────────────────────────────────────────

class UNSWAdapter(DatasetAdapter):
    """
    UNSW-NB15 adapter supporting two file schemas:

    Schema A — raw CSV (UNSW-NB15_{1..4}.csv):
      Headerless, 49 columns. First row is data.
      Columns defined in UNSW_RAW_COLUMNS. Contains srcip/dstip.

    Schema B — aggregated sets (UNSW_NB15_training-set.csv, testing-set.csv):
      Has header, no IP columns. Uses (proto, service) pseudo-node IDs.
    """

    # Files to skip entirely
    SKIP_PATTERNS = {"nusw-nb15_features", "unsw-nb15_list_events", ".crdownload"}

    def load_flow_data(self, file_path: Union[Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(file_path, pd.DataFrame):
            df = file_path.copy()
            schema = "unknown"
            fname = "DataFrame"
        else:
            fname = Path(file_path).name
            fname_lower = fname.lower()

            # Skip metadata / non-telemetry files
            if any(p in fname_lower for p in self.SKIP_PATTERNS):
                logger.info("UNSW: skipping non-telemetry file: %s", fname)
                return pd.DataFrame()

            # Determine schema from filename
            if re.match(r"unsw.nb15_\d+\.csv", fname_lower):
                schema = "raw"
            else:
                schema = "headered"

            if schema == "raw":
                df = pd.read_csv(file_path, header=None, names=UNSW_RAW_COLUMNS,
                                 low_memory=False)
                # Verify column count matches expectation
                if df.shape[1] != len(UNSW_RAW_COLUMNS):
                    raise ValueError(
                        f"UNSW raw CSV {fname} has {df.shape[1]} columns, "
                        f"expected {len(UNSW_RAW_COLUMNS)}."
                    )
                logger.info("UNSW [raw schema]: %s (%d rows, %d cols)",
                            fname, len(df), df.shape[1])
            else:
                df = pd.read_csv(file_path, low_memory=False)
                logger.info("UNSW [headered schema]: %s (%d rows)", fname, len(df))

        unified_df = pd.DataFrame()

        try:
            has_ips = "srcip" in df.columns and "dstip" in df.columns

            # Timestamp
            if "stime" in df.columns:
                unified_df["timestamp"] = pd.to_datetime(
                    _safe_numeric(df["stime"]), unit="s", errors="coerce")
            elif "ltime" in df.columns:
                unified_df["timestamp"] = pd.to_datetime(
                    _safe_numeric(df["ltime"]), unit="s", errors="coerce")
            else:
                base_ts = pd.Timestamp("2015-01-22").value // 10**9
                unified_df["timestamp"] = pd.to_datetime(
                    base_ts + np.arange(len(df)), unit="s")

            # Node IDs
            if has_ips:
                unified_df["src_node"] = df["srcip"].astype(str).str.strip()
                unified_df["dst_node"] = df["dstip"].astype(str).str.strip()
            else:
                # Use (proto, service/state) as pseudo-node IDs
                proto_s   = df.get("proto",   pd.Series("unknown", index=df.index)).astype(str)
                service_s = df.get("service", df.get("state", pd.Series("?", index=df.index))).astype(str)
                unified_df["src_node"] = "proto_" + proto_s
                unified_df["dst_node"] = "svc_"   + service_s

            unified_df["protocol"] = _safe_numeric(
                df.get("proto", pd.Series(0, index=df.index))
                if df.get("proto", pd.Series()).dtype != object
                else _proto_to_int(df.get("proto", pd.Series("tcp", index=df.index))))
            unified_df["duration"]          = _safe_numeric(df.get("dur",   pd.Series(0, index=df.index)))
            unified_df["packets_forward"]   = _safe_numeric(df.get("spkts", pd.Series(0, index=df.index)))
            unified_df["packets_backward"]  = _safe_numeric(df.get("dpkts", pd.Series(0, index=df.index)))
            unified_df["bytes_forward"]     = _safe_numeric(df.get("sbytes",pd.Series(0, index=df.index)))
            unified_df["bytes_backward"]    = _safe_numeric(df.get("dbytes",pd.Series(0, index=df.index)))

            # Label
            if "attack_cat" in df.columns and not df["attack_cat"].isna().all():
                unified_df["label"] = df["attack_cat"].fillna("Benign").astype(str)
            elif "label" in df.columns:
                unified_df["label"] = df["label"].apply(
                    lambda x: "Benign" if str(x).strip() in {"0", "Normal", "normal", ""} else "Attack")
            else:
                raise ValueError(f"No label column found. Available: {df.columns.tolist()}")

        except ValueError as e:
            raise ValueError(f"UNSW-NB15 Mapping Error [{fname}]: {e}") from e

        unified_df["source_dataset"] = "UNSW-NB15"
        unified_df = compute_derived_features(unified_df)
        return unified_df


# ── CTU-13 adapter ────────────────────────────────────────────────────────────

class CTU13Adapter(DatasetAdapter):
    """
    CTU-13 binetflow adapter.
    Real IP columns: SrcAddr (src_node), DstAddr (dst_node).
    Labels contain 'flow=…' prefix which is stripped.
    """

    def load_flow_data(self, file_path: Union[Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(file_path, pd.DataFrame):
            df = file_path.copy()
            fname = "DataFrame"
        else:
            df = pd.read_csv(file_path, low_memory=False)
            fname = Path(file_path).name

        unified_df = pd.DataFrame()

        try:
            ts_col = self._find_column(df, ["StartTime"], required=True)
            unified_df["timestamp"] = pd.to_datetime(ts_col, format="mixed", errors="coerce")

            unified_df["src_node"] = self._find_column(df, ["SrcAddr"], required=True).astype(str).str.strip()
            unified_df["dst_node"] = self._find_column(df, ["DstAddr"], required=True).astype(str).str.strip()

            unified_df["protocol"] = _proto_to_int(
                self._find_column(df, ["Proto"], required=False, default="unknown"))
            unified_df["duration"] = _safe_numeric(
                self._find_column(df, ["Dur"], required=False, default=0))
            unified_df["packets_forward"] = _safe_numeric(
                self._find_column(df, ["TotPkts"], required=False, default=0))
            unified_df["packets_backward"] = 0.0

            src_bytes = _safe_numeric(
                self._find_column(df, ["SrcBytes"], required=False, default=0))
            tot_bytes = _safe_numeric(
                self._find_column(df, ["TotBytes"], required=False, default=0))
            unified_df["bytes_forward"]  = src_bytes
            unified_df["bytes_backward"] = (tot_bytes - src_bytes).clip(lower=0)

            raw_label = self._find_column(df, ["Label"], required=True)
            # Strip "flow=" prefix if present; preserve semantic label
            unified_df["label"] = raw_label.astype(str).str.replace(r"^flow=", "", regex=True).str.strip()

        except ValueError as e:
            raise ValueError(f"CTU-13 Mapping Error [{fname}]: {e}") from e

        unified_df["source_dataset"] = "CTU-13"
        unified_df = compute_derived_features(unified_df)
        return unified_df
