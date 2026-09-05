# Dataset Inventory

## 1. CIC-IDS2018
- **Path:** `./datasets/CIC-IDS2018/csv/`
- **File Types:** `.csv`
- **Files Found:** 10 flow-based CSV files, including `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`, `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv`, etc.
- **Available PCAPs:** None directly found in the directory.
- **Key Columns:** `Dst Port`, `Protocol`, `Timestamp`, `Flow Duration`, `Tot Fwd Pkts`, `Tot Bwd Pkts`, `Flow Byts/s`, `Pkt Size Avg`, `Label`, etc.
- **Notes:** High-dimensional flow features extracted by CICFlowMeter. Includes standard network stats and temporal metrics (IAT).

## 2. CTU-13 Dataset
- **Path:** `./datasets/CTU-13-Dataset/`
- **Scenarios Found:** 1, 3, 4, 5, 7, 9, 10, 12, 13
- **File Types:** `.pcap`, `.binetflow` (CSV-like)
- **Files in Scenario 1 (example):** `botnet-capture-20110810-neris.pcap`, `capture20110810.binetflow`
- **Key Columns (binetflow):** `StartTime`, `Dur`, `Proto`, `SrcAddr`, `Sport`, `Dir`, `DstAddr`, `Dport`, `State`, `TotPkts`, `TotBytes`, `SrcBytes`, `Label`.
- **Notes:** Contains both PCAP and binetflow. Excellent for botnet modeling.

## 3. UNSW-NB15
- **Path:** `./datasets/UNSW-NB15/`
- **File Types:** `.csv`, `.pdf`
- **Files Found:** `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`, `UNSW-NB15_1.csv` ... `UNSW-NB15_4.csv`
- **Key Columns:** `id`, `dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `attack_cat`, `label`, etc.
- **Notes:** High-quality structured flow dataset. Includes specific attack categories (`attack_cat`) and a binary `label`. No raw PCAP found directly in the immediate folder.

## Summary & Action Plan
- **Unified Telemetry:** We will map columns from these 3 distinct CSV formats into a single `Unified Telemetry Schema` representing the fields mentioned in the core requirements (src/dst IPs, ports, proto, timestamps, packet/byte counts, etc.).
- **PCAP Support:** CTU-13 has PCAP files which will be parsed to demonstrate packet-level feature extraction.
- **Graph Nodes & Edges:** Features like `SrcAddr` and `DstAddr` (or inferred pairs in CIC-IDS/UNSW) will act as nodes, while flow metrics will act as node/edge features. Timestamps will enable chronological windowing.
