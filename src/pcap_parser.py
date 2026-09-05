import pandas as pd
import numpy as np
from pathlib import Path
from scapy.all import PcapReader, IP, TCP, UDP
import logging

logger = logging.getLogger(__name__)

def _safe_var(values):
    if len(values) < 2:
        return 0.0
    return np.var(values, ddof=1)

def extract_pcap_features(pcap_path: str, max_packets: int = 100000) -> pd.DataFrame:
    """
    Reads a PCAP file, extracts packet-level features, and aggregates them into flows.
    """
    logger.info(f"Extracting packet features from {pcap_path}")
    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    # Flow aggregation dictionary
    flows = {}
    count = 0
    
    try:
        with PcapReader(str(path)) as pcap_reader:
            for pkt in pcap_reader:
                if count >= max_packets:
                    break
                    
                if IP in pkt:
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                    proto = pkt[IP].proto
                    length = len(pkt)
                    timestamp = float(pkt.time)
                    ttl = pkt[IP].ttl
                    frag = pkt[IP].frag
                    
                    src_port = 0
                    dst_port = 0
                    protocol_str = 'unknown'
                    tcp_window = 0
                    payload_len = 0
                    is_retransmission = 0 # Naive heuristic if we wanted
                    
                    if TCP in pkt:
                        src_port = pkt[TCP].sport
                        dst_port = pkt[TCP].dport
                        protocol_str = 'tcp'
                        tcp_window = pkt[TCP].window
                        payload_len = len(pkt[TCP].payload)
                        # Very naive retransmission heuristic could be checking for specific flags/sequences,
                        # but without stateful sequence tracking it's an approximation.
                    elif UDP in pkt:
                        src_port = pkt[UDP].sport
                        dst_port = pkt[UDP].dport
                        protocol_str = 'udp'
                        payload_len = len(pkt[UDP].payload)
                    
                    # Flow key is strictly unidirectional for metric tracking, 
                    # but often network datasets use bidirectional. We will use unidirectional for simplicity
                    # and group bidirectional later if needed.
                    flow_key = (src_ip, dst_ip, src_port, dst_port, protocol_str)
                    
                    if flow_key not in flows:
                        flows[flow_key] = {
                            'src_ip': src_ip,
                            'dst_ip': dst_ip,
                            'src_port': src_port,
                            'dst_port': dst_port,
                            'protocol': protocol_str,
                            
                            'start_time': timestamp,
                            'end_time': timestamp,
                            'packets_forward': 0,
                            'bytes_forward': 0,
                            
                            'ttls': [],
                            'tcp_windows': [],
                            'frag_flags': [],
                            'payload_sizes': [],
                            'timestamps': []
                        }
                    
                    f = flows[flow_key]
                    f['end_time'] = timestamp
                    f['packets_forward'] += 1
                    f['bytes_forward'] += length
                    f['ttls'].append(ttl)
                    if protocol_str == 'tcp':
                        f['tcp_windows'].append(tcp_window)
                    f['frag_flags'].append(frag)
                    f['payload_sizes'].append(payload_len)
                    f['timestamps'].append(timestamp)
                    
                count += 1
                
    except Exception as exc:
        # Invalid / empty PCAP files (e.g. zero-byte, wrong magic number) must
        # return an empty DataFrame rather than propagating the exception.
        logger.warning("Could not read PCAP %s: %s", path, exc)
        return pd.DataFrame()
            
    # Process flows into final schema
    records = []
    for f in flows.values():
        duration = max(0.0, f['end_time'] - f['start_time'])
        
        # Inter-arrival times
        timestamps = sorted(f['timestamps'])
        iats = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        mean_iat = np.mean(iats) if iats else 0.0
        
        ttl_var = _safe_var(f['ttls'])
        
        # Append to records
        records.append({
            'timestamp': pd.to_datetime(f['start_time'], unit='s'),
            'src_ip': f['src_ip'],
            'dst_ip': f['dst_ip'],
            'src_port': f['src_port'],
            'dst_port': f['dst_port'],
            'protocol': f['protocol'],
            'duration': duration,
            'packets_forward': f['packets_forward'],
            'packets_backward': 0, # Since we did unidirectional flows
            'bytes_forward': f['bytes_forward'],
            'bytes_backward': 0,
            
            # PCAP specific advanced features
            'mean_ttl': np.mean(f['ttls']),
            'ttl_var': ttl_var,
            'mean_tcp_window': np.mean(f['tcp_windows']) if f['tcp_windows'] else 0.0,
            'mean_payload_size': np.mean(f['payload_sizes']),
            'frag_count': sum(1 for x in f['frag_flags'] if x > 0),
            'mean_iat': mean_iat,
            
            'label': 'Unknown', # Inherently unknown for raw PCAP unless externally mapped
            'source_dataset': 'PCAP-Upload'
        })
        
    df = pd.DataFrame(records)
    return df
