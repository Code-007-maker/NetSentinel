import os
from pathlib import Path
from typing import Dict, List

def discover_datasets(base_path: str) -> Dict[str, List[Path]]:
    """
    Recursively discovers dataset files (.csv, .pcap, .binetflow) in the given base path.
    """
    base = Path(base_path)
    if not base.exists():
        print(f"Warning: Base dataset path {base_path} does not exist.")
        return {}

    inventory = {
        'csv': [],
        'pcap': [],
        'binetflow': []
    }

    print(f"Scanning {base_path} for datasets...")
    for root, _, files in os.walk(base):
        for file in files:
            path = Path(root) / file
            if file.endswith('.csv'):
                inventory['csv'].append(path)
            elif file.endswith('.pcap'):
                inventory['pcap'].append(path)
            elif file.endswith('.binetflow'):
                inventory['binetflow'].append(path)

    for dtype, paths in inventory.items():
        print(f"Found {len(paths)} {dtype} files.")
        for p in paths[:3]: # Print first 3 as examples
            print(f"  - {p}")
        if len(paths) > 3:
            print("  - ...")

    return inventory

if __name__ == "__main__":
    from src.config import CONFIG
    dataset_dir = CONFIG['paths']['dataset_dir']
    discover_datasets(dataset_dir)
