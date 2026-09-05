import yaml
from pathlib import Path
from typing import Dict, Tuple

class MitreMapper:
    def __init__(self, mapping_file: str = "mitre_mapping.yaml"):
        path = Path(mapping_file)
        if not path.exists():
            self.mapping = {"dataset_mapping": {}, "fallback": {"tactic": "Unknown / Unsupported", "technique": "Unknown / Unsupported"}}
        else:
            with open(path, 'r', encoding='utf-8') as f:
                self.mapping = yaml.safe_load(f)
                
        self.dataset_mapping = self.mapping.get('dataset_mapping', {})
        self.fallback = self.mapping.get('fallback', {"tactic": "Unknown / Unsupported", "technique": "Unknown / Unsupported"})
        
    def get_mapping(self, label: str) -> Dict[str, str]:
        """
        Maps a dataset label to an evidence-based MITRE ATT&CK tactic and technique.
        Returns fallback if unsupported to prevent fabricating techniques.
        """
        if label == "Benign" or label == "Background":
            return {"tactic": "None", "technique": "None"}
            
        # Try direct match
        if label in self.dataset_mapping:
            return self.dataset_mapping[label]
            
        # Try substring match
        for key, value in self.dataset_mapping.items():
            if key.lower() in label.lower():
                return value
                
        return self.fallback
