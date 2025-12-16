import yaml
import os
import sys
from typing import Dict, List, Tuple, Optional, Any

class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            print(f"Error: config file not found at {self.config_path}")
            sys.exit(1)
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def csv_path(self): return self.data.get('CSV_path')
    
    @property
    def xml_path(self): return self.data.get('XML_path')
    
    @property
    def pla_threshold(self): return self.data.get('PLA_threshold', 14.0)
    
    @property
    def alignment_threshold(self): return self.data.get('Alignment_threshold', 5.0)
    
    @property
    def ref_csv_path(self): return self.data.get('Reference_Impacts_CSV_path')

    @property
    def schedule_path(self): return self.data.get('schedule_path')
