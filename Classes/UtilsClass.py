import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

class DataUtils:
    NAME_MAP = {
        "Rueben": "Reuben",
        "Te": "Te Reimana",
        "Hybrid 3": "Jimmy",
        "Wheturangi": "Whetu",
        "Sam": "Sam G",
        "Joshua": "Josh",
        "Samuel": "Sam G"    
    }

    @staticmethod
    def is_hybrid4_name(name: str) -> bool:
        if not name:
            return False
        s = ''.join(name.split()).lower()
        return s == 'hybrid4' or s == 'hybrid2'

    @classmethod
    def parse_name(cls, xml_name: str) -> Optional[str]:
        if not xml_name:
            return None
        if ',' in xml_name:
            first_name = xml_name.split(',', 1)[1].strip()
        else:
            first_name = xml_name.strip()
        
        mapped = cls.NAME_MAP.get(first_name, first_name)
        if cls.is_hybrid4_name(mapped):
            return None
        return mapped

    @staticmethod
    def parse_datetime(timestamp_str: str) -> Optional[datetime]:
        if not timestamp_str:
            return None
        patterns = (
            '%Y-%m-%d %I:%M:%S.%f %p %z', 
            '%Y-%m-%d %I:%M:%S %p %z', 
            '%Y-%m-%d %H:%M:%S %z',
            '%Y-%m-%d %H:%M:%S %z', # ISO-like
            '%d/%m/%Y %H:%M:%S %z'  # Fallback
        )
        
        # Pre-normalization for ISO timezones if needed
        if timestamp_str and ':' in timestamp_str[-5:] and timestamp_str[-3] == ':':
             # Simple heuristic for colon in timezone offset
             pass 

        for fmt in patterns:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # Manual fallback for weird timezone formats if needed
        try:
            # Attempt to strip colon from timezone if parsing failed
            # This is a basic implementation of the logic in the original script
            if '+' in timestamp_str or '-' in timestamp_str:
                return datetime.strptime(timestamp_str.replace(':', ''), '%Y-%m-%d %H:%M:%S %z')
        except:
            pass
            
        return None

