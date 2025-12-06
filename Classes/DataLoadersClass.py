import csv
import os
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional
# Import from the sibling file in the same folder
from Classes.UtilsClass import DataUtils

class ReferenceDataManager:
    """Loads reference data used to repair 1970/1980 timestamps."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> Dict[str, Dict[int, datetime]]:
        ref_data = {}
        if not self.filepath or not os.path.exists(self.filepath):
            print(f"Reference CSV not found or not provided: {self.filepath}. Continuing without reference repair.")
            return ref_data

        print(f"Loading reference impacts from: {self.filepath}")
        try:
            try:
                f = open(self.filepath, mode='r', encoding='utf-8-sig')
                f.read(100); f.seek(0)
            except UnicodeDecodeError:
                f = open(self.filepath, mode='r', encoding='utf-16')
            
            with f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    try:
                        dev_id = row.get('Device SN') or row.get('DeviceId') or row.get('DeviceId ')
                        if not dev_id: continue
                        
                        persisted = row.get('Persisted Event #') or row.get('Persisted Event')
                        if not persisted: continue
                        
                        event_id = int(persisted)
                        
                        # Construct datetime string
                        date_part = row.get('Date') or row.get('Event Date') or ''
                        time_part = row.get('Local Time') or row.get('LocalTime') or ''
                        tz_part = row.get('Time Zone From UTC') or row.get('Timezone') or '+00:00'
                        tz_norm = tz_part.replace(':', '') if ':' in tz_part else tz_part
                        
                        dt_str = f"{date_part} {time_part} {tz_norm}"
                        
                        # Attempt parse
                        dt_obj = DataUtils.parse_datetime(dt_str)
                        
                        if dt_obj and dt_obj.year > 1980:
                            ref_data.setdefault(dev_id.strip(), {})[event_id] = dt_obj
                            count += 1
                    except Exception:
                        continue
            print(f"Loaded {count} valid reference events.")
        except Exception as e:
            print(f"Error reading Reference CSV: {e}")
        return ref_data


class CollisionDataManager:
    """Loads Composer CSV data (coded collisions)."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.timestamps: Optional[np.ndarray] = None
        self.identifiers: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        collisions = []
        ids = []
        if not os.path.exists(self.filepath):
            print(f"Error: Composer CSV not found at {self.filepath}")
            return

        try:
            try:
                f = open(self.filepath, mode='r', encoding='utf-8-sig')
                f.read(100); f.seek(0)
            except UnicodeDecodeError:
                f = open(self.filepath, mode='r', encoding='utf-16')
            
            with f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        pb_time = float(row.get('Qualifier: Time', row.get('Qualifier Time', 0)))
                        raw_name = (row.get('Topic Name') or '').strip()
                        parsed = DataUtils.parse_name(raw_name)
                        if parsed:
                            collisions.append(pb_time)
                            ids.append(parsed)
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error reading Composer CSV: {e}")
            return

        print(f"Loaded {len(collisions)} collision events from CSV.")
        self.timestamps = np.array(collisions)
        self.identifiers = np.array(ids)


class SensorDataManager:
    """Loads and repairs XML SAE data."""
    def __init__(self, filepath: str, reference_data: Dict, pla_threshold: float):
        self.filepath = filepath
        self.ref_data = reference_data
        self.pla_threshold = pla_threshold
        
        self.timestamps: Optional[np.ndarray] = None
        self.identifiers: Optional[np.ndarray] = None
        self.session_start_utc: Optional[float] = None
        
        self._load_and_repair()

    def _load_and_repair(self):
        if not os.path.exists(self.filepath):
            print(f"Error: XML file not found at {self.filepath}")
            return

        try:
            tree = ET.parse(self.filepath)
            root = tree.getroot()
            
            # 1. Get Session Start
            self._parse_session_start(root)
            
            # 2. Extract Raw Events
            device_groups = self._extract_events(root)
            
            # 3. Repair Timestamps
            self._repair_timestamps(device_groups)
            
            # 4. Flatten to Arrays
            self._flatten_data(device_groups)

        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
        except Exception as e:
            print(f"Error processing XML: {e}")

    def _parse_session_start(self, root):
        node = root.find('.//SESSION_INFO/start_time')
        if node is not None and node.text:
            dt = DataUtils.parse_datetime(node.text)
            if dt:
                self.session_start_utc = dt.timestamp()
                print(f"Found session start time: {dt.isoformat()}")

    def _extract_events(self, root) -> Dict[str, List[Dict]]:
        groups = {}
        for instance in root.findall('.//instance'):
            data = self._parse_instance(instance)
            if data:
                groups.setdefault(data['dev_id'], []).append(data)
        return groups

    def _parse_instance(self, instance) -> Optional[Dict]:
        meta = {'name': None, 'dt_str': None, 'pla': None, 'dev_id': None, 'id': None}
        
        for label in instance.findall('label'):
            group = label.find('group')
            text_node = label.find('text')
            text = text_node.text if text_node is not None else None
            
            if group is None or text is None: continue
            g_text = group.text
            
            if g_text == 'Athlete Name': meta['name'] = text
            elif g_text == 'ImpactDate': meta['dt_str'] = text
            elif g_text == 'PLA': 
                try: meta['pla'] = float(text)
                except: pass
            elif g_text == 'DeviceId': meta['dev_id'] = text.strip()
            elif g_text == 'LocalImpactId':
                try: meta['id'] = int(text)
                except: pass

        # Validation
        parsed_name = DataUtils.parse_name(meta['name'])
        if not parsed_name: return None
        if meta['pla'] is None or meta['pla'] <= self.pla_threshold: return None
        
        dt_obj = DataUtils.parse_datetime(meta['dt_str'])
        if not dt_obj: return None

        return {
            'dt': dt_obj,
            'name': parsed_name,
            'pla': meta['pla'],
            'dev_id': meta['dev_id'],
            'id': meta['id']
        }

    def _repair_timestamps(self, device_groups):
        print(f"Processing events for {len(device_groups)} devices (repair attempt)...")
        total_repaired = 0
        
        for dev_id, items in device_groups.items():
            offsets = []
            
            # Calculate offsets based on reference data
            if dev_id in self.ref_data:
                ref_events = self.ref_data[dev_id]
                for item in items:
                    lid = item.get('id')
                    if lid in ref_events:
                        offsets.append(ref_events[lid] - item['dt'])

            # Apply Repair
            if offsets:
                # Median strategy
                offsets.sort()
                median_offset = offsets[len(offsets)//2]
                for item in items:
                    if item['dt'].year <= 1980:
                        item['dt'] += median_offset
                        total_repaired += 1
            elif self.session_start_utc is not None:
                # Fallback strategy
                sess_dt = datetime.fromtimestamp(self.session_start_utc, tz=timezone.utc)
                for item in items:
                    if item['dt'].year <= 1980:
                        try:
                            item['dt'] = item['dt'].replace(year=sess_dt.year, month=sess_dt.month, day=sess_dt.day, tzinfo=timezone.utc)
                            total_repaired += 1
                        except: continue
                        
        print(f" -> Repaired {total_repaired} events via median offsets / fallback.")

    def _flatten_data(self, device_groups):
        saes_list = []
        ids_list = []
        
        for dev_id, items in device_groups.items():
            for item in items:
                # Double check PLA threshold (already checked, but safety first)
                if item['pla'] > self.pla_threshold:
                    dt_obj = item['dt']
                    if dt_obj.tzinfo is None:
                        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                    saes_list.append(dt_obj.timestamp())
                    ids_list.append(item['name'])
                    
        self.timestamps = np.array(saes_list)
        self.identifiers = np.array(ids_list)
        print(f"Loaded {len(saes_list)} valid SAE events from XML after repair.")


