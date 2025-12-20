import csv
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional
from Classes.UtilsClass import DataUtils

# [ReferenceDataManager, CollisionDataManager, SensorDataManager REMAIN UNCHANGED]
# I will retain them for completeness.

class ReferenceDataManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = self._load()
    def _load(self) -> Dict[str, Dict[int, datetime]]:
        ref_data = {}
        if not self.filepath or not os.path.exists(self.filepath): return ref_data
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
                        dev_id = row.get('Device SN') or row.get('DeviceId')
                        if not dev_id: continue
                        persisted = row.get('Persisted Event #') or row.get('Persisted Event')
                        if not persisted: continue
                        event_id = int(persisted)
                        dt_str = f"{row.get('Date','')} {row.get('Local Time','')} {row.get('Time Zone From UTC','+00:00')}"
                        dt_obj = DataUtils.parse_datetime(dt_str)
                        if dt_obj and dt_obj.year > 1980:
                            ref_data.setdefault(dev_id.strip(), {})[event_id] = dt_obj
                    except: continue
        except Exception as e: print(f"Error reading Reference CSV: {e}")
        return ref_data

class CollisionDataManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.timestamps = None
        self.identifiers = None
        self._load()
    def _load(self):
        collisions, ids = [], []
        if not os.path.exists(self.filepath): return
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
                    except: continue
        except Exception as e: print(f"Error reading Composer CSV: {e}")
        self.timestamps = np.array(collisions)
        self.identifiers = np.array(ids)

class SensorDataManager:
    def __init__(self, filepath: str, reference_data: Dict, pla_threshold: float):
        self.filepath = filepath
        self.ref_data = reference_data
        self.pla_threshold = pla_threshold
        self.timestamps = None
        self.identifiers = None
        self.session_start_utc = None
        self._load_and_repair()
    def _load_and_repair(self):
        if not os.path.exists(self.filepath): return
        try:
            tree = ET.parse(self.filepath)
            root = tree.getroot()
            self._parse_session_start(root)
            device_groups = self._extract_events(root)
            self._repair_timestamps(device_groups)
            self._flatten_data(device_groups)
        except Exception as e: print(f"Error processing XML: {e}")
    def _parse_session_start(self, root):
        node = root.find('.//SESSION_INFO/start_time')
        if node is not None and node.text:
            dt = DataUtils.parse_datetime(node.text)
            if dt: self.session_start_utc = dt.timestamp()
    def _extract_events(self, root):
        groups = {}
        for instance in root.findall('.//instance'):
            data = self._parse_instance(instance)
            if data: groups.setdefault(data['dev_id'], []).append(data)
        return groups
    def _parse_instance(self, instance):
        meta = {'name': None, 'dt_str': None, 'pla': None, 'dev_id': None, 'id': None}
        for label in instance.findall('label'):
            group = label.find('group')
            text_node = label.find('text')
            text = text_node.text if text_node is not None else None
            if group is None or text is None: continue
            if group.text == 'Athlete Name': meta['name'] = text
            elif group.text == 'ImpactDate': meta['dt_str'] = text
            elif group.text == 'PLA': meta['pla'] = float(text)
            elif group.text == 'DeviceId': meta['dev_id'] = text.strip()
            elif group.text == 'LocalImpactId': meta['id'] = int(text)
        parsed = DataUtils.parse_name(meta['name'])
        if not parsed or meta['pla'] is None or meta['pla'] <= self.pla_threshold: return None
        dt_obj = DataUtils.parse_datetime(meta['dt_str'])
        if not dt_obj: return None
        return {'dt': dt_obj, 'name': parsed, 'pla': meta['pla'], 'dev_id': meta['dev_id'], 'id': meta['id']}
    def _repair_timestamps(self, device_groups):
        print(f"Repairing timestamps for {len(device_groups)} devices...")
        for dev_id, items in device_groups.items():
            offsets = []
            if dev_id in self.ref_data:
                ref_events = self.ref_data[dev_id]
                for item in items:
                    if item['id'] in ref_events: offsets.append(ref_events[item['id']] - item['dt'])
            if offsets:
                offsets.sort()
                median = offsets[len(offsets)//2]
                for item in items:
                    if item['dt'].year <= 1980: item['dt'] += median
            elif self.session_start_utc:
                sess_dt = datetime.fromtimestamp(self.session_start_utc, tz=timezone.utc)
                for item in items:
                    if item['dt'].year <= 1980:
                        try: item['dt'] = item['dt'].replace(year=sess_dt.year, month=sess_dt.month, day=sess_dt.day, tzinfo=timezone.utc)
                        except: continue
    def _flatten_data(self, device_groups):
        saes_list, ids_list = [], []
        for dev_id, items in device_groups.items():
            for item in items:
                if item['pla'] > self.pla_threshold:
                    dt = item['dt']
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                    saes_list.append(dt.timestamp())
                    ids_list.append(item['name'])
        self.timestamps = np.array(saes_list)
        self.identifiers = np.array(ids_list)
        print(f"Loaded {len(saes_list)} valid SAE events.")


# --- UPDATED: Capture ALL players even if status is empty ---

class MatchMetadataManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.schedule_data = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            print(f"Warning: Schedule file not found: {self.filepath}")
            return
     
        try:
            if self.filepath.endswith('.xlsx') or self.filepath.endswith('.xls'):
                df = pd.read_excel(self.filepath)
            else:
                df = pd.read_csv(self.filepath)

            # Normalize headers to lowercase and underscores
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

            # List of columns that ARE NOT player names
            known_cols = {
                'date', 'team', 'type', 'contact', 'opponent', 
                'data_available', 'video_available', 'mg_off_charge', 
                'mg_out', 'kickoff', 'end_first', 'start_second', 'end_second',
                'start_third', 'end_third', 'start_fourth', 'end_fourth', 'contact_drill_a',
                'start_contact_a', 'end_contact_a', 'contact_drill_b',
                'start_contact_b', 'end_contact_b', 'contact_drill_c',
                'start_contact_c', 'end_contact_c'
            }

            for _, row in df.iterrows():
                date_val = row.get('date')
                if pd.isna(date_val): continue
                
                dt = None
                if isinstance(date_val, datetime):
                    dt = date_val
                else:
                    try:
                        dt = datetime.strptime(str(date_val).strip(), '%d/%m/%Y')
                    except ValueError:
                        continue

                key = dt.strftime('%Y-%m-%d')
                
                # Markers
                markers = {
                   'mg_out': row.get('mg_out') if pd.notna(row.get('mg_out')) else None,
                   'kickoff': row.get('kickoff') if pd.notna(row.get('kickoff')) else None,
                   'end_first': row.get('end_first') if pd.notna(row.get('end_first')) else None,
                   'start_second': row.get('start_second') if pd.notna(row.get('start_second')) else None,
                   'end_second': row.get('end_second') if pd.notna(row.get('end_second')) else None
                }

                # --- NEW: Extract ALL Players ---
                player_statuses = {}
                for col in df.columns:
                    # If it's not a marker/metadata column, it's a player
                    if col not in known_cols:
                        # Clean name (e.g., 'levi_a' -> 'Levi A')
                        p_name = col.replace('_', ' ').title()
                        
                        # Get status value
                        val = row.get(col)
                        
                        if pd.notna(val):
                            status_str = str(val).lower().strip()
                            # Clean up common "empty" indicators like dashes if necessary
                            if status_str in ['-', '.', 'nan']:
                                status_str = ""
                            player_statuses[p_name] = status_str
                        else:
                            # IMPORTANT: Register the player even if status is empty
                            player_statuses[p_name] = "" 

                self.schedule_data[key] = {
                    'markers': markers,
                    'players': player_statuses
                }
                
            print(f"Loaded match schedule for {len(self.schedule_data)} dates.")
            
        except Exception as e:
            print(f"Error reading Schedule file: {e}")

    def get_markers(self, target_date: datetime) -> Dict[str, float]:
        key = target_date.strftime('%Y-%m-%d')
        if key not in self.schedule_data: return {}

        data_row = self.schedule_data[key]
        raw_times = data_row['markers']
        
        markers = {}
        base_date_str = target_date.strftime('%Y-%m-%d')

        def parse_time(t_val):
            if not t_val: return None
            
            if isinstance(t_val, (datetime, pd.Timestamp)):
                 full_dt = datetime.combine(target_date.date(), t_val.time())
                 if target_date.tzinfo: full_dt = full_dt.replace(tzinfo=target_date.tzinfo)
                 return full_dt.timestamp()
            
            if hasattr(t_val, 'time'):
                 full_dt = datetime.combine(target_date.date(), t_val)
                 if target_date.tzinfo: full_dt = full_dt.replace(tzinfo=target_date.tzinfo)
                 return full_dt.timestamp()

            t_str = str(t_val).strip().lower().replace('.', ':')
            formats = ['%Y-%m-%d %I:%M%p', '%Y-%m-%d %I:%M:%S%p', '%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M%p']
            
            full_str = f"{base_date_str} {t_str}"
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(full_str, fmt)
                    if target_date.tzinfo: dt = dt.replace(tzinfo=target_date.tzinfo)
                    return dt.timestamp()
                except ValueError: pass
                
                try:
                    dt = datetime.strptime(t_str, fmt)
                    dt = dt.replace(year=target_date.year, month=target_date.month, day=target_date.day)
                    if target_date.tzinfo: dt = dt.replace(tzinfo=target_date.tzinfo)
                    return dt.timestamp()
                except ValueError: continue
            return None

        for event, val in raw_times.items():
            ts = parse_time(val)
            if ts: markers[event] = ts
        
        return markers

    def get_player_statuses(self, target_date: datetime) -> Dict[str, str]:
        key = target_date.strftime('%Y-%m-%d')
        if key not in self.schedule_data: return {}
        return self.schedule_data[key]['players']