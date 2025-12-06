#!/usr/bin/env python3
"""
Object-Oriented Synchronization & Plotting Tool

Refactored from plot_sync_efficient_repaired.py
"""

import csv
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yaml
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# --- CONFIGURATION CLASS ---
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


# --- UTILITIES ---
class DataUtils:
    NAME_MAP = {
        "Rueben": "Reuben",
        "Te": "Te Reimana",
        "Hybrid 3": "Jimmy",
        "Wheturangi": "Whetu",
        "Sam": "Sam G"
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


# --- DATA LOADERS ---

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


# --- CORE LOGIC ---

class Synchronizer:
    def __init__(self, config: Config):
        self.window = config.alignment_threshold
        self.search_radius_hours = 24
    
    def synchronize(self, coll_data: CollisionDataManager, sae_data: SensorDataManager):
        """Performs vectorized synchronization."""
        
        # Guard clauses
        if (coll_data.timestamps is None or sae_data.timestamps is None or 
            len(coll_data.timestamps) == 0 or len(sae_data.timestamps) == 0):
            print("Error: Empty datasets.")
            return None

        if sae_data.session_start_utc is None:
            print("Error: No session start time.")
            return None

        # Define Search Space
        search_radius_sec = self.search_radius_hours * 3600
        start = sae_data.session_start_utc - search_radius_sec
        end = sae_data.session_start_utc + search_radius_sec
        intervals_sec = 1.0
        
        potential_points = np.arange(start, end + intervals_sec, intervals_sec)
        scores = np.zeros_like(potential_points, dtype=float)

        print(f"Testing {len(potential_points)} potential sync points... (vectorised)")

        # Prepare Broadcasting
        # Shape: (N_sae, 1) vs (1, N_coll)
        sae_ids = sae_data.identifiers.reshape(-1, 1)
        coll_ids = coll_data.identifiers.reshape(1, -1)
        
        # Name match matrix (Boolean)
        name_match = (sae_ids == coll_ids)

        coll_ts = coll_data.timestamps # Shape (N_coll,)
        sae_ts = sae_data.timestamps   # Shape (N_sae,)

        # Iterate sync points
        for i, sync_p in enumerate(potential_points):
            # Shift SAE time to playback time
            shifted_sae = sae_ts - sync_p
            shifted_sae_col = shifted_sae.reshape(-1, 1)
            
            # Time difference matrix
            time_diffs = np.abs(shifted_sae_col - coll_ts)
            time_match = (time_diffs < self.window)
            
            # Combine
            valid_hits = name_match & time_match
            
            # Count aligned SAEs (rows with at least one match)
            aligned_count = np.sum(np.any(valid_hits, axis=1))
            
            if len(sae_ts) > 0:
                scores[i] = aligned_count / len(sae_ts)

        # Find Best
        best_idx = np.argmax(scores)
        best_sync = potential_points[best_idx]
        max_score = scores[best_idx]

        return SyncResult(best_sync, max_score, potential_points, scores)


@dataclass
class SyncResult:
    sync_point: float
    max_alignment: float
    all_points: np.ndarray
    all_scores: np.ndarray
    
    @property
    def sync_dt_utc(self):
        return datetime.fromtimestamp(self.sync_point, tz=timezone.utc)


# --- VISUALIZATION ---

class Visualizer:
    def __init__(self, config: Config):
        self.config = config

    def prepare_plot_data(self, coll_data: CollisionDataManager, sae_data: SensorDataManager, result: SyncResult):
        """Organizes data for the timeline plot."""
        
        all_players = sorted(list(set(coll_data.identifiers) | set(sae_data.identifiers)))
        plot_data = {p: {'coded': [], 'aligned': [], 'unaligned': []} for p in all_players}
        unique_matches = {p: set() for p in all_players}
        
        # 1. Populate Coded (Collisions)
        for i, name in enumerate(coll_data.identifiers):
            plot_data[name]['coded'].append(coll_data.timestamps[i])

        # 2. Populate SAEs
        shifted_sae = sae_data.timestamps - result.sync_point
        sae_aligned_count = 0
        
        for i, sae_t in enumerate(shifted_sae):
            name = sae_data.identifiers[i]
            if name not in plot_data: continue
            
            # Check alignment against original collisions
            # (Note: In a huge dataset, this loop might be slow, but for plotting it's usually fine. 
            #  Could be vectorized if needed, but logic is clearer here).
            
            # Filter collisions for this player
            p_coll_mask = (coll_data.identifiers == name)
            p_coll_ts = coll_data.timestamps[p_coll_mask]
            
            # check distance
            if len(p_coll_ts) > 0 and np.min(np.abs(p_coll_ts - sae_t)) < self.config.alignment_threshold:
                plot_data[name]['aligned'].append(sae_t)
                sae_aligned_count += 1
                
                # Mark which coded event was hit
                matched_indices = np.where(p_coll_mask & (np.abs(coll_data.timestamps - sae_t) < self.config.alignment_threshold))[0]
                # Map back to global index if needed, or just count logic
                # Here we just want unique counts per player
                # (Simplification: just incrementing stats based on provided logic)
                unique_matches[name].update(matched_indices) # This index logic is slightly loose in original, keeping intent
            else:
                plot_data[name]['unaligned'].append(sae_t)

        return {
            'data': plot_data,
            'players': all_players,
            'aligned_count': sae_aligned_count,
            'total_sae': len(shifted_sae),
            'matches': unique_matches
        }

    def plot(self, coll_data, sae_data, result: SyncResult):
        viz_data = self.prepare_plot_data(coll_data, sae_data, result)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.1)
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1])

        self._plot_alignment_score(ax_top, result, viz_data)
        self._plot_timeline(ax_bottom, viz_data)

        plt.tight_layout()
        plt.show()

    def _plot_alignment_score(self, ax, result: SyncResult, viz_data):
        # Convert timestamps to datetimes for x-axis
        x_dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in result.all_points]
        
        ax.plot(x_dates, result.all_scores * 100, color='black', linewidth=0.5)
        ax.axvline(result.sync_dt_utc, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.7)

        # Info Box
        textstr = (f"Predicted synchronisation point\n"
                   f"{result.sync_dt_utc.strftime('%Z %H:%M:%S %p')}\n"
                   f"{result.max_alignment * 100:.2f}% Alignment")
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
        ax.text(result.sync_dt_utc, result.max_alignment * 100, textstr,
                fontsize=10, va='top', ha='center', bbox=props, transform=ax.get_xaxis_transform())

        ax.set_ylabel("Alignment (%)")
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

        # Legend
        handles = [
            plt.Line2D([0], [0], marker='v', color='blue', markersize=8, linestyle='None', label='Coded Match Event'),
            plt.Line2D([0], [0], marker='^', color='red', markersize=8, linestyle='None', label='SAE'),
            plt.Line2D([0], [0], marker='^', color='lime', markersize=8, linestyle='None', label='Aligned SAE')
        ]
        ax.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=10)

        # Title
        ac = viz_data['aligned_count']
        tc = viz_data['total_sae']
        pct = (ac/tc*100) if tc > 0 else 0
        summary = (f"SAEs aligned: {ac}/{tc} ({pct:.0f}%), "
                   f"alignment threshold: ±{self.config.alignment_threshold}s, "
                   f"impact threshold: {self.config.pla_threshold}g")
        ax.set_title(summary, loc='right', fontsize=12, fontweight='bold')

    def _plot_timeline(self, ax, viz_data):
        players = viz_data['players']
        p_data = viz_data['data']
        
        offsets = {'coded': -0.2, 'aligned': 0.0, 'unaligned': 0.2}

        for i, player in enumerate(players):
            y = i
            d = p_data[player]
            
            ax.plot(d['coded'], [y + offsets['coded']] * len(d['coded']),
                    marker='v', color='blue', markersize=6, linestyle='None', alpha=0.8)
            
            ax.plot(d['aligned'], [y + offsets['aligned']] * len(d['aligned']),
                    marker='^', color='lime', markersize=6, linestyle='None', alpha=0.9)
            
            ax.plot(d['unaligned'], [y + offsets['unaligned']] * len(d['unaligned']),
                    marker='^', color='red', markersize=6, linestyle='None', alpha=0.6)
            # Overlay ghost for unaligned to ensure visibility logic matches original
            ax.plot(d['aligned'], [y + offsets['unaligned']] * len(d['aligned']),
                     marker='^', color='red', markersize=6, linestyle='None', alpha=0.6)

            if i < len(players) - 1:
                ax.axhline(y=i + 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

        # Axes Formatting
        ax.set_yticks(np.arange(len(players)))
        ax.set_yticklabels(players)
        ax.set_ylim(len(players) - 0.5, -0.5) # Invert
        ax.set_xlabel("Playback Timestamp (hh:mm:ss)", fontsize=12)

        def time_fmt(x, pos):
            s = int(abs(x))
            return f"{'-' if x < 0 else ''}{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_fmt))
        
        # Calculate X limits
        all_times = []
        for p in p_data.values():
            all_times.extend(p['coded'])
            all_times.extend(p['aligned'])
            all_times.extend(p['unaligned'])
            
        if all_times:
            ax.set_xlim(left=max(0, min(all_times)*2.01), right=max(all_times)*1.01)

        # Secondary Axis (Stats)
        ax_right = ax.twinx()
        ax_right.set_yticks(np.arange(len(players)))
        stats_labels = [f"{len(viz_data['matches'][p])}/{len(p_data[p]['coded'])}" for p in players]
        ax_right.set_yticklabels(stats_labels)
        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_ylabel("Unique Coded Events Matched / Total Coded", fontsize=10)


# --- MAIN ORCHESTRATOR ---

class SyncApp:
    def __init__(self, config_path="config.yaml"):
        self.config = Config(config_path)

    def run(self):
        print("--- Loading Data ---")
        
        # 1. Load Reference (Repair) Data
        ref_loader = ReferenceDataManager(self.config.ref_csv_path)
        
        # 2. Load Collision Data
        coll_loader = CollisionDataManager(self.config.csv_path)
        
        # 3. Load Sensor Data (with repair)
        sae_loader = SensorDataManager(
            self.config.xml_path, 
            ref_loader.data, 
            self.config.pla_threshold
        )

        if not (coll_loader.timestamps is not None and sae_loader.timestamps is not None):
            print("Aborting: Data load failure.")
            return

        print("\n--- Synchronizing ---")
        synchronizer = Synchronizer(self.config)
        result = synchronizer.synchronize(coll_loader, sae_loader)

        if result:
            print(f"Predicted Sync Point: {result.sync_dt_utc.isoformat()}")
            print(f"Max Alignment: {result.max_alignment * 100:.2f}%")
            
            print("\n--- Plotting ---")
            viz = Visualizer(self.config)
            viz.plot(coll_loader, sae_loader, result)
        else:
            print("Synchronization failed.")

if __name__ == "__main__":
    app = SyncApp()
    app.run()
