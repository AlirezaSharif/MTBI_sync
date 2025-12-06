#!/usr/bin/env python3
"""
plot_sync_efficient_repaired.py

- Vectorised synchronization & plotting (based on Plot_sync_efficient.py)
- Adds 1970/1980 timestamp repair pipeline from Sync_1970.py
- Ignores participant named "Hybrid 4" (option A) by athlete name
- Expects config.yaml with keys:
    CSV_path
    XML_path
    PLA_threshold
    Alignment_threshold
    Reference_Impacts_CSV_path   # optional but recommended for repair
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

# --- CONFIG LOAD ---
file_path = "config.yaml"
if not os.path.exists(file_path):
    print(f"Error: config file not found at {file_path}")
    sys.exit(1)

with open(file_path, 'r') as f:
    config = yaml.safe_load(f)

# --- NAME MAPPING ---
NAME_MAP = {
    "Rueben": "Reuben",
    "Te": "Te Reimana",
    "Hybrid 3": "Jimmy",
    "Wheturangi":"Whetu",
    "Sam" : "Sam G"
}

# Normalise Hybrid4 detection by removing whitespace & lowercasing
def is_hybrid4_name(name: str) -> bool:
    if not name:
        return False
    s = ''.join(name.split()).lower()
    return s == 'hybrid4'or s=='hybrid2'

def parse_xml_name(xml_name):
    if not xml_name:
        return None
    if ',' in xml_name:
        first_name = xml_name.split(',', 1)[1].strip()
    else:
        first_name = xml_name.strip()
    # map names, then final check for Hybrid4
    mapped = NAME_MAP.get(first_name, first_name)
    if is_hybrid4_name(mapped):
        return None   # treat as missing (ignored)
    return mapped

# Parse an XML timestamp into a timezone-aware datetime (returns datetime or None)
def parse_xml_datetime(timestamp_str):
    if not timestamp_str:
        return None
    # Try the high-precision pattern first, then fallback to no fractional seconds
    for fmt in ('%Y-%m-%d %I:%M:%S.%f %p %z', '%Y-%m-%d %I:%M:%S %p %z', '%Y-%m-%d %H:%M:%S %z'):
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            return dt
        except Exception:
            continue
    # Try parsing ISO-like with numeric TZ like +13:00 (already covered above), otherwise None
    return None

# --- REFERENCE CSV LOADER (for repair) ---
def load_reference_impacts(ref_csv_path):
    """
    Loads a reference CSV mapping Device SN -> LocalImpactId -> datetime
    Expect columns similar to Sync_1970: 'Device SN', 'Persisted Event #', 'Date', 'Local Time', 'Time Zone From UTC'
    Returns: dict: ref_data[device_id][local_id] = datetime (aware)
    """
    ref_data = {}
    if not ref_csv_path or not os.path.exists(ref_csv_path):
        print(f"Reference CSV not found or not provided: {ref_csv_path}. Continuing without reference repair.")
        return ref_data

    print(f"Loading reference impacts from: {ref_csv_path}")
    try:
        # try utf-8-sig then fallback
        try:
            f = open(ref_csv_path, mode='r', encoding='utf-8-sig')
            f.read(100); f.seek(0)
        except UnicodeDecodeError:
            f = open(ref_csv_path, mode='r', encoding='utf-16')
        with f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                try:
                    dev_id = row.get('Device SN') or row.get('DeviceId') or row.get('DeviceId ')
                    if not dev_id:
                        continue
                    dev_id = dev_id.strip()
                    persisted = row.get('Persisted Event #') or row.get('Persisted Event')
                    if not persisted:
                        continue
                    event_id = int(persisted)
                    date_part = row.get('Date') or row.get('Event Date') or ''
                    time_part = row.get('Local Time') or row.get('LocalTime') or ''
                    tz_part = row.get('Time Zone From UTC') or row.get('Timezone') or '+00:00'
                    # normalize tz like +13:00 -> +1300 for strptime if necessary
                    tz_norm = tz_part.replace(':', '') if ':' in tz_part else tz_part
                    dt_str = f"{date_part} {time_part} {tz_norm}"
                    # Try parse like '2023-03-24 12:45:03 +1300'
                    try:
                        dt_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S %z')
                    except Exception:
                        # Try fallback patterns
                        try:
                            dt_obj = datetime.strptime(dt_str, '%d/%m/%Y %H:%M:%S %z')
                        except Exception:
                            continue
                    if dt_obj.year > 1980:
                        ref_data.setdefault(dev_id, {})[event_id] = dt_obj
                        count += 1
                except Exception:
                    continue
        print(f"Loaded {count} valid reference events.")
    except Exception as e:
        print(f"Error reading Reference CSV: {e}")
    return ref_data

# --- COLLISIONS LOADER (Composer CSV) ---
def load_collisions(csv_path):
    collisions = []
    collisions_identifier = []
    if not os.path.exists(csv_path):
        print(f"Error: Composer CSV not found at {csv_path}")
        return None, None

    try:
        # determine encoding
        try:
            f = open(csv_path, mode='r', encoding='utf-8-sig')
            f.read(100); f.seek(0)
        except UnicodeDecodeError:
            f = open(csv_path, mode='r', encoding='utf-16')
        with f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    playback_time = float(row.get('Qualifier: Time', row.get('Qualifier Time', 0)))
                    player_name = (row.get('Topic Name') or '').strip()
                    # Parse and map name; ignore Hybrid4
                    parsed = parse_xml_name(player_name)
                    if not parsed:
                        continue
                    collisions.append(playback_time)
                    collisions_identifier.append(parsed)
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading Composer CSV: {e}")
        return None, None

    print(f"Loaded {len(collisions)} collision events from CSV.")
    return np.array(collisions), np.array(collisions_identifier)

# --- SAE LOADER with REPAIR ---
def load_saes_with_repair(xml_path, ref_data, pla_threshold=14.0):
    """
    Loads SAE events from XML, groups by device, tries to repair 1970/1980 timestamps
    using ref_data (if available), or falling back to session start date.
    Returns: saes (np.array of epoch seconds), sae_identifiers (np.array names), session_start_utc (epoch seconds)
    """
    saes = []
    sae_identifier = []
    session_start_utc = None
    device_groups = {}  # device_id -> list of event dicts

    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return None, None, None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # session start
        session_node = root.find('.//SESSION_INFO/start_time')
        if session_node is not None and session_node.text:
            sess_dt = parse_xml_datetime(session_node.text)
            if sess_dt:
                session_start_utc = sess_dt.timestamp()
                print(f"Found session start time: {sess_dt.isoformat()}")
            else:
                print("Warning: Could not parse session start time.")

        # iterate instances and collect device grouped items
        for instance in root.findall('.//instance'):
            player_name = None
            impact_date_str = None
            pla_value = None
            device_id = None
            local_id = None

            for label in instance.findall('label'):
                group = label.find('group')
                text_node = label.find('text')
                text = text_node.text if text_node is not None else None
                if group is None or text is None:
                    continue
                g = group.text
                if g == 'Athlete Name':
                    player_name = text
                elif g == 'ImpactDate':
                    impact_date_str = text
                elif g == 'PLA':
                    try:
                        pla_value = float(text)
                    except Exception:
                        pla_value = None
                elif g == 'DeviceId':
                    device_id = text.strip()
                elif g == 'LocalImpactId':
                    try:
                        local_id = int(text)
                    except Exception:
                        local_id = None

            # Map name and ignore Hybrid 4
            parsed_name = parse_xml_name(player_name)
            if not parsed_name:
                continue

            # skip low PLA early
            if pla_value is None or pla_value <= pla_threshold:
                continue

            # parse dt as datetime (may be incorrect year)
            dt_obj = parse_xml_datetime(impact_date_str)
            if not dt_obj:
                # we will skip entries with no parseable date
                continue

            # keep event
            item = {
                'dt': dt_obj,
                'name': parsed_name,
                'pla': pla_value,
                'dev_id': device_id,
                'id': local_id
            }
            device_groups.setdefault(device_id, []).append(item)

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error reading XML: {e}")
        return None, None, None

    # --- REPAIR LOGIC per device ---
    print(f"Processing events for {len(device_groups)} devices (repair attempt)...")
    total_repaired = 0

    for dev_id, items in device_groups.items():
        offsets = []

        # If we have reference events for this device, compute offsets
        if dev_id in ref_data:
            ref_events = ref_data[dev_id]
            for item in items:
                local_id = item.get('id')
                if local_id in ref_events:
                    # subtract xml dt from reference dt to get offset to add to xml
                    offsets.append(ref_events[local_id] - item['dt'])

        # If offsets found, use median (robust)
        if offsets:
            # compute median offset in timedelta form
            offsets_sorted = sorted(offsets)
            median_offset = offsets_sorted[len(offsets_sorted)//2]
            for item in items:
                if item['dt'].year <= 1980:
                    item['dt'] = item['dt'] + median_offset
                    total_repaired += 1
        else:
            # fallback: if session start exists, copy session date into those bad-year entries
            if session_start_utc is not None:
                sess_dt = datetime.fromtimestamp(session_start_utc, tz=timezone.utc)
                for item in items:
                    if item['dt'].year <= 1980:
                        try:
                            item['dt'] = item['dt'].replace(year=sess_dt.year, month=sess_dt.month, day=sess_dt.day,
                                                            tzinfo=timezone.utc)
                            total_repaired += 1
                        except Exception:
                            continue

    print(f" -> Repaired {total_repaired} events via median offsets / fallback.")

    # --- COLLATE final SAEs (apply PLA threshold again) ---
    for dev_id, items in device_groups.items():
        for item in items:
            if item['pla'] is not None and item['pla'] > pla_threshold:
                # ensure tz-aware; if not, make it UTC
                dt_obj = item['dt']
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                saes.append(dt_obj.timestamp())
                sae_identifier.append(item['name'])

    print(f"Loaded {len(saes)} valid SAE events (PLA > {pla_threshold}) from XML after repair.")
    return np.array(saes), np.array(sae_identifier), session_start_utc

# --- SYNCHRONISATION function (vectorised) ---
def synchronise(collisions, saes, collisions_identifier, sae_identifier, session_start_time):
    if not all(arr is not None and arr.size > 0 for arr in [collisions, saes, collisions_identifier, sae_identifier]):
        print("Error: Input arrays are empty or missing. Cannot synchronize.")
        return None, 0, None, None

    if session_start_time is None:
        print("Error: No session start time provided. Cannot determine search range.")
        return None, 0, None, None

    alignment_window_sec = config.get('Alignment_threshold', 5.0)
    intervals_sec = 1
    search_radius_hours = 24
    search_radius_sec = search_radius_hours * 3600

    start_point_sec = session_start_time - search_radius_sec
    end_point_sec = session_start_time + search_radius_sec

    potential_sync_points = np.arange(start_point_sec, end_point_sec + intervals_sec, intervals_sec)
    alignment_prctage = np.zeros_like(potential_sync_points, dtype=float)

    print(f"Testing {len(potential_sync_points)} potential sync points... (vectorised)")

    # Prepare for broadcasting
    sae_id_col = sae_identifier.reshape(-1, 1)
    collisions_id_row = collisions_identifier.reshape(1, -1)
    id_match_matrix = (sae_id_col == collisions_id_row)

    # iterate potential sync points
    for i, sync_point in enumerate(potential_sync_points):
        sae_playback = saes - sync_point
        # Vectorised differences
        sae_playback_col = sae_playback.reshape(-1, 1)
        time_diff_matrix = np.abs(sae_playback_col - collisions)
        time_match_matrix = (time_diff_matrix < alignment_window_sec)
        combined_match_matrix = id_match_matrix & time_match_matrix
        aligned_sae_mask = np.any(combined_match_matrix, axis=1)
        aligned_saes_count = np.sum(aligned_sae_mask)
        if len(sae_playback) > 0:
            alignment_prctage[i] = aligned_saes_count / len(sae_playback)

    max_alignment_prctage = np.max(alignment_prctage)
    best_index = np.argmax(alignment_prctage)
    predicted_syncpoint = potential_sync_points[best_index]
    return predicted_syncpoint, max_alignment_prctage, potential_sync_points, alignment_prctage

def get_plot_data(collisions_np, saes_np, collisions_id_np, saes_id_np, sync_point_seconds, alignment_window_sec=5.0):
    all_players = sorted(list(set(collisions_id_np) | set(saes_id_np)))
    plot_data = {player: {'coded': [], 'aligned': [], 'unaligned': []} for player in all_players}
    
    # Track unique indices of coded events that have been matched
    unique_matches = {player: set() for player in all_players}

    for i in range(len(collisions_np)):
        player = collisions_id_np[i]
        playback_time = collisions_np[i]
        if player in plot_data:
            plot_data[player]['coded'].append(playback_time)

    sae_playback_times = saes_np - sync_point_seconds
    sae_aligned_count = 0
    sae_total_count = len(sae_playback_times)

    for i in range(sae_total_count):
        current_sae_time = sae_playback_times[i]
        current_sae_id = saes_id_np[i]
        if current_sae_id not in plot_data:
            continue
        id_check = (collisions_id_np == current_sae_id) & \
                   (np.abs(collisions_np - current_sae_time) < alignment_window_sec)
        if np.any(id_check):
            plot_data[current_sae_id]['aligned'].append(current_sae_time)
            sae_aligned_count += 1
            # Add matched coded indices to the set
            matched_indices = np.where(id_check)[0]
            unique_matches[current_sae_id].update(matched_indices)
        else:
            plot_data[current_sae_id]['unaligned'].append(current_sae_time)

    return plot_data, sae_aligned_count, sae_total_count, all_players, unique_matches




def plot_alignment(plot_data, players, sync_point_utc_dt, max_alignment, 
                   sae_aligned_count, sae_total_count, 
                   alignment_scores, potential_sync_points, unique_matches):
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.1) # Increased height ratio for bottom plot
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # --- TOP PLOT: Synchronization Score ---
    sync_point_datetimes = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in potential_sync_points]
    ax_top.plot(sync_point_datetimes, alignment_scores * 100, color='black', linewidth=0.5)
    ax_top.axvline(sync_point_utc_dt, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.7)

    textstr = (f"Predicted synchronisation point\n"
               f"{sync_point_utc_dt.strftime('%Z %H:%M:%S %p')}\n"
               f"{max_alignment * 100:.2f}% Alignment")
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
    ax_top.text(sync_point_utc_dt, np.max(alignment_scores) * 100, textstr,
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                bbox=props, transform=ax_top.get_xaxis_transform())

    ax_top.set_ylabel("Alignment (%)")
    ax_top.set_ylim(bottom=0)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis='x', length=0)

    # --- TOP PLOT: Legend & Title ---
    coded_proxy = plt.Line2D([0], [0], marker='v', color='blue', markersize=8, linestyle='None')
    sae_proxy   = plt.Line2D([0], [0], marker='^', color='red',  markersize=8, linestyle='None')
    aligned_proxy = plt.Line2D([0], [0], marker='^', color='lime', markersize=8, linestyle='None')

    legend_handles = [coded_proxy, sae_proxy, aligned_proxy]
    legend_labels  = ['Coded Match Event', 'SAE', 'Aligned SAE']

    ax_top.legend(
        legend_handles,
        legend_labels,
        loc='upper left',
        ncol=3,
        frameon=False,
        fontsize=10
    )

    # Note: Ensure 'config' is available in scope or passed as argument. 
    # Using defaults here to prevent NameError if config is missing.
    summary_text = f"SAEs aligned: {sae_aligned_count}/{sae_total_count} ({(sae_aligned_count/sae_total_count*100):.0f}%), aligment threshold: ±{config.get('Alignment_threshold', 5.0)}s, impact threhold: {config.get('PLA_threshold', 14.0)}g"
#     
    ax_top.set_title(summary_text, loc='right', fontsize=12, fontweight='bold')

    # --- BOTTOM PLOT: Timeline with Offsets ---
    
    # Define offsets to separate the lines
    offset_coded = -0.2
    offset_aligned = 0.0
    offset_unaligned = 0.2

    for i, player in enumerate(players):
        y_center = i
        data = plot_data[player]
        
        # 1. Plot Coded Events (Top sub-line)
        ax_bottom.plot(data['coded'], [y_center + offset_coded] * len(data['coded']),
                       marker='v', color='blue', markersize=6, linestyle='None', alpha=0.8)
        
        # 2. Plot Aligned SAEs (Middle sub-line)
        ax_bottom.plot(data['aligned'], [y_center + offset_aligned] * len(data['aligned']),
                       marker='^', color='lime', markersize=6, linestyle='None', alpha=0.9)
        
        # 3. Plot Unaligned SAEs (Bottom sub-line)
        ax_bottom.plot(data['unaligned'], [y_center + offset_unaligned] * len(data['unaligned']),
                       marker='^', color='red', markersize=6, linestyle='None', alpha=0.6)
        ax_bottom.plot(data['aligned'], [y_center + offset_unaligned] * len(data['aligned']),
                       marker='^', color='red', markersize=6, linestyle='None', alpha=0.6)

        # Draw a separator line between players (except after the last one)
        if i < len(players) - 1:
            ax_bottom.axhline(y=i + 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        

    # Formatting
    ax_bottom.set_yticks(np.arange(len(players)))
    ax_bottom.set_yticklabels(players)
    
    # Invert Y axis so first player is at top. 
    # Padding added (-0.5 to len + 0.5) to accommodate the offsets
    ax_bottom.set_ylim(len(players) - 0.5, -0.5)
    
    ax_bottom.set_xlabel("Playback Timestamp (hh:mm:ss)", fontsize=12)
    def format_playback_time(seconds, _):
        sign = '-' if seconds < 0 else ''
        seconds = abs(int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{sign}{h:02d}:{m:02d}:{s:02d}"

    ax_bottom.xaxis.set_major_formatter(mticker.FuncFormatter(format_playback_time))
    all_times = [t for p in players for k in plot_data[p] for t in plot_data[p][k]]
    if all_times:
        max_time = max(all_times)
        ax_bottom.set_xlim(left=max(0, min(all_times) * 2.01), right=max_time * 1.01)
 

    # Secondary stats axis
    y_levels = np.arange(len(players))
    ax_right = ax_bottom.twinx()
    ax_right.set_yticks(y_levels)
    stats_labels = []
    for player in players:
        num_aligned = len(unique_matches[player])
        num_coded = len(plot_data[player]['coded'])
        stats_labels.append(f"{num_aligned}/{num_coded}")
    ax_right.set_yticklabels(stats_labels)
    ax_right.set_ylim(ax_bottom.get_ylim())
    ax_right.set_ylabel("Unique Coded Events Matched / Total Coded", fontsize=10)
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()
    
    plt.tight_layout()
    plt.show()
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    CSV_FILE = config.get('CSV_path')
    XML_FILE = config.get('XML_path')
    PLA_THRESHOLD = config.get('PLA_threshold', 14.0)
    ALIGNMENT_THRESHOLD = config.get('Alignment_threshold', 5.0)
    REF_CSV = config.get('Reference_Impacts_CSV_path', None)

    print("--- Loading Reference Impacts (if provided) ---")
    ref_map = load_reference_impacts(REF_CSV)
    # ref_map = load_reference_impacts('Matai - impacts - 2023 0201-0901.csv')

    print("--- Loading Collisions ---")
    collisions_np, collisions_id_np = load_collisions(CSV_FILE)

    print("--- Loading SAEs and repairing (if needed) ---")
    saes_np, saes_id_np, session_start = load_saes_with_repair(XML_FILE, ref_map, pla_threshold=PLA_THRESHOLD)

    if collisions_np is not None and saes_np is not None and session_start is not None:
        print("\n--- Starting Synchronization ---")
        # Put alignment window into config temporarily for synchronise
        config['Alignment_threshold'] = ALIGNMENT_THRESHOLD

        sync_point_seconds, max_alignment, potential_sync_points, alignment_prctage = synchronise(
            collisions_np,
            saes_np,
            collisions_id_np,
            saes_id_np,
            session_start
        )

        print("\n--- Synchronization Complete ---")

        if sync_point_seconds is not None:
            predicted_utc_datetime = datetime.fromtimestamp(sync_point_seconds, tz=timezone.utc)
            print(f"Predicted Sync Point (UTC Seconds): {sync_point_seconds}")
            print(f"Predicted Sync Point (Human-Readable): {predicted_utc_datetime.isoformat()}")
            print(f"Max Alignment Percentage: {max_alignment * 100:.2f}%")

            print("\n--- Generating Plot Data ---")
            plot_data, sae_aligned_count, sae_total_count, player_list, unique_matches = get_plot_data(
                collisions_np,
                saes_np,
                collisions_id_np,
                saes_id_np,
                sync_point_seconds,
                alignment_window_sec=ALIGNMENT_THRESHOLD
            )

            print(f"Total SAEs for plotting: {sae_total_count}")
            print(f"Aligned SAEs for plotting: {sae_aligned_count}")

            print("--- Displaying Plot ---")
            plot_alignment(
                plot_data,
                player_list,
                predicted_utc_datetime,
                max_alignment,
                sae_aligned_count,
                sae_total_count,
                alignment_prctage,
                potential_sync_points,
                unique_matches
            )
        else:
            print("Synchronization failed, possibly due to empty input data.")
    else:
        print("\nSynchronization aborted due to errors loading data.")
