import csv
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yaml

# --- CONFIGURATION ---
config = {
    'XML_path': "gbhs-firstXV-silverstream-2023-filtered_5.0.xml",
    'Composer_CSV_path': "2023_gbhs1stxv_vs_silverstream_composer CSV.csv",
    'Reference_Impacts_CSV_path': "Matai - impacts - 2023 0201-0901.csv", 
    'PLA_threshold': 14.0,
    'Alignment_threshold': 5.0
}

# --- NAME MAPPING ---
NAME_MAP = {
    "Rueben": "Reuben",
    "Te": "Te Reimana",
    "Hybrid 3": "Jimmy" 
}

def parse_xml_name(xml_name):
    if ',' in xml_name:
        first_name = xml_name.split(',', 1)[1].strip()
    else:
        first_name = xml_name.strip()
    return NAME_MAP.get(first_name, first_name)

def parse_utc_timestamp(timestamp_str):
    try:
        return datetime.strptime(timestamp_str, '%Y-%m-%d %I:%M:%S.%f %p %z')
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%d %I:%M:%S %p %z')
        except ValueError:
            return None

def load_reference_impacts(csv_path):
    """Loads Reference CSV to build offset map."""
    ref_data = {}
    print(f"Loading reference data from {csv_path}...")
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                try:
                    dev_id = row['Device SN'].strip()
                    event_id = int(row['Persisted Event #'])
                    dt_str = f"{row['Date']} {row['Local Time']} {row.get('Time Zone From UTC', '+00:00').replace(':', '')}"
                    dt_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S %z')
                    
                    if dt_obj.year > 1980:
                        if dev_id not in ref_data: ref_data[dev_id] = {}
                        ref_data[dev_id][event_id] = dt_obj
                        count += 1
                except: continue
            print(f"Loaded {count} valid reference events.")
            return ref_data
    except Exception as e:
        print(f"Error reading Reference CSV: {e}")
        return {}

def load_collisions(csv_path):
    collisions = []
    collisions_identifier = []
    try:
        try:
            f = open(csv_path, mode='r', encoding='utf-8')
            f.read(100); f.seek(0)
        except UnicodeDecodeError:
            f = open(csv_path, mode='r', encoding='utf-16')

        with f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    playback_time = float(row['Qualifier: Time'])
                    player_name = row['Topic Name'].strip()
                    if player_name:
                        collisions.append(playback_time)
                        collisions_identifier.append(player_name)
                except: continue
    except Exception as e:
        print(f"Error reading Composer CSV: {e}")
        return None, None
    return np.array(collisions), np.array(collisions_identifier)

def load_saes(xml_path, ref_data, pla_threshold=14.0): 
    saes = []
    sae_identifier = []
    session_start_utc = None
    device_groups = {}
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        session_start_node = root.find('.//SESSION_INFO/start_time')
        if session_start_node is not None:
            dt_obj = parse_utc_timestamp(session_start_node.text)
            if dt_obj: session_start_utc = dt_obj.timestamp()

        for instance in root.findall('.//instance'):
            player_name = None
            impact_date_str = None
            pla_value = None 
            device_id = None
            local_id = None
            
            for label in instance.findall('label'):
                group = label.find('group')
                if group is not None:
                    txt = label.find('text').text
                    if group.text == 'Athlete Name': player_name = txt
                    elif group.text == 'ImpactDate': impact_date_str = txt
                    elif group.text == 'PLA': 
                        try: pla_value = float(txt)
                        except: pass
                    elif group.text == 'DeviceId': device_id = txt.strip()
                    elif group.text == 'LocalImpactId': 
                        try: local_id = int(txt)
                        except: pass

            if player_name and impact_date_str and pla_value is not None:
                dt_object = parse_utc_timestamp(impact_date_str)
                if dt_object and device_id:
                    item = {
                        'dt': dt_object,
                        'name': parse_xml_name(player_name),
                        'pla': pla_value,
                        'dev_id': device_id,
                        'id': local_id
                    }
                    if device_id not in device_groups: device_groups[device_id] = []
                    device_groups[device_id].append(item)
        
        # --- REPAIR LOGIC ---
        print(f"Processing events for {len(device_groups)} devices...")
        total_repaired = 0
        
        for dev_id, items in device_groups.items():
            offsets = []
            if dev_id in ref_data:
                ref_events = ref_data[dev_id]
                for item in items:
                    if item['id'] in ref_events:
                        offsets.append(ref_events[item['id']] - item['dt'])
            
            if offsets:
                avg_offset = sorted(offsets)[len(offsets)//2]
                for item in items:
                    if item['dt'].year <= 1980:
                        item['dt'] = item['dt'] + avg_offset
                        total_repaired += 1
            elif session_start_utc:
                sess_dt = datetime.fromtimestamp(session_start_utc, tz=timezone.utc)
                for item in items:
                    if item['dt'].year <= 1980:
                        try:
                            item['dt'] = item['dt'].replace(year=sess_dt.year, month=sess_dt.month, day=sess_dt.day)
                            total_repaired += 1
                        except: pass

        print(f" -> Repaired {total_repaired} events.")

        for dev_id, items in device_groups.items():
            for item in items:
                if item['pla'] > pla_threshold:
                    saes.append(item['dt'].timestamp())
                    sae_identifier.append(item['name'])
                    
    except Exception as e:
        print(f"Error processing XML: {e}")
        return None, None, None

    return np.array(saes), np.array(sae_identifier), session_start_utc

def synchronise(collisions, saes, collisions_identifier, sae_identifier, session_start_time):
    """
    Vectorized sync that returns data compatible with the original plotter.
    """
    print("Calculating synchronization...")
    deltas = []
    unique_sae_players = np.unique(sae_identifier)
    
    for player in unique_sae_players:
        s_times = saes[sae_identifier == player]
        c_times = collisions[collisions_identifier == player]
        if len(c_times) == 0: continue
        diff_matrix = s_times[:, None] - c_times[None, :]
        deltas.extend(diff_matrix.flatten())
        
    deltas = np.array(deltas)
    
    # Restrict search to +/- 12 hours from session start
    search_radius = 24 * 3600
    valid_deltas = deltas[np.abs(deltas - session_start_time) < search_radius]
    
    if len(valid_deltas) == 0:
        print("No matches found within 12 hours.")
        return None, 0, None, None
        
    # Histogram for alignment scores
    bins = np.arange(np.min(valid_deltas), np.max(valid_deltas) + 2, 1)
    counts, bin_edges = np.histogram(valid_deltas, bins=bins)
    
    best_idx = np.argmax(counts)
    best_sync_point = bin_edges[best_idx]
    
    # Calculate exact score for best point
    window = config['Alignment_threshold']
    sae_playback = saes - best_sync_point
    match_count = 0
    for i in range(len(saes)):
        t = sae_playback[i]
        p = sae_identifier[i]
        c_times = collisions[collisions_identifier == p]
        if len(c_times) > 0 and np.min(np.abs(c_times - t)) < window:
             match_count += 1
                 
    score = match_count / len(saes)
    
    # Return data arrays for plotting (similar to original output)
    # alignment_scores = counts normalized
    # potential_sync_points = bin_edges
    return best_sync_point, score, counts / len(saes), bin_edges[:-1]

def get_plot_data(collisions, saes, col_ids, sae_ids, sync_point, window=5.0):
    """Identical data prep to original code."""
    all_players = sorted(list(set(col_ids) | set(sae_ids)))
    plot_data = {p: {'coded': [], 'aligned': [], 'unaligned': []} for p in all_players}
    
    for t, p in zip(collisions, col_ids):
        plot_data[p]['coded'].append(t)
        
    sae_pb = saes - sync_point
    aligned_count = 0
    
    for t, p in zip(sae_pb, sae_ids):
        if p not in plot_data: continue
        c_times = collisions[col_ids == p]
        is_aligned = False
        if len(c_times) > 0 and np.min(np.abs(c_times - t)) < window:
             is_aligned = True
        
        if is_aligned:
            plot_data[p]['aligned'].append(t)
            aligned_count += 1
        else:
            plot_data[p]['unaligned'].append(t)
            
    return plot_data, aligned_count, len(saes), all_players

def plot_alignment(plot_data, players, sync_point_utc_dt, max_alignment, 
                   sae_aligned_count, sae_total_count, 
                   alignment_scores, potential_sync_points):
    """
    Original Plotting Function (Restored)
    """
    fig = plt.figure(figsize=(20, 12)) 
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # --- 1. Top Plot (Alignment Score) ---
    sync_point_datetimes = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in potential_sync_points]
    
    # Handle sparse data for plotting (downsample if too large)
    if len(sync_point_datetimes) > 10000:
        indices = np.linspace(0, len(sync_point_datetimes)-1, 10000, dtype=int)
        plot_dates = [sync_point_datetimes[i] for i in indices]
        plot_scores = alignment_scores[indices]
    else:
        plot_dates = sync_point_datetimes
        plot_scores = alignment_scores

    ax_top.plot(plot_dates, plot_scores * 100, color='black', linewidth=0.5)
    
    ax_top.axvline(sync_point_utc_dt, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.7)
    
    textstr = (f"Predicted synchronisation point\n"
               f"{sync_point_utc_dt.strftime('%Z %H:%M:%S %p')}\n"
               f"{max_alignment * 100:.2f}% Alignment")
               
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
    # Place text near the peak
    ax_top.text(sync_point_utc_dt, max_alignment * 100, textstr,
              fontsize=10, verticalalignment='top', horizontalalignment='center',
              bbox=props)

    ax_top.set_ylabel("Alignment (%)")
    ax_top.set_ylim(bottom=0)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis='x', length=0)
    
    summary_text = f"SAEs aligned: {sae_aligned_count}/{sae_total_count}"
    ax_top.set_title(summary_text, loc='right', fontsize=12, fontweight='bold')

    # --- 2. Bottom Plot (Player Timelines) ---
    y_levels = np.arange(len(players))
    
    for i, player in enumerate(players):
        y = y_levels[i]
        data = plot_data[player]
        
        ax_bottom.plot(data['coded'], [y] * len(data['coded']),
                       marker='v', color='blue', markersize=8, linestyle='None',
                       label='Coded Match Event' if i == 0 else "")
                       
        ax_bottom.plot(data['unaligned'], [y] * len(data['unaligned']),
                       marker='^', color='red', markersize=8, linestyle='None',
                       label='SAE' if i == 0 else "")
                       
        ax_bottom.plot(data['aligned'], [y] * len(data['aligned']),
                       marker='^', color='lime', markersize=8, linestyle='None',
                       label='Aligned SAE' if i == 0 else "")

    handles, labels = ax_bottom.get_legend_handles_labels()
    # Move legend to top plot area to match original layout
    ax_top.legend(handles, labels, loc='upper left', ncol=3, frameon=False, fontsize=10)

    ax_bottom.set_yticks(y_levels)
    ax_bottom.set_yticklabels(players) 
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
    
    # Set x-limits based on data presence
    all_times = [t for p in players for k in plot_data[p] for t in plot_data[p][k]]
    if all_times:
        max_time = max(all_times)
        ax_bottom.set_xlim(left=min(0, min(all_times)), right=max_time * 1.01)
    
    ax_bottom.grid(axis='y', linestyle='-', color='black', linewidth=1)

    # Secondary Axis (Stats)
    ax_right = ax_bottom.twinx()
    ax_right.set_yticks(y_levels) 
    
    stats_labels = []
    for player in players:
        num_aligned = len(plot_data[player]['aligned'])
        num_coded = len(plot_data[player]['coded'])
        stats_labels.append(f"{num_aligned}/{num_coded}")
        
    ax_right.set_yticklabels(stats_labels)
    ax_right.set_ylim(ax_bottom.get_ylim()) 
    ax_right.set_ylabel("Aligned SAEs / Coded Events", fontsize=10)
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()

    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    ref = load_reference_impacts(config['Reference_Impacts_CSV_path'])
    col_t, col_id = load_collisions(config['Composer_CSV_path'])
    sae_t, sae_id, sess_start = load_saes(config['XML_path'], ref, config['PLA_threshold'])
    
    if col_t is not None and sae_t is not None and sess_start:
        sync_pt, score, alignment_scores, potential_points = synchronise(col_t, sae_t, col_id, sae_id, sess_start)
        
        if sync_pt:
            dt = datetime.fromtimestamp(sync_pt, tz=timezone.utc)
            print(f"\nSUCCESS: Sync Point Found: {dt}")
            print(f"Alignment Score: {score*100:.1f}%")
            
            pdata, n_al, n_tot, plist = get_plot_data(col_t, sae_t, col_id, sae_id, sync_pt, config['Alignment_threshold'])
            
            plot_alignment(
                pdata, 
                plist, 
                dt, 
                score, 
                n_al, 
                n_tot, 
                alignment_scores, 
                potential_points
            )
        else:
            print("Sync failed.")
