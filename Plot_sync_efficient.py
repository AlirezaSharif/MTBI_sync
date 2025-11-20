import csv
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timezone, time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker



def parse_xml_name(xml_name):
    """
    Parses the 'Last, First' format from the XML to extract the first name
    to match the CSV's 'First' format.
    """
    if ',' in xml_name:
        parts = xml_name.split(',', 1)
        return parts[1].strip()  # Return the part after the first comma
    return xml_name.strip()

def parse_utc_timestamp(timestamp_str):
    """
    Parses the specific timestamp format from the XML file into a
    UTC timestamp in seconds.
    Format: '2023-03-24 12:45:03.430 PM +13:00'
    """
    try:
        # The %z directive handles the +13:00 timezone offset
        dt_aware = datetime.strptime(timestamp_str, '%Y-%m-%d %I:%M:%S.%f %p %z')
        # .timestamp() automatically converts to UTC seconds
        return dt_aware.timestamp()
    except ValueError as e:
        print(f"Warning: Could not parse timestamp '{timestamp_str}'. Error: {e}")
        return None

def load_collisions(csv_path):
    """
    Loads collision data from the Compser CSV file.
    
    - 'collisions' (playback time) comes from 'Qualifier: Time'
    - 'collisions_identifier' (player) comes from 'Topic Name'
    """
    collisions = []
    collisions_identifier = []
    
    try:
        # Try 'utf-16' encoding, as 'utf-8-sig' failed.
        # This is common for files exported from Windows environments.
        with open(csv_path, mode='r', encoding='utf-16') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Get playback timestamp in seconds
                    playback_time = float(row['Qualifier: Time'])
                    player_name = row['Topic Name'].strip()
                    
                    if player_name:
                        collisions.append(playback_time)
                        collisions_identifier.append(player_name)
                except (ValueError, KeyError, TypeError):
                    # Skip rows with missing or invalid data
                    continue
                    
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

    print(f"Loaded {len(collisions)} collision events from CSV.")
    return np.array(collisions), np.array(collisions_identifier)

def load_saes(xml_path, pla_threshold=14.0): # <-- NEW: Added pla_threshold parameter
    """
    Loads SAE data from the XML file.
    
    - 'saes' (UTC timestamp) comes from 'ImpactDate'
    - 'sae_identifier' (player) comes from 'Athlete Name'
    - 'session_start_utc' (UTC timestamp) from '<start_time>'
    - NOW FILTERS by 'PLA' > pla_threshold
    """
    saes = []
    sae_identifier = []
    session_start_utc = None
    
    # A safe filter to ignore junk 1970 dates
    MIN_VALID_TIMESTAMP = parse_utc_timestamp('2020-03-24 12:00:00.000 AM +13:00')    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # --- Find Session Start Time (our anchor) ---
        session_start_node = root.find('.//SESSION_INFO/start_time')
        if session_start_node is not None and session_start_node.text:
            session_start_utc = parse_utc_timestamp(session_start_node.text)
            if session_start_utc:
                print(f"Found session start: {session_start_node.text}")
            else:
                print("Warning: Could not parse session start time.")
        else:
            print("Error: Could not find <start_time> in XML. Cannot proceed.")
            return None, None, None

        # --- Find all SAE instances ---
        for instance in root.findall('.//instance'):
            player_name = None
            impact_date_str = None
            pla_value = None # <-- NEW: Initialize PLA value
            
            for label in instance.findall('label'):
                group = label.find('group')
                if group is not None:
                    if group.text == 'Athlete Name':
                        player_name = label.find('text').text
                    elif group.text == 'ImpactDate':
                        impact_date_str = label.find('text').text
                    elif group.text == 'PLA': # <-- NEW: Check for PLA group
                        try:
                            # Convert the PLA value text to a float
                            pla_value = float(label.find('text').text)
                        except (ValueError, TypeError):
                            pla_value = None # Failed to parse
            
            # --- Check all conditions before appending ---
            # We now require all 3 values to be valid
            if player_name and impact_date_str and pla_value is not None:
                parsed_name = parse_xml_name(player_name)
                utc_timestamp = parse_utc_timestamp(impact_date_str)
                
                # --- Apply data filter ---
                if parsed_name and utc_timestamp is not None:
                    
                    # --- NEW: Add the PLA threshold check ---
                    if utc_timestamp > MIN_VALID_TIMESTAMP and pla_value > pla_threshold:
                        saes.append(utc_timestamp)
                        sae_identifier.append(parsed_name)
                    
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_path}")
        return None, None, None
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error reading XML: {e}")
        return None, None, None

    # <-- NEW: Updated print statement
    print(f"Loaded {len(saes)} valid SAE events (PLA > {pla_threshold}) from XML.")
    return np.array(saes), np.array(sae_identifier), session_start_utc

def synchronise(collisions, saes, collisions_identifier, sae_identifier, session_start_time):
    """
    Python implementation of the MATLAB synchronise function.
    
    Finds the best UTC sync point (video start time) by maximizing
    the alignment between collision events and SAEs.
    """
    if not all(arr.size > 0 for arr in [collisions, saes, collisions_identifier, sae_identifier]):
        print("Error: Input arrays are empty. Cannot synchronize.")
        return None, 0, None, None
    
    if session_start_time is None:
        print("Error: No session start time provided. Cannot determine search range.")
        return None, 0, None, None
           
    # --- Magic numbers from MATLAB script ---
    alignment_window_sec = 5.0  # 5 seconds
    intervals_sec = 1           # 1 second step
    
    # --- RECOMMENDATION 1: REDUCE THIS RADIUS ---
    # 24 hours is likely massive overkill and the main cause of slowness.
    # Try 2 or 3 hours first.
    search_radius_hours = 24 
    search_radius_sec = search_radius_hours * 3600
    
    start_point_sec = session_start_time - search_radius_sec
    end_point_sec = session_start_time + search_radius_sec
    
    potential_sync_points = np.arange(
        start_point_sec, 
        end_point_sec + intervals_sec, 
        intervals_sec
    )
    alignment_prctage = np.zeros_like(potential_sync_points, dtype=float)
    
    print(f"Testing {len(potential_sync_points)} potential sync points...")

    # --- Pre-prepare data for vectorization ---
    # This avoids doing it inside the loop
    # We want sae_identifier and collisions_identifier as 2D arrays for broadcasting
    sae_id_col = sae_identifier.reshape(-1, 1)          # Shape: (Num_SAEs, 1)
    collisions_id_row = collisions_identifier.reshape(1, -1) # Shape: (1, Num_Collisions)

    # Pre-calculate the ID match matrix. This never changes.
    # This matrix is (Num_SAEs, Num_Collisions)
    # id_match_matrix[i, j] is True if SAE 'i' and Collision 'j' have the same player
    id_match_matrix = (sae_id_col == collisions_id_row)
    
    # --- Main loop ---
    for i, sync_point in enumerate(potential_sync_points):
        # Apply potential sync point
        sae_playback = saes - sync_point
        
        # --- *** START: VECTORIZED REPLACEMENT *** ---
        # 1. Reshape sae_playback for broadcasting
        sae_playback_col = sae_playback.reshape(-1, 1) # Shape: (Num_SAEs, 1)

        # 2. Calculate time difference matrix (Num_SAEs, Num_Collisions)
        #    NumPy broadcasting subtracts each collision time from each SAE time
        time_diff_matrix = np.abs(sae_playback_col - collisions)
        
        # 3. Create a boolean matrix of time matches
        time_match_matrix = (time_diff_matrix < alignment_window_sec)

        # 4. Combine with the pre-calculated ID match matrix
        #    We want where *both* ID and Time match
        combined_match_matrix = id_match_matrix & time_match_matrix

        # 5. Check if *any* collision matched a given SAE
        #    np.any(axis=1) collapses the (Num_SAEs, Num_Collisions) matrix
        #    down to a (Num_SAEs,) array.
        aligned_sae_mask = np.any(combined_match_matrix, axis=1)

        # 6. The count is just the sum of the True values
        aligned_saes_count = np.sum(aligned_sae_mask)
        # --- *** END: VECTORIZED REPLACEMENT *** ---

        if len(sae_playback) > 0:
            alignment_prctage[i] = aligned_saes_count / len(sae_playback)
            
        if i > 0 and i % 3600 == 0:
            print(f"  ...processed {i} points, "
                  f"Best alignment so far: {np.max(alignment_prctage)*100:.2f}%")

    # --- Find the best result ---
    max_alignment_prctage = np.max(alignment_prctage)
    best_index = np.argmax(alignment_prctage)
    predicted_syncpoint = potential_sync_points[best_index]
    
    return predicted_syncpoint, max_alignment_prctage, potential_sync_points, alignment_prctage

# --- NEW FUNCTION 1 ---
def get_plot_data(collisions_np, saes_np, collisions_id_np, saes_id_np, sync_point_seconds, alignment_window_sec=5.0): # Defaulted to 5.0
    """
    Calculates the final event lists for plotting after the best
    sync point has been found.
    """
    
    # Get all unique player names, sorted, to be our 'Players'
    all_players = sorted(list(set(collisions_id_np) | set(saes_id_np)))
    
    # Initialize the data structure
    plot_data = {player: {'coded': [], 'aligned': [], 'unaligned': []} for player in all_players}
    
    # 1. Populate all 'Coded Match Events' (blue)
    for i in range(len(collisions_np)):
        player = collisions_id_np[i]
        playback_time = collisions_np[i]
        if player in plot_data:
            plot_data[player]['coded'].append(playback_time)
            
    # 2. Calculate SAE playback times based on the *best* sync point
    sae_playback_times = saes_np - sync_point_seconds
    
    sae_aligned_count = 0
    sae_total_count = len(sae_playback_times)

    # 3. Populate 'Aligned' (green) and 'Unaligned' (red) SAEs
    for i in range(sae_total_count):
        current_sae_time = sae_playback_times[i]
        current_sae_id = saes_id_np[i]
        
        if current_sae_id not in plot_data:
            continue # Skip if this SAE player isn't in the collision data
            
        # Check for a match
        id_check = (collisions_id_np == current_sae_id) & \
                   (np.abs(collisions_np - current_sae_time) < alignment_window_sec)
                   
        if np.any(id_check):
            # This is an 'Aligned SAE' (green)
            plot_data[current_sae_id]['aligned'].append(current_sae_time)
            sae_aligned_count += 1
        else:
            # This is an 'Unaligned SAE' (red)
            plot_data[current_sae_id]['unaligned'].append(current_sae_time)
            
    return plot_data, sae_aligned_count, sae_total_count, all_players

# --- NEW FUNCTION 2 (MODIFIED) ---
def plot_alignment(plot_data, players, sync_point_utc_dt, max_alignment, 
                   sae_aligned_count, sae_total_count, 
                   alignment_scores, potential_sync_points):
    """
    Generates the two-part alignment plot.
    """
    
    # --- Setup Figure and Subplots ---
    fig = plt.figure(figsize=(20, 12)) # Increased height slightly for new axis
    # Create a 2-row grid: 1 part for the top plot, 3 parts for the bottom
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # --- 1. Top Plot (Alignment Score) ---
    
    # Convert UTC second timestamps to datetime objects for plotting
    sync_point_datetimes = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in potential_sync_points]
    
    ax_top.plot(sync_point_datetimes, alignment_scores * 100, color='black', linewidth=0.5)
    
    # Add the "Predicted" line and text
    ax_top.axvline(sync_point_utc_dt, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.7)
    
    textstr = (f"Predicted synchronisation point\n"
               f"{sync_point_utc_dt.strftime('%Z %H:%M:%S %p')}\n"
               f"{max_alignment * 100:.2f}% Alignment")
               
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
    # Place text annotation near the line
    ax_top.text(sync_point_utc_dt, np.max(alignment_scores) * 100, textstr,
              fontsize=10, verticalalignment='top', horizontalalignment='center',
              bbox=props, transform=ax_top.get_xaxis_transform())

    # --- 2. Bottom Plot (Player Timelines) ---
    
    # We will plot each player as a horizontal line (y=0, y=1, ...)
    y_levels = np.arange(len(players))
    
    # Plot the events for each player
    for i, player in enumerate(players):
        y = y_levels[i]
        data = plot_data[player]
        
        # 'Coded Match Event' (Blue ▼)
        ax_bottom.plot(data['coded'], [y] * len(data['coded']),
                       marker='v', color='blue', markersize=8, linestyle='None',
                       label='Coded Match Event' if i == 0 else "")
                       
        # 'SAE' (Red ▲)
        ax_bottom.plot(data['unaligned'], [y] * len(data['unaligned']),
                       marker='^', color='red', markersize=8, linestyle='None',
                       label='SAE' if i == 0 else "")
                       
        # 'Aligned SAE' (Green ▲)
        ax_bottom.plot(data['aligned'], [y] * len(data['aligned']),
                       marker='^', color='lime', markersize=8, linestyle='None',
                       label='Aligned SAE' if i == 0 else "")

    # --- Formatting (Top Plot) ---
    ax_top.set_ylabel("Alignment (%)")
    ax_top.set_ylim(bottom=0)
    # Hide x-axis labels/ticks on the top plot to merge with bottom
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis='x', length=0)
    
    # Create the legend and summary text
    handles, labels = ax_bottom.get_legend_handles_labels()
    ax_top.legend(handles, labels, loc='upper left', ncol=3, frameon=False, fontsize=10)
    
    summary_text = f"SAEs aligned: {sae_aligned_count}/{sae_total_count}"
    ax_top.set_title(summary_text, loc='right', fontsize=12, fontweight='bold')

    # --- Formatting (Bottom Plot) ---
    ax_bottom.set_yticks(y_levels)
    ax_bottom.set_yticklabels(players) # Use actual player names
    ax_bottom.set_ylim(len(players) - 0.5, -0.5) # Inverts y-axis
    ax_bottom.set_xlabel("Playback Timestamp (hh:mm:ss)", fontsize=12)
    
    # Format the x-axis (which is in seconds) to hh:mm:ss
    def format_playback_time(seconds, _):
        # Handle negative times that might appear before sync point
        sign = '-' if seconds < 0 else ''
        seconds = abs(int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{sign}{h:02d}:{m:02d}:{s:02d}"

    ax_bottom.xaxis.set_major_formatter(mticker.FuncFormatter(format_playback_time))
    
    # Set x-axis limits (e.g., from 0 to max event time)
    all_times = [t for p in players for k in plot_data[p] for t in plot_data[p][k]]
    if all_times:
        max_time = max(all_times)
        # Ensure x-limit starts at 0 or earlier, and extends past last event
        ax_bottom.set_xlim(left=min(0, min(all_times)), right=max_time * 1.01)
    
    # Add horizontal grid lines for each player
    ax_bottom.grid(axis='y', linestyle='-', color='black', linewidth=1)

    # --- *** START OF MODIFICATION *** ---
    # Create a second y-axis on the right to show the stats
    ax_right = ax_bottom.twinx()
    ax_right.set_yticks(y_levels) # Set ticks at the same player levels
    
    # Calculate and create the statistic labels
    stats_labels = []
    for player in players:
        num_aligned = len(plot_data[player]['aligned'])
        num_coded = len(plot_data[player]['coded'])
        stats_labels.append(f"{num_aligned}/{num_coded}")
        
    ax_right.set_yticklabels(stats_labels)
    ax_right.set_ylim(ax_bottom.get_ylim()) # Match the y-axis limits (inverted)
    ax_right.set_ylabel("Aligned SAEs / Coded Events", fontsize=10)
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()
    # --- *** END OF MODIFICATION *** ---

    # --- Final Display ---
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define file paths
    # <<< UPDATE THESE PATHS IF YOUR FILES ARE IN A DIFFERENT LOCATION >>>
    CSV_FILE = "2023_gbhs1stxv_vs_delasalle_compser CSV.csv"
    XML_FILE = "gbhs-firstXV-de_la_salle-2023-filtered_15.0.xml"
    
    print("--- Loading Data ---")
    collisions_np, collisions_id_np = load_collisions(CSV_FILE)
    saes_np, saes_id_np, session_start = load_saes(XML_FILE)

    if collisions_np is not None and saes_np is not None and session_start is not None:
        print("\n--- Starting Synchronization ---")
        
        # --- MODIFIED: Run the main analysis and get plot data ---
        sync_point_seconds, max_alignment, potential_sync_points, alignment_prctage = synchronise(
            collisions_np, 
            saes_np, 
            collisions_id_np, 
            saes_id_np,
            session_start
        )
        
        print("\n--- Synchronization Complete ---")
        
        if sync_point_seconds is not None:
            # Convert the UTC seconds timestamp to a readable string
            predicted_utc_datetime = datetime.fromtimestamp(
                sync_point_seconds, 
                tz=timezone.utc
            )
            
            print(f"Predicted Sync Point (UTC Seconds): {sync_point_seconds}")
            print(f"Predicted Sync Point (Human-Readable): {predicted_utc_datetime.isoformat()}")
            print(f"Max Alignment Percentage: {max_alignment * 100:.2f}%")
            
            print("\n--- Generating Plot Data ---")
            
            # --- NEW: Get the final data for plotting ---
            plot_data, sae_aligned_count, sae_total_count, player_list = get_plot_data(
                collisions_np, 
                saes_np, 
                collisions_id_np, 
                saes_id_np,
                sync_point_seconds,
                alignment_window_sec=5.0 # Must match the value in synchronise()
            )

            print(f"Total SAEs for plotting: {sae_total_count}")
            print(f"Aligned SAEs for plotting: {sae_aligned_count}")
            
            # --- NEW: Call the plotting function ---
            print("--- Displaying Plot ---")
            plot_alignment(
                plot_data,
                player_list,
                predicted_utc_datetime,
                max_alignment,
                sae_aligned_count,
                sae_total_count,
                alignment_prctage,
                potential_sync_points
            )
            
        else:
            print("Synchronization failed, possibly due to empty input data.")
    else:
        print("\nSynchronization aborted due to errors loading data.")