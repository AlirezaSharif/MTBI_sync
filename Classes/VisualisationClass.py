import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from datetime import datetime, timezone
from Classes.ConfigClass import Config
from Classes.DataLoadersClass import CollisionDataManager, SensorDataManager
from Classes.SynchronizationClass import SyncResult

class Visualizer:
    def __init__(self, config: Config):
        self.config = config
        self.marker_styles = {
            'mg_out':       {'color': 'purple', 'style': '--', 'label': 'MG Out'},
            'kickoff':      {'color': 'green',  'style': '--', 'label': 'Kickoff'},
            'end_first':    {'color': 'red',    'style': '--', 'label': 'End 1st'},
            'start_second': {'color': 'green',  'style': '--', 'label': 'Start 2nd'},
            'end_second':   {'color': 'red',    'style': '--', 'label': 'Game End'}
        }

    def _refine_halftime(self, coll_data: CollisionDataManager, markers: dict, sync_point: float):
        """
        Refines the halftime markers by finding the largest gap in CODED events 
        strictly within the scheduled end_first and start_second interval.
        """
        if not markers or 'end_first' not in markers or 'start_second' not in markers:
            return markers

        scheduled_start = markers['end_first']
        scheduled_end = markers['start_second']

        if scheduled_start >= scheduled_end:
            return markers

        # --- FIX: Convert Relative Video Time to Absolute Unix Time ---
        # coll_data.timestamps are relative (e.g., 500s). 
        # sync_point is the Unix start time (e.g., 1700000000).
        # We add them to match the domain of the 'markers'.
        all_ts_abs = np.sort(coll_data.timestamps) + sync_point
        
        # Filter events strictly within the scheduled interval
        events_in_break = all_ts_abs[(all_ts_abs > scheduled_start) & (all_ts_abs < scheduled_end)]

        # Create critical points: [Scheduled Start, ...Events..., Scheduled End]
        critical_points = np.concatenate(([scheduled_start], events_in_break, [scheduled_end]))

        diffs = np.diff(critical_points)
        max_gap_idx = np.argmax(diffs)
        max_gap_duration = diffs[max_gap_idx]

        # The gap starts at critical_points[i] and ends at critical_points[i+1]
        true_end_first = critical_points[max_gap_idx]
        true_start_second = critical_points[max_gap_idx + 1]

        if max_gap_duration > 120: 
            markers['end_first'] = true_end_first + 1
            markers['start_second'] = true_start_second - 1
            print(f"Refined Halftime: Found {max_gap_duration/60:.2f} min gap inside scheduled interval.")
        else:
            print(f"No significant gap found within scheduled halftime. Keeping original.")

        return markers

    def _refine_game_end(self, coll_data: CollisionDataManager, markers: dict, sync_point: float):
        """
        Updates 'end_second' (Game End) to be the timestamp of the very last CODED event,
        ensuring the plot ends exactly when the coding ends.
        """
        if not markers or coll_data.timestamps is None or len(coll_data.timestamps) == 0:
            return markers

        # Calculate absolute time of the last coded event
        last_coded_abs = np.max(coll_data.timestamps) + sync_point
        
        # Only update if we are essentially in the second half (or after kickoff if 2nd half undefined)
        # We check if it is later than start_second to ensure we are looking at the end of the game
        if 'start_second' in markers and last_coded_abs > markers['start_second']:
            print(f"Refined Game End: Shifted from {markers.get('end_second', 'Unknown')} to {last_coded_abs} (Last Coded Event)")
            markers['end_second'] = last_coded_abs + 30
        
        return markers
    
    def prepare_plot_data(self, coll_data: CollisionDataManager, sae_data: SensorDataManager, result: SyncResult, player_statuses: dict = None):
        """Organizes data for the timeline plot (Relative to Sync Point)."""
        
        coll_players = set(coll_data.identifiers)
        sae_players = set(sae_data.identifiers)
        schedule_players = set(player_statuses.keys()) if player_statuses else set()
        
        all_players = sorted(list(coll_players | sae_players | schedule_players))
        
        plot_data = {p: {'coded': [], 'aligned': [], 'unaligned': []} for p in all_players}
        unique_matches = {p: set() for p in all_players}
        
        for i, name in enumerate(coll_data.identifiers):
            if name in plot_data: 
                plot_data[name]['coded'].append(coll_data.timestamps[i])

        shifted_sae = sae_data.timestamps - result.sync_point
        # Access the boolean array
        sae_fps = sae_data.is_false_positive 
        sae_aligned_count = 0
        sae_devs = sae_data.device_ids
        sae_ids = sae_data.impact_ids
        
        for i, sae_t in enumerate(shifted_sae):
            name = sae_data.identifiers[i]
            is_fp = sae_fps[i] if sae_fps is not None else False
            dev_id = sae_devs[i] if sae_devs is not None else ""
            imp_id = sae_ids[i] if sae_ids is not None else ""
            
            if name not in plot_data: continue
            
            p_coll_mask = (coll_data.identifiers == name)
            p_coll_ts = coll_data.timestamps[p_coll_mask]
            
            if len(p_coll_ts) > 0 and np.min(np.abs(p_coll_ts - sae_t)) < self.config.alignment_threshold:
                # Store tuple (time, is_false_positive)
                plot_data[name]['aligned'].append((sae_t, is_fp, dev_id, imp_id))
                sae_aligned_count += 1
                matched_indices = np.where(p_coll_mask & (np.abs(coll_data.timestamps - sae_t) < self.config.alignment_threshold))[0]
                unique_matches[name].update(matched_indices)
            else:
                plot_data[name]['unaligned'].append((sae_t, is_fp, dev_id, imp_id))

        return {
            'data': plot_data,
            'players': all_players,
            'aligned_count': sae_aligned_count,
            'total_sae': len(shifted_sae),
            'matches': unique_matches
        }

    def save_outputs(self, viz_data, match_meta):
            out_name = self.config.output_name
            if not out_name: return

            # 1. Create Directory if needed
            import os
            folder = os.path.dirname(out_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            # 2. Save Figure
            plt.savefig(out_name)
            print(f"Figure saved to {out_name}")

            # 3. Save CSV
            csv_name = os.path.splitext(out_name)[0] + "_stats.csv"
            import csv
            
            # Determine Header Info
            team = self.config.team or match_meta.get('team', '')
            opponent = match_meta.get('opponent', '')
            
            headers = ['Player', 'Team', 'Opponent', 'DeviceId', 'LocalImpactId', 
                    'IsFalsePositive_final', 'IsFalsePositive_orig', 'aligned_with_cme']

            with open(csv_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for player in viz_data['players']:
                    p_data = viz_data['data'][player]
                    
                    # Build set of aligned IDs for lookup (to determine aligned_with_cme)
                    # Key: (DeviceId, LocalImpactId)
                    aligned_set = set((item[2], item[3]) for item in p_data['aligned'])
                    
                    # Iterate over ALL events (stored in 'unaligned' list per previous logic)
                    for item in p_data['unaligned']:
                        # Unpack: time, is_fp, dev_id, imp_id
                        t, is_fp_orig, dev, imp = item
                        
                        is_aligned = (dev, imp) in aligned_set
                        
                        # Logic: If aligned, it's NOT a False Positive (Final=False)
                        # If unaligned, it keeps its original status
                        is_fp_final = False if is_aligned else is_fp_orig
                        
                        writer.writerow([player, team, opponent, dev, imp, is_fp_final, is_fp_orig, is_aligned])
            
            print(f"Stats saved to {csv_name}")

    def plot(self, coll_data, sae_data, result: SyncResult, markers: dict = None, player_statuses: dict = None, match_meta=None):
        """Main plotting function."""
        
        # Refine halftime
        if markers:
            markers = self._refine_halftime(coll_data, markers, result.sync_point)
            markers = self._refine_game_end(coll_data, markers, result.sync_point)

        viz_data = self.prepare_plot_data(coll_data, sae_data, result, player_statuses)
        
        fig = plt.figure(figsize=(20, max(10, len(viz_data['players']) * 0.4))) 
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.1)
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1])

        self._plot_alignment_score(ax_top, result, viz_data)
        self._plot_timeline(ax_bottom, viz_data, result.sync_point, markers, player_statuses)

        plt.tight_layout()
        self.save_outputs(viz_data, match_meta or {})
        plt.show()

    def _plot_alignment_score(self, ax, result: SyncResult, viz_data):
        x_dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in result.all_points]
        ax.plot(x_dates, result.all_scores * 100, color='black', linewidth=0.5)
        ax.axvline(result.sync_dt_utc, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.7)

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

        handles = [
            plt.Line2D([0], [0], marker='v', color='blue', markersize=8, linestyle='None', label='Coded Match Event'),
            plt.Line2D([0], [0], marker='^', color='red', markersize=8, linestyle='None', label='SAE'),
            plt.Line2D([0], [0], marker='^', color='lime', markersize=8, linestyle='None', label='Aligned SAE')
        ]
        ax.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=10)

        ac = viz_data['aligned_count']
        tc = viz_data['total_sae']
        pct = (ac/tc*100) if tc > 0 else 0
        summary = (f"SAEs aligned: {ac}/{tc} ({pct:.0f}%), "
                   f"alignment threshold: ±{self.config.alignment_threshold}s, "
                   f"impact threshold: {self.config.pla_threshold}g")
        ax.set_title(summary, loc='right', fontsize=12, fontweight='bold')

    def _plot_timeline(self, ax, viz_data, sync_point, markers, player_statuses):
        players = viz_data['players']
        p_data = viz_data['data']
        
        # --- DETERMINE TIME ORIGIN (0 = Kickoff) ---
        t_zero = sync_point # Default to sync point (video start) if no kickoff
        is_game_time = False
        
        if markers and 'kickoff' in markers:
            t_zero = markers['kickoff']
            is_game_time = True
            
        # Shift calculation:
        # Currently p_data is relative to sync_point.
        # We want it relative to t_zero.
        # New_Time = (Old_Time_Relative_to_Sync + Sync_Point) - t_zero
        #          = Old_Time + (Sync_Point - t_zero)
        display_shift = sync_point - t_zero

        offsets = {'coded': -0.2, 'aligned': 0.0, 'unaligned': 0.2}

        # 1. Plot Player Data (Shifted)
        for i, player in enumerate(players):
            y = i
            d = p_data[player]
            if i < len(players) - 1:
                ax.axhline(y=i + 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

            
            # Coded points are just timestamps
            coded_pts = [t + display_shift for t in d['coded']]
            
            # SAE points are now (timestamp, is_fp) tuples. We must unpack them.
            # Helper to split based on config
            def split_points(data_list):
                true_pts = []
                fp_pts = []
                for item in data_list:
                    t, is_fp = item[0], item[1]
                    shifted_t = t + display_shift
                    if is_fp and self.config.false_positive_mode == 'aligned':
                        fp_pts.append(shifted_t)
                    else:
                        true_pts.append(shifted_t)
                return true_pts, fp_pts

            aligned_true, aligned_fp = split_points(d['aligned'])
            unaligned_true, unaligned_fp = split_points(d['unaligned'])
            
            ax.plot(coded_pts, [y + offsets['coded']] * len(coded_pts),
                    marker='v', color='blue', markersize=6, linestyle='None', alpha=0.8)
            
            # Plot Aligned (True Positives) -> Green
            ax.plot(aligned_true, [y + offsets['aligned']] * len(aligned_true),
                    marker='^', color='lime', markersize=6, linestyle='None', alpha=0.9)
            
            # Plot Aligned (False Positives) -> Different Color (e.g., Orange/Yellow)
            if aligned_fp:
                ax.plot(aligned_fp, [y + offsets['aligned']] * len(aligned_fp),
                        marker='^', color='orange', markersize=6, linestyle='None', alpha=0.9, label='Aligned FP')

            # Plot Unaligned -> Red
            # Note: We merge unaligned FP and TP as Red usually, unless you want them distinct too. 
            # The prompt specifically asked for "Aligned... with a different color". 
            # We will plot unaligned FPs as Red still, or maybe dark orange. Let's keep them Red for simplicity unless specified.
            all_unaligned = unaligned_true + aligned_fp + aligned_true
            ax.plot(all_unaligned, [y + offsets['unaligned']] * len(all_unaligned),
                    marker='^', color='red', markersize=6, linestyle='None', alpha=0.6)
        # 2. Plot Match Markers (Shifted)
        if markers:
            # Marker Time Relative to Origin
            def get_disp_x(ts): return ts - t_zero

            for name, ts in markers.items():
                if name in self.marker_styles and ts:
                    style = self.marker_styles[name]
                    if name == 'mg_out':
                        rel_x = -60 * 30  # 30 mins before kickoff
                    else:
                        rel_x = get_disp_x(ts)
                    ax.axvline(x=rel_x, color=style['color'], linestyle=style['style'], 
                               linewidth=2, alpha=0.8)

            regions = [
                ('mg_out',       'kickoff',      'Pre-Game',  'purple'),
                ('kickoff',      'end_first',    '1st Half',  'green'),
                ('end_first',    'start_second', 'Half Time', 'yellow'),
                ('start_second', 'end_second',   '2nd Half',  'green')
            ]

            for start_key, end_key, label_text, color in regions:
                if markers.get(start_key) and markers.get(end_key):
                    if start_key == 'mg_out':
                        x_start = -60 * 30  # 30 mins before kickoff
                    else:
                        x_start = get_disp_x(markers[start_key])
                    x_end = get_disp_x(markers[end_key])
                    if x_start < x_end:
                        ax.axvspan(x_start, x_end, color=color, alpha=0.1, label=label_text)
                        mid_point = (x_start + x_end) / 2
                        ax.text(mid_point, -0.6, label_text, 
                                ha='center', va='bottom', 
                                fontsize=10, fontweight='bold', color='black',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        # Axes Formatting
        ax.set_yticks(np.arange(len(players)))
        
        y_labels = []
        for p in players:
            label = p
            status = None
            if player_statuses:
                if p in player_statuses: status = player_statuses[p]
                else:
                    for k, v in player_statuses.items():
                        if k in p or p in k:
                            status = v
                            break
            if status and status.lower() not in ['p', 'playing', 'present', '']:
                label += f" ({status.title()})"
            y_labels.append(label)
        
        ax.set_yticklabels(y_labels)
        ax.set_ylim(len(players) - 0.5, -0.5) 
        
        xlabel = "Time Relative to Kickoff (hh:mm:ss)" if is_game_time else "Playback Timestamp (hh:mm:ss)"
        ax.set_xlabel(xlabel, fontsize=12)

        def time_fmt(x, pos):
            s = int(x) 
            sign = '-' if s < 0 else ''
            s = abs(s)
            return f"{sign}{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_fmt))
        
        # --- Set Limits ---
        # Collect all points to determine right limit
        all_times = []
        for p in p_data.values():
            # Add shift to these for limit calculation
            # 1. Coded data is still just a list of timestamps
            all_times.extend([t + display_shift for t in p['coded']])
            
            # 2. Aligned/Unaligned are now tuples: (time, is_fp)
            # We must access item[0] to get the time
            all_times.extend([item[0] + display_shift for item in p['aligned']])
            all_times.extend([item[0] + display_shift for item in p['unaligned']])
        
        if markers:
            for ts in markers.values():
                if ts: all_times.append(ts - t_zero)

        if all_times:
            # Left limit: 30 mins before game (Kickoff=0, so -1800)
            # However, if we don't have kickoff, we fall back to standard margins
            if is_game_time:
                x_left = -1800 # 30 mins before kickoff
            else:
                x_left = min(all_times) - 60

            if markers and 'end_second' in markers and is_game_time:
                 x_right = markers['end_second'] - t_zero
            else:
                 # Standard fallback (max event time)
                 x_right = max(all_times)
            ax.set_xlim(left=x_left, right=x_right)
        else:
            ax.set_xlim(-60, 60)

        ax_right = ax.twinx()
        ax_right.set_yticks(np.arange(len(players)))
        stats_labels = [f"{len(viz_data['matches'][p])}/{len(p_data[p]['coded'])}" for p in players]
        ax_right.set_yticklabels(stats_labels)
        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_ylabel("Unique Coded Events Matched / Total Coded", fontsize=10)