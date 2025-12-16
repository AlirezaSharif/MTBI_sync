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
        # Definitions for the vertical lines and regions
        self.marker_styles = {
            'mg_out':       {'color': 'purple', 'style': '--', 'label': 'MG Out'},
            'kickoff':      {'color': 'green',  'style': '--',  'label': 'Kickoff'},
            'end_first':    {'color': 'red',    'style': '--', 'label': 'End 1st'},
            'start_second': {'color': 'green',  'style': '--', 'label': 'Start 2nd'},
            'end_second':   {'color': 'red',  'style': '--',  'label': 'Game End'}
        }

    def prepare_plot_data(self, coll_data: CollisionDataManager, sae_data: SensorDataManager, result: SyncResult):
        """Organizes data for the timeline plot."""
        
        all_players = sorted(list(set(coll_data.identifiers) | set(sae_data.identifiers)))
        plot_data = {p: {'coded': [], 'aligned': [], 'unaligned': []} for p in all_players}
        unique_matches = {p: set() for p in all_players}
        
        # 1. Populate Coded (Collisions)
        # Note: Collisions are already relative to video time (usually), or need shifting?
        # Assuming coll_data.timestamps are already in "video time" or similar.
        for i, name in enumerate(coll_data.identifiers):
            plot_data[name]['coded'].append(coll_data.timestamps[i])

        # 2. Populate SAEs
        # Shift absolute timestamps to relative time based on sync point
        shifted_sae = sae_data.timestamps - result.sync_point
        sae_aligned_count = 0
        
        for i, sae_t in enumerate(shifted_sae):
            name = sae_data.identifiers[i]
            if name not in plot_data: continue
            
            # Filter collisions for this player
            p_coll_mask = (coll_data.identifiers == name)
            p_coll_ts = coll_data.timestamps[p_coll_mask]
            
            # Check alignment against original collisions
            if len(p_coll_ts) > 0 and np.min(np.abs(p_coll_ts - sae_t)) < self.config.alignment_threshold:
                plot_data[name]['aligned'].append(sae_t)
                sae_aligned_count += 1
                
                # Mark which coded event was hit
                matched_indices = np.where(p_coll_mask & (np.abs(coll_data.timestamps - sae_t) < self.config.alignment_threshold))[0]
                unique_matches[name].update(matched_indices)
            else:
                plot_data[name]['unaligned'].append(sae_t)

        return {
            'data': plot_data,
            'players': all_players,
            'aligned_count': sae_aligned_count,
            'total_sae': len(shifted_sae),
            'matches': unique_matches
        }

    def plot(self, coll_data, sae_data, result: SyncResult, markers: dict = None):
        """
        Main plotting function.
        :param markers: Optional dictionary of {event_name: timestamp_float}
        """
        viz_data = self.prepare_plot_data(coll_data, sae_data, result)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.1)
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1])

        self._plot_alignment_score(ax_top, result, viz_data)
        # Pass markers to the timeline plotter
        self._plot_timeline(ax_bottom, viz_data, result.sync_point, markers)

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

    def _plot_timeline(self, ax, viz_data, sync_point, markers):
        players = viz_data['players']
        p_data = viz_data['data']
        
        offsets = {'coded': -0.2, 'aligned': 0.0, 'unaligned': 0.2}

        # 1. Plot Player Data
        for i, player in enumerate(players):
            y = i
            d = p_data[player]
            
            ax.plot(d['coded'], [y + offsets['coded']] * len(d['coded']),
                    marker='v', color='blue', markersize=6, linestyle='None', alpha=0.8)
            
            ax.plot(d['aligned'], [y + offsets['aligned']] * len(d['aligned']),
                    marker='^', color='lime', markersize=6, linestyle='None', alpha=0.9)
            
            ax.plot(d['unaligned'], [y + offsets['unaligned']] * len(d['unaligned']),
                    marker='^', color='red', markersize=6, linestyle='None', alpha=0.6)
            # Ghost marker
            ax.plot(d['aligned'], [y + offsets['unaligned']] * len(d['aligned']),
                     marker='^', color='red', markersize=6, linestyle='None', alpha=0.6)

            if i < len(players) - 1:
                ax.axhline(y=i + 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

        
        # 2. Plot Match Markers & Regions
        if markers:
            # Helper: Shift absolute marker time to relative plot time
            def get_rel_x(ts): return ts - sync_point

            # A. Draw Vertical Lines (Visual separators only, no text)
            for name, ts in markers.items():
                if name in self.marker_styles and ts:
                    style = self.marker_styles[name]
                    rel_x = get_rel_x(ts)
                    ax.axvline(x=rel_x, color=style['color'], linestyle=style['style'], 
                               linewidth=2, alpha=0.8)

            # B. Plot Regions (Shading + Centered Labels)
            # Define regions: (start_marker, end_marker, Label, Color)
            regions = [
                ('mg_out',       'kickoff',      'Pre-Game',  'purple'),
                ('kickoff',      'end_first',    '1st Half',  'green'),
                ('end_first',    'start_second', 'Half Time', 'yellow'),
                ('start_second', 'end_second',   '2nd Half',  'green')
            ]

            for start_key, end_key, label_text, color in regions:
                if markers.get(start_key) and markers.get(end_key):
                    x_start = get_rel_x(markers[start_key])
                    x_end = get_rel_x(markers[end_key])
                    
                    # 1. Shade the region
                    # Note: We keep label=label_text so it appears in the Legend once
                    # Matplotlib handles duplicate labels automatically if we plot carefully, 
                    # but usually, it's safer to only label specific handles if duplicates appear.
                    # For now, adding label here is fine.
                    ax.axvspan(x_start, x_end, color=color, alpha=0.1, label=label_text)
                    
                    # 2. Add Text Label centered at the top
                    mid_point = (x_start + x_end) / 2
                    # y = -0.6 places it just above the top player (since y=0 is the first row)
                    ax.text(mid_point, -0.6, label_text, 
                            ha='center', va='bottom', 
                            fontsize=10, fontweight='bold', color='black',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        # Axes Formatting
        ax.set_yticks(np.arange(len(players)))
        ax.set_yticklabels(players)
        ax.set_ylim(len(players) - 0.5, -0.5) # Invert
        ax.set_xlabel("Playback Timestamp (hh:mm:ss)", fontsize=12)

        def time_fmt(x, pos):
            s = int(x) # x is already in seconds relative to sync
            sign = '-' if s < 0 else ''
            s = abs(s)
            return f"{sign}{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_fmt))
        
        # Calculate X limits automatically
        all_times = []
        for p in p_data.values():
            all_times.extend(p['coded'])
            all_times.extend(p['aligned'])
            all_times.extend(p['unaligned'])
        
        # Ensure markers are included in the view limits
        if markers:
            for ts in markers.values():
                if ts: all_times.append(ts - sync_point)

        if all_times:
            x_min, x_max = min(all_times), max(all_times)
            # Add some padding (5%)
            margin = (x_max - x_min) * 0.001
            ax.set_xlim(left=x_min - margin, right=x_max + margin)

        # Secondary Axis (Stats)
        ax_right = ax.twinx()
        ax_right.set_yticks(np.arange(len(players)))
        stats_labels = [f"{len(viz_data['matches'][p])}/{len(p_data[p]['coded'])}" for p in players]
        ax_right.set_yticklabels(stats_labels)
        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_ylabel("Unique Coded Events Matched / Total Coded", fontsize=10)