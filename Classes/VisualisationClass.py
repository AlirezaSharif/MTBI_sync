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
