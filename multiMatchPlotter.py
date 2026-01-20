import yaml
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

class MultiMatchPlotter:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.player_name = self.config.get('player_name')
        
    def _load_config(self, path):
        if not os.path.exists(path):
            print(f"Error: Config file {path} not found.")
            sys.exit(1)
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self):
        """Iterates through files and extracts data for the specific player."""
        matches_data = []
        
        files_list = self.config.get('files', [])
        
        for file_entry in files_list:
            filepath = file_entry.get('path')
            custom_label = file_entry.get('label')
            
            if not os.path.exists(filepath):
                print(f"Warning: File {filepath} not found. Skipping.")
                continue
                
            try:
                df = pd.read_csv(filepath)
                
                # Check required columns
                if 'Player' not in df.columns:
                    print(f"Warning: File {filepath} missing 'Player' column. Skipping.")
                    continue
                
                # Filter for player
                player_df = df[df['Player'] == self.player_name].copy()
                
                if player_df.empty:
                    print(f"Info: Player '{self.player_name}' not found in {filepath}.")
                    # We still add an empty entry to show the match existed but player had no events
                    match_label = custom_label or "Unknown Match"
                else:
                    # Determine Label (Use Custom, or 'Opponent' col, or Filename)
                    if custom_label:
                        match_label = custom_label
                    elif 'Opponent' in player_df.columns:
                        match_label = f"{player_df['Opponent'].iloc[0]}"
                    else:
                        match_label = os.path.basename(filepath)

                matches_data.append({
                    'label': match_label,
                    'data': player_df
                })
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                
        return matches_data

    def plot(self):
        match_list = self.load_data()
        if not match_list:
            print("No data loaded.")
            return

        # Setup Figure
        # Height dynamic based on number of matches
        fig, ax = plt.subplots(figsize=(15, max(4, len(match_list) * 1.5)))
        
        # Y-Positions (0 is top match, len-1 is bottom match)
        y_positions = range(len(match_list))
        
        # Store all times for x-axis limits
        all_times = []

        efficiency_labels = []

        for i, match in enumerate(match_list):
            df = match['data']
            y = i

            OFF_CODED = -0.2      # Shift Blue Up
            OFF_ALIGNED = 0.0     # Keep Green Center
            OFF_UNALIGNED = 0.2  # Shift Red Down
            
            # Draw Timeline Line
            ax.axhline(y=y + 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.2) # Separator
            ax.axhline(y=y, color='gray', linestyle='-', linewidth=1, alpha=0.1)
            
            if df.empty:
                efficiency_labels.append("N/A")
                continue
            if 'MatchEfficiency' in df.columns:
                eff = df['MatchEfficiency'].iloc[0]
                efficiency_labels.append(str(eff))
            else:
                efficiency_labels.append("")
            
            if 'CMETime' in df.columns:
                cme_data = df['CMETime'].dropna().values
                if len(cme_data) > 0:
                    ax.scatter(cme_data, [y + OFF_CODED] * len(cme_data), 
                               marker='v', c='blue', s=80, alpha=0.8, label='Coded Event', zorder=2)
                    all_times.extend(cme_data)
            # Extract Data
            times = df['Time'].values
            is_aligned = df['aligned_with_cme'].astype(bool).values
            is_fp = df['IsFalsePositive_final'].astype(bool).values
            
            all_times.extend(times)
            
            # --- Plotting Logic ---
            
            # 1. Aligned Events (True Positives) -> Green
            # Condition: Aligned AND Not FP
            mask_aligned = is_aligned
            if np.any(mask_aligned):
                ax.scatter(times[mask_aligned], [y + OFF_ALIGNED] * np.sum(mask_aligned), 
                               marker='^', c='lime', s=80, alpha=0.9, 
                               label='Aligned (TP)', zorder=3)
            # 2. Unaligned Events (Potential True Positives missed by video or just unverified) -> Red
            # Condition: Not Aligned AND Not FP
            mask_unaligned = (~is_aligned) & (~is_fp)
            if np.any(mask_unaligned):
                ax.scatter(times[mask_unaligned], [y + OFF_UNALIGNED] * np.sum(mask_unaligned), 
                               marker='^', c='red', s=80, alpha=0.6, 
                               label='Unaligned', zorder=2)
            # 3. False Positives (if any) -> Orange
            mask_fp = is_fp
            if np.any(mask_fp):
                ax.scatter(times[mask_fp], [y + OFF_ALIGNED] * np.sum(mask_fp), 
                           marker='x', c='orange', s=80, linewidth=2, label='False Positive', zorder=2)

        # --- Formatting ---
        
        # Y-Axis
        ax.set_yticks(y_positions)
        ax.set_yticklabels([m['label'] for m in match_list], fontsize=12)
        ax.invert_yaxis() # Top match is first in list
        

        ax_right = ax.twinx()
        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_yticks(y_positions)
        ax_right.set_yticklabels(efficiency_labels, fontsize=11)
        ax_right.set_ylabel("Matched / Total Coded", fontsize=11, rotation=270, labelpad=20)     
        # X-Axis Time Formatter
        def time_fmt(x, pos):
            s = int(x)
            sign = '-' if s < 0 else ''
            s = abs(s)
            return f"{sign}{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_fmt))
        ax.set_xlabel("Time Relative to Kickoff (HH:MM:SS)", fontsize=12)
        
        # Grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8) # Kickoff Line
        
        # Title
        ax.set_title(f"Season Impact Timeline: {self.player_name}", fontsize=14, fontweight='bold')
        
        # Legend (Deduplicate handles)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper center', 
                      bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        
        # Limits
        if all_times:
            ax.set_xlim(min(min(all_times), -60) - 60, max(all_times) + 60)
        
        plt.tight_layout()
        
        # Save
        out_path = self.config.get('output_plot', 'timeline_plot.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {out_path}")
        plt.show()

if __name__ == "__main__":
    # You can pass the config file as an argument or default to 'multi_match_config.yaml'
    config_file = "multi_match_config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        
    plotter = MultiMatchPlotter(config_file)
    plotter.plot()
