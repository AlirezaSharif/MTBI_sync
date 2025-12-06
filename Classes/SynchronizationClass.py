import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from Classes.ConfigClass import Config
from Classes.DataLoadersClass import CollisionDataManager, SensorDataManager

@dataclass
class SyncResult:
    sync_point: float
    max_alignment: float
    all_points: np.ndarray
    all_scores: np.ndarray
    
    @property
    def sync_dt_utc(self):
        return datetime.fromtimestamp(self.sync_point, tz=timezone.utc)


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

