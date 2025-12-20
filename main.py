from datetime import datetime
from Classes.ConfigClass import Config
from Classes.DataLoadersClass import (
    ReferenceDataManager, 
    CollisionDataManager, 
    SensorDataManager,
    MatchMetadataManager
)
from Classes.SynchronizationClass import Synchronizer
from Classes.VisualisationClass import Visualizer

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

        # 4. Load Match Schedule
        schedule_path = self.config.schedule_path or 'schedule.csv'
        schedule_mgr = MatchMetadataManager(schedule_path)

        # Validation Check
        if not (coll_loader.timestamps is not None and sae_loader.timestamps is not None):
            print("Aborting: Data load failure.")
            return

        print("\n--- Synchronizing ---")
        synchronizer = Synchronizer(self.config)
        result = synchronizer.synchronize(coll_loader, sae_loader)

        if result:
            print(f"Predicted Sync Point: {result.sync_dt_utc.isoformat()}")
            print(f"Max Alignment: {result.max_alignment * 100:.2f}%")
            
            # --- Get Markers & Player Statuses for this Date ---
            markers = {}
            player_statuses = {}
            
            if len(sae_loader.timestamps) > 0:
                # Determine date from session start
                ref_ts = sae_loader.session_start_utc if sae_loader.session_start_utc else sae_loader.timestamps[0]
                target_date = datetime.fromtimestamp(ref_ts)
                
                print(f"Fetching schedule data for date: {target_date.strftime('%Y-%m-%d')}")
                markers = schedule_mgr.get_markers(target_date)
                player_statuses = schedule_mgr.get_player_statuses(target_date)
                
                if markers:
                    print(f"Found markers: {list(markers.keys())}")
                if player_statuses:
                    print(f"Found statuses for {len(player_statuses)} players")
       
            print("\n--- Plotting ---")
            viz = Visualizer(self.config)
            # Pass both markers and player_statuses to the plot function
            viz.plot(coll_loader, sae_loader, result, markers=markers, player_statuses=player_statuses)
        else:
            print("Synchronization failed.")

if __name__ == "__main__":
    app = SyncApp(config_path="config.yaml")
    app.run()
