from datetime import datetime
from Classes.ConfigClass import Config
from Classes.DataLoadersClass import (
    ReferenceDataManager, 
    CollisionDataManager, 
    SensorDataManager,
    MatchMetadataManager  # <--- Added Import
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

        # 4. Load Match Schedule (NEW)
        # Note: Ensure your config.yaml has a 'schedule_path' entry, 
        # or it will default to 'schedule.csv' in the current directory.
        #schedule_path = getattr(self.config, self.config.schedule_path, 'schedule.csv')
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
            
            # --- NEW: Get Markers for this Date ---
            markers = {}
            if len(sae_loader.timestamps) > 0:
                # Use session start if available, otherwise use the first event timestamp
                # to determine which date we are looking at.
                ref_ts = sae_loader.session_start_utc if sae_loader.session_start_utc else sae_loader.timestamps[0]
                target_date = datetime.fromtimestamp(ref_ts)
                
                print(f"Fetching schedule markers for date: {target_date.strftime('%Y-%m-%d')}")
                markers = schedule_mgr.get_markers(target_date)
                if markers:
                    print(f"Found markers: {list(markers.keys())}")
                else:
                    print("No markers found in schedule for this date.")
       
            print("\n--- Plotting ---")
            viz = Visualizer(self.config)
            # Pass the markers dictionary to the plot function
            viz.plot(coll_loader, sae_loader, result, markers=markers)
        else:
            print("Synchronization failed.")

if __name__ == "__main__":
    app = SyncApp(config_path="config.yaml")
    app.run()
