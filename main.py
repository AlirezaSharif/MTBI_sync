from Classes.ConfigClass import Config
from Classes.DataLoadersClass import ReferenceDataManager, CollisionDataManager, SensorDataManager
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

        if not (coll_loader.timestamps is not None and sae_loader.timestamps is not None):
            print("Aborting: Data load failure.")
            return

        print("\n--- Synchronizing ---")
        synchronizer = Synchronizer(self.config)
        result = synchronizer.synchronize(coll_loader, sae_loader)

        if result:
            print(f"Predicted Sync Point: {result.sync_dt_utc.isoformat()}")
            print(f"Max Alignment: {result.max_alignment * 100:.2f}%")
            
            print("\n--- Plotting ---")
            viz = Visualizer(self.config)
            viz.plot(coll_loader, sae_loader, result)
        else:
            print("Synchronization failed.")

if __name__ == "__main__":
    app = SyncApp(config_path="config2.yaml")
    app.run()
