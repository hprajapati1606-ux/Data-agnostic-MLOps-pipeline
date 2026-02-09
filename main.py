from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.train import train_model
from src.utils.common import get_latest_file  # <--- Ye humne abhi banaya tha!
import os
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    # 1. Folders Define karo (File name nahi!)
    raw_data_dir = "data/raw/"
    processed_data_path = "data/processed/clean_data.csv"
    model_path = "models/model.pkl"

    print("\n--- 🚀 Universal Pipeline Starting ---")
    
    try:
        # Step 0: Auto-Detect File
        # Manager khud dhoondhega ki konsi file padi hai
        logger.info("🔍 Scanning for data...")
        file_path = get_latest_file(raw_data_dir)
        
        # Step 1: Data Load (Jo file mili, use load karo)
        logger.info(f"📂 Dynamically loading: {file_path}")
        df = load_data(file_path)
        
        # Step 2: Data Clean
        # Chahe columns kuch bhi ho, ye khud handle karega
        logger.info("🧹 Auto-Cleaning data...")
        df_clean = preprocess_data(df)
        
        # Save Clean Data (Training module isko padhega)
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df_clean.to_csv(processed_data_path, index=False)
        logger.info(f"💾 Clean data saved to: {processed_data_path}")
        
        # Step 3: Model Train
        # Regression ho ya Classification, ye khud decide karega
        logger.info("🧠 Auto-Training Model...")
        train_model(processed_data_path, model_path)
        
        print("\n--- ✅ Pipeline Finished Successfully! ---")
        
    except Exception as e:
        logger.error(f"❌ Pipeline Failed: {e}")
        print("\n--- ❌ Pipeline Failed ---")