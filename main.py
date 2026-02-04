import os
import glob  # <--- Ye naya magic tool hai
from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.train import train_model

# --- 🪄 NEW FUNCTION: LATEST FILE DHUNDNE WALA ---
def get_latest_file(directory):
    import glob
    # 1. Saare files uthao
    list_of_files = glob.glob(os.path.join(directory, '*'))
    
    # 2. Filter: Ab humne range bada di hai
    # YAML ya System files ko ignore karne ke liye extension check
    valid_extensions = ('.csv', '.xlsx', '.xls', '.json', '.parquet', '.xml', '.txt')
    
    data_files = [f for f in list_of_files if f.lower().endswith(valid_extensions)]
    
    if not data_files:
        raise FileNotFoundError(f"❌ Bhai '{directory}' me koi valid Data File nahi mili! (Supported: CSV, Excel, JSON, XML, Parquet)")
        
    # 3. Latest select karo
    latest_file = max(data_files, key=os.path.getctime)
    return latest_file
# ---------------------------------------------------

if __name__ == "__main__":
    
    print("\n--- 🚀 Smart MLOps Pipeline Starting ---")
    
    try:
        # Step 0: Auto-Detect File (Bina naam bataye)
        raw_data_dir = "data/raw"
        raw_data_path = get_latest_file(raw_data_dir) # <--- Magic yahan ho raha hai
        
        print(f"📂 Automatically detected latest file: {raw_data_path}")

        processed_data_path = "data/processed/clean_data.csv"
        model_path = "models/model.pkl"
        
        # Step 1: Ingestion
        df = load_data(raw_data_path)
        
        # Step 2: Preprocessing
        df_clean = preprocess_data(df)
        
        # Save processed data
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df_clean.to_csv(processed_data_path, index=False)
        
        # Step 3: Model Training
        print("\n--- 🧠 Training Model ---")
        train_model(processed_data_path, model_path)
        
        print("\n--- ✅ End-to-End Pipeline Success! ---")
        
    except Exception as e:
        print(f"\n❌ Pipeline Fail ho gayi: {e}")