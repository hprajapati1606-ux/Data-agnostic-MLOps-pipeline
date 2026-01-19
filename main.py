import os
from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import preprocess_data

if __name__ == "__main__":
    
    # 1. RAW Data ka Rasta (Input)
    raw_data_path = "data/raw/test_data.csv"
    
    # 2. PROCESSED Data ka Rasta (Output - Jahan save hoga)
    processed_data_path = "data/processed/clean_data.csv"
    
    print("\n--- 🚀 Pipeline Starting ---")
    
    # Step A: Load Data
    df = load_data(raw_data_path)
    print(f"\n✅ Raw Data Loaded: {df.shape}")

    # Step B: Clean Data
    df_clean = preprocess_data(df)
    print(f"✅ Data Cleaned & Encoded")

    # Step C: Save Data (Ye hai main step)
    # Pehle check karo ki folder exist karta hai ya nahi
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # File save karo (index=False jaruri hai taaki extra numbers na aayein)
    df_clean.to_csv(processed_data_path, index=False)
    
    print(f"\n📦 Data Saved at: {processed_data_path}")
    print("--- 🏁 Data Engineering Pipeline Finished! ---")