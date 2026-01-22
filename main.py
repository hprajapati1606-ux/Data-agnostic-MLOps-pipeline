from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.train import train_model  # <-- Ye nayi line hai
import os

if __name__ == "__main__":
    
    # Raste set kar rahe hain
    raw_data_path = "data/raw/test_data.csv"
    processed_data_path = "data/processed/clean_data.csv"
    model_path = "models/model.pkl"

    print("\n--- 🚀 Pipeline Starting ---")
    
    # Step 1: Data Load Karo
    df = load_data(raw_data_path)
    
    # Step 2: Data Clean Karo
    df_clean = preprocess_data(df)
    
    # Clean Data ko Save Karo (Training ke liye)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_clean.to_csv(processed_data_path, index=False)
    
    # Step 3: Model Train Karo (Asli Magic)
    print("\n--- 🧠 Training Model ---")
    train_model(processed_data_path, model_path)
    
    print("\n--- ✅ End-to-End Pipeline Success! ---")