import os
import sys
from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.train import train_model

def main():
    # 1. Start
    print("\n--- Smart MLOps Pipeline Starting ---")

    # 2. Ingest
    raw_data_dir = os.path.join("data", "raw")
    files = os.listdir(raw_data_dir)
    
    if not files:
        print("Error: No files found in data/raw!")
        return

    file_path = os.path.join(raw_data_dir, files[0])
    print(f"Processing file: {file_path}")

    df = load_data(file_path)

    # 3. Preprocess
    df_clean = preprocess_data(df)
    
    processed_path = os.path.join("data", "processed", "clean_data.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    print("Preprocessing Complete.")

    # 4. Train
    model_path = os.path.join("models", "random_forest.pkl")
    print("Training Model...")
    train_model(processed_path, model_path)
    
    print("End-to-End Pipeline Success!")

if __name__ == "__main__":
    main()