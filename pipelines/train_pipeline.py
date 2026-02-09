import os
import pandas as pd

from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.train import train_model

def run_pipeline():
    print("\n--- 🚀 Pipeline Starting ---")

    # 🔴 YAHAN APNA REAL DATASET PATH DAALO
    raw_data_path = "data/raw/telecom_churn.csv"
    processed_data_path = "data/processed/clean_data.csv"

    # 1️⃣ Load raw data
    df_raw = load_data(raw_data_path)

    # 2️⃣ Preprocess
    df_clean = preprocess_data(df_raw)

    # 3️⃣ Save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_clean.to_csv(processed_data_path, index=False)

    # 4️⃣ Train model (DAY-3 style)
    train_model(df_clean)

    print("🏁 DAY-3 PIPELINE FINISHED SUCCESSFULLY")
