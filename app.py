import streamlit as st
import pandas as pd
import os
import time
from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.train import train_model

# Page Configuration (Tab ka naam aur icon)
st.set_page_config(page_title="MLOps Pipeline", page_icon="🚀", layout="wide")

# Title and Description
st.title("🤖 Data-Agnostic MLOps Pipeline")
st.markdown("""
**Welcome!** Upload any raw dataset (CSV, Excel, JSON), and this pipeline will automatically:
1. 📥 **Ingest** the data
2. 🧹 **Clean & Preprocess** it
3. 🧠 **Train a Model**
""")

# Sidebar for controls
st.sidebar.header("🔧 Pipeline Settings")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "json"])

# Main App Logic
if uploaded_file is not None:
    # 1. Save Uploaded File
    raw_path = os.path.join("data", "raw", uploaded_file.name)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"✅ File Uploaded: {uploaded_file.name}")

    # 2. Show Raw Data
    st.subheader("1️⃣ Raw Data Preview")
    try:
        df = load_data(raw_path)
        st.dataframe(df.head())
        st.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Error loading file: {e}")

    # 3. Train Button
    if st.button("🚀 Run MLOps Pipeline", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Stage A: Preprocessing
            status_text.text("🧹 Cleaning data...")
            progress_bar.progress(30)
            time.sleep(1) # Thoda drama (visual effect) ke liye
            
            df_clean = preprocess_data(df)
            
            # Save processed data
            clean_path = os.path.join("data", "processed", "clean_data.csv")
            os.makedirs(os.path.dirname(clean_path), exist_ok=True)
            df_clean.to_csv(clean_path, index=False)
            
            st.subheader("2️⃣ Processed Data")
        # Count dikhayega
            st.info(f"Total Rows Processed: {df_clean.shape[0]} | Total Columns: {df_clean.shape[1]}") 

        # Preview dikhayega (Sirf top 10 rows)
            st.dataframe(df_clean.head(10)) 
            st.success("Data Cleaned & Encoded Successfully!")

            # Stage B: Training
            status_text.text("🧠 Training Model...")
            progress_bar.progress(70)
            time.sleep(1)
            
            model_path = os.path.join("models", "model.pkl")
            train_model(clean_path, model_path)
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            st.balloons() # Party time! 🎈
            st.success(f"Model Trained & Saved at `{model_path}`")

        except Exception as e:
            st.error(f"Pipeline Failed: {e}")

else:
    st.info("👈 Please upload a dataset from the sidebar to start.")