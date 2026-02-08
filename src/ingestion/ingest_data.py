import pandas as pd
import os
import logging
import json  # <--- Ye import zaruri hai

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Super Loader: Loads data and flattens nested JSONs automatically.
    """
    try:
        # 1. Check file extension
        _, file_extension = os.path.splitext(file_path)
        ext = file_extension.lower()
        
        logger.info(f"📂 Detected file format: {ext}")

        # 2. Smart Loading Logic
        if ext == '.csv':
            df = pd.read_csv(file_path)
            
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            
        elif ext == '.json':
            # --- 🛠️ SPECIAL FIX FOR JSON ---
            # Pehle file ko raw padho, fir 'normalize' (flatten) karo
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.json_normalize(data) # Magic function jo dicts ko columns bana deta hai
            
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
            
        elif ext == '.xml':
            df = pd.read_xml(file_path)
            
        elif ext == '.txt':
            df = pd.read_csv(file_path, sep=None, engine='python')
            
        else:
            raise ValueError(f"❌ Unsupported File Format: {ext}")

        logger.info(f"✅ Data Loaded & Flattened! Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise e