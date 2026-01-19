import pandas as pd
import os
import logging

# Logging Setup: Terminal me professional messages dikhayega
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from CSV, Excel, or JSON dynamically based on file extension.
    """
    try:
        # Check agar file exist karti hai
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File nahi mili bro: {file_path}")

        # File extension nikalo (e.g., .csv or .xlsx)
        _, file_extension = os.path.splitext(file_path)
        
        logger.info(f"Detected file type: {file_extension}")

        # Data-Agnostic Loading Logic
        if file_extension.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_extension}")

        logger.info(f"✅ Data loaded successfully with shape {df.shape}")
        return df

    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise e