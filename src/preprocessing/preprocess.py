import pandas as pd
import numpy as np
import logging

# Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Universal Cleaner: Handles Missing Values, Categorical Encoding, AND Lists.
    """
    try:
        # 1. Handling Lists & Dictionaries (The Fix for your Error) 🛠️
        # Agar cell me List ya Dict hai, to usse String bana do taki crash na ho
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

        # 2. Separate Numeric & Categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        logger.info(f"Numerics: {list(numeric_cols)}")
        logger.info(f"Categoricals: {list(categorical_cols)}")

        # 3. Handle Missing Values (Numerics -> Mean)
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # 4. Handle Missing Values (Categorical -> Mode)
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)

        # 5. Convert Text to Numbers (Label Encoding)
        # Simple Logic: Har unique text ko ek number de do
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes

        logger.info("✅ Preprocessing Complete!")
        return df

    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {e}")
        raise e