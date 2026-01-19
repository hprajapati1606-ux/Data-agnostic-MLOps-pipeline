import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-detects column types and cleans them:
    1. Fills missing values.
    2. Converts text to numbers (Encoding).
    """
    try:
        df_clean = df.copy()
        
        # --- 1. NUMERICAL COLUMNS (Numbers) Handle karna ---
        # Code dhundega ki kaunse columns numbers hain
        num_cols = df_clean.select_dtypes(include=['number']).columns
        
        if len(num_cols) > 0:
            # Missing values ko 'Mean' (Average) se bharo
            imputer = SimpleImputer(strategy='mean')
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
            logger.info(f"🔢 Numerics handled: {list(num_cols)}")

        # --- 2. CATEGORICAL COLUMNS (Text) Handle karna ---
        # Code dhundega ki kaunse columns text hain
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) > 0:
            # Pehle missing values ko 'Most Frequent' (Jo sabse zyada aaya ho) se bharo
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_clean[cat_cols] = imputer_cat.fit_transform(df_clean[cat_cols])
            
            # Text ko Numbers mein badlo (Label Encoding)
            le = LabelEncoder()
            for col in cat_cols:
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            
            logger.info(f"🔤 Text converted to numbers: {list(cat_cols)}")
            
        logger.info("✅ Preprocessing Complete!")
        return df_clean

    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {e}")
        raise e