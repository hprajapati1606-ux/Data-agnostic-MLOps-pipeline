import pandas as pd
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, model_save_path: str, target_column: str) -> None:
    """
    Trains a model predicting the specific 'target_column'.
    Auto-detects Regression vs Classification based on target values.
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # Separate Features (X) and Target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"🎯 Target set to: '{target_column}'")

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Auto-Detect Problem Type
        unique_values = y.nunique()
        if unique_values <= 10:
            problem_type = "Classification"
            model = LogisticRegression(max_iter=1000)
        else:
            problem_type = "Regression"
            model = LinearRegression()
            
        logger.info(f"🕵️ Detected Problem Type: {problem_type}")

        # Train Model
        model.fit(X_train, y_train)

        # Save Model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        
        # Save Metadata (to remember what we trained for)
        meta_data = {"target_col": target_column, "problem_type": problem_type}
        joblib.dump(meta_data, model_save_path.replace(".pkl", "_meta.pkl"))
        
        logger.info(f"💾 Model and Metadata saved successfully.")

    except Exception as e:
        logger.error(f"❌ Training Error: {e}")
        raise e