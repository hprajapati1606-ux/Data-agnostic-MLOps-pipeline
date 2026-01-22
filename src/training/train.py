import pandas as pd
import pickle
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path, model_path):
    try:
        # 1. Data Load
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded for training: {df.shape}")

        # Assume last column is Target (y), rest are Features (X)
        X = df.iloc[:, :-1] # Saare columns except last
        y = df.iloc[:, -1]  # Sirf last column (Target)

        # 2. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Model Train (Random Forest)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        logger.info("🤖 Model Training Complete!")

        # 4. Evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logger.info(f"📉 Model Error (MSE): {mse}")

        # 5. Save Model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"📦 Model saved at {model_path}")

    except Exception as e:
        logger.error(f"❌ Error in training: {e}")
        raise e