import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os
import logging
import mlflow               
import mlflow.sklearn       

# Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path, model_path):
    try:
        # 1. Data Load
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded for training. Shape: {df.shape}")

        if df.shape[0] < 2:
            raise ValueError("❌ Data too small! Kam se kam 2 rows honi chahiye.")

        # 2. X aur y alag karo
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Smart Split Logic
        if len(df) < 10:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Model Selection
        if y.nunique() <= 10 or y.dtype == 'object':
            model_type = "Classification"
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            is_classification = True
        else:
            model_type = "Regression"
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            is_classification = False

        # --- 🧪 MLFLOW MAGIC STARTS HERE ---
        # Ye line MLflow ko batati hai ki experiment ka naam kya rakhna hai
        mlflow.set_experiment("Data_Agnostic_Pipeline") 
        
        with mlflow.start_run(): # CCTV On kiya
            
            # A. Tags & Params Log karo
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("data_rows", df.shape[0])
            mlflow.log_param("n_estimators", 100)

            # B. Training
            logger.info(f"Training {model_type} Model...")
            model.fit(X_train, y_train)
            
            # C. Metrics Log karo
            predictions = model.predict(X_test)
            
            if is_classification:
                acc = accuracy_score(y_test, predictions)
                mlflow.log_metric("accuracy", acc) # <--- Report Card mein likh diya
                logger.info(f"🎯 Model Accuracy: {acc * 100:.2f}%")
            else:
                err = mean_squared_error(y_test, predictions)
                mlflow.log_metric("mse", err) # <--- Report Card mein likh diya
                logger.info(f"📉 Model Error (MSE): {err:.4f}")

            # D. Model Save karo (Cloud ke liye)
            mlflow.sklearn.log_model(model, "model")
            
        # --- 🧪 MLFLOW MAGIC ENDS ---

        # Local Save
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"💾 Model saved locally at {model_path}")

    except Exception as e:
        logger.error(f"❌ Error in training: {e}")
        raise e