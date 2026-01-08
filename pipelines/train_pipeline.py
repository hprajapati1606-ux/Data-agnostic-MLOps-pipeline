from src.ingestion.ingest_data import load_data
from src.validation.validate_data import validate_data
from src.preprocessing.preprocess import preprocess_data
from src.training.train import train_model

def run_pipeline():
    # Step 1: Load data
    df = load_data("data/raw/sample.csv")

    # Step 2: Validate data
    validate_data(df)

    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Step 4: Train model
    train_model(X_train, X_test, y_train, y_test)

    print("✅ Day-1 pipeline executed successfully")
