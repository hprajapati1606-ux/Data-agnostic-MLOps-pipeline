import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from given path
    """
    df = pd.read_csv(path)
    print("✅ Data loaded successfully")
    return df
