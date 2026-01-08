def validate_data(df):
    """
    Basic data validation
    """
    print("🔍 Checking missing values...")
    missing = df.isnull().sum()
    print(missing)
    return missing
