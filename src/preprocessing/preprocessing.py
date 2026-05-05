import pandas as pd

def preprocess_transactions(input_path="../../data/raw/transactions.csv", output_path="../../data/preprocessed/transactions_preprocessed.csv"):
    # Define expected schema
    expected_columns = ["user_id", "product_id", "timestamp", "quantity", "price"]
    expected_dtypes = {
        "user_id": "object",
        "product_id": "object", 
        "timestamp": "datetime64[ns]",
        "quantity": "int64",
        "price": "float64"
    }
    
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    
    # Validate schema
    if list(df.columns) != expected_columns:
        raise ValueError(f"Unexpected columns. Expected: {expected_columns}, Got: {list(df.columns)}")
    
    for col, dtype in expected_dtypes.items():
        if str(df[col].dtype) != dtype:
            raise ValueError(f"Unexpected dtype for {col}. Expected: {dtype}, Got: {df[col].dtype}")
    
    # Handle missing values
    df = df.dropna()
    
    # Filter out transactions with non-positive total price
    df = df[df["price"] > 0]
    
    #Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Save preprocessed data
    df.to_csv(output_path, index=False)
    
    print(f"Preprocessed data saved to {output_path}")
    
if __name__ == "__main__":
    preprocess_transactions()