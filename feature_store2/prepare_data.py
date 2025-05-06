import pandas as pd
from datetime import datetime

# Load CSV
df = pd.read_csv("feature_repo/data/bank_customer_churn_dummy_modified.csv")

print(df.info())
# Add required timestamp column for Feast
df["event_timestamp"] = [datetime.now()] * len(df)

# Save as Parquet
df.to_parquet("feature_repo/data/churn_data.parquet", index=False)

print("✅ Parquet file created with event_timestamp.")
