from feast import FeatureStore
import pandas as pd

# Initialize the Feature Store (assumes you're in the same folder as feature_store.yaml)
store = FeatureStore(repo_path="feature_store2/feature_repo")

# Load entity + timestamp information from the same data used to generate features
entity_df = pd.read_parquet("feature_store2/feature_repo/data/churn_data.parquet")[["CustomerID", "event_timestamp"]]

# Fetch features from the feature view registered in Feast
feature_data = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_features1:CreditScore",
        "customer_features1:Age",
        "customer_features1:Balance",
        "customer_features1:EstimatedSalary",
        "customer_features1:Tenure",
        "customer_features1:NumOfProducts"

    ],
).to_df()

# Join the label (Exited) for supervised learning
labels = pd.read_parquet("feature_store2/feature_repo/data/churn_data.parquet")[["CustomerID", "event_timestamp", "Exited"]]

# Localize label timestamps to UTC (to match Feast's output)
labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"]).dt.tz_localize("UTC")

# Merge feature matrix with label
df_train = feature_data.merge(labels, on=["CustomerID", "event_timestamp"], how="left")

# Optional: print preview
print("✅ Training data shape:", df_train.shape)
print(df_train.info())
# Save to CSV without index
df_train.to_csv("data_store/feast_training_data.csv", index=False)