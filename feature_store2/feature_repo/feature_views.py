# churn_feature_store/feature_views.py
from feast import FeatureView, Field
from feast.types import Int64, Float64
from feast import FileSource
from datetime import timedelta
from entities import customer

customer_data = FileSource(
    path="data/churn_data.parquet",
    timestamp_field="event_timestamp",
)

customer_features1 = FeatureView(
    name="customer_features1",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="CreditScore", dtype=Int64),
        Field(name="Age", dtype=Int64),
        Field(name="Balance", dtype=Float64),
        Field(name="EstimatedSalary", dtype=Float64),
        Field(name="Tenure", dtype=Int64),
        Field(name="NumOfProducts", dtype=Float64),
    ],
    source=customer_data,
)