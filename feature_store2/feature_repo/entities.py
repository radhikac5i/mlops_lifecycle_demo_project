# churn_feature_store/entities.py
from feast import Entity, ValueType

customer = Entity(
    name="CustomerID",
    value_type=ValueType.INT64,
    description="Customer ID",
)
