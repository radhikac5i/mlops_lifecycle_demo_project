from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.pyfunc

# ---------------------------------------------------------
# FastAPI app initialization
# ---------------------------------------------------------
app = FastAPI(title="Bank Customer Churn Prediction API")

# ---------------------------------------------------------
# Configure MLflow tracking URI and model loading
# ---------------------------------------------------------
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# MODEL_NAME = "XGBoostCustomerChurnModel"
# MODEL_STAGE = "Production"

# # Load the model from MLflow Registry
# model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

model_name="XGBoostCustomerChurnModel"
model_stage = "Production"
tracking_uri = "http://127.0.0.1:5000"
def load_model_from_registry(model_name,model_stage,tracking_uri):
    """
    Loads a model directly from the MLflow Model Registry.

    Args:
        model_name (str): The registered model name.
        model_stage (str): Stage from which to load the model ("Production", "Staging", etc.).
        tracking_uri (str): URI of the MLflow tracking server.

    Returns:
        model: A loaded MLflow PyFunc model ready for inference.
    """
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Build model URI
    model_uri = f"models:/{model_name}/{model_stage}"

    print(f"🔄 Loading model from: {model_uri}")
    
    # Load and return model
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully.")
    return model

model=load_model_from_registry(model_name,model_stage,tracking_uri)
print(model)
# ---------------------------------------------------------
# Input data schema using Pydantic
# ---------------------------------------------------------
class ChurnInput(BaseModel):
    CreditScore: int
    Age: int
    Balance: float
    EstimatedSalary: float
    Tenure: int
    NumOfProducts: int
    

# ---------------------------------------------------------
# Root endpoint for health check
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Bank Churn Prediction API"}

# ---------------------------------------------------------
# Inference endpoint
# ---------------------------------------------------------
@app.post("/predict/")
def predict_churn(payload: ChurnInput):
    # Convert incoming request to DataFrame
    # Convert to DataFrame
    # input_dict = payload.dict()
    # print("🔍 Converted dict:", input_dict)

    # input_df = pd.DataFrame([input_dict])
    input_df = pd.DataFrame([payload.model_dump()])
    model=load_model_from_registry(model_name,model_stage,tracking_uri)
    # Perform prediction using the loaded MLflow model
    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction),
        "message": "Customer will churn" if prediction == 1 else "Customer will not churn"
    }
