from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from app.preprocess import apply_intensive_engineering

app = FastAPI(title="Stormbreaker House Price API")

# Load the trained CatBoost brain
model = joblib.load("app/model.pkl")

@app.get("/")
def home():
    return {"status": "Live", "benchmark_rmse": 0.11911}

@app.post("/predict")
def predict(data: dict):
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([data])
    
    # Apply the same Engineering Habit used in training
    processed_df = apply_intensive_engineering(input_df)
    
    # Predict and convert from Log space back to Dollars
    prediction = model.predict(processed_df)
    final_price = np.expm1(prediction[0])
    
    return {"estimated_sale_price": float(round(final_price, 2))}
