# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Classifier API")


# Load model
model = joblib.load("model/model.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Classifier API!"}

@app.post("/predict")
def predict_species(features: dict):
    input_df = pd.DataFrame([features])
    prediction = int(model.predict(input_df)[0])
    return {
        "predicted_class": prediction
}