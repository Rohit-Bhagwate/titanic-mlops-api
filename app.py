from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: int
    Embarked_S: int

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API"}

@app.post("/predict")
def predict(data: Passenger):

    sample = [[
        data.Pclass,
        data.Sex,
        data.Age,
        data.SibSp,
        data.Parch,
        data.Fare,
        data.Embarked_Q,
        data.Embarked_S
    ]]

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)

    return {"prediction": int(prediction[0])}
