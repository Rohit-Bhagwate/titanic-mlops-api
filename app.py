from fastapi import FastAPI
import numpy as np
import mlflow.pyfunc
from pydantic import BaseModel
import mlflow

app = FastAPI()

mlflow.set_tracking_uri("http://host.docker.internal:5001")
mlflow.set_registry_uri("http://host.docker.internal:5001")

# Load model
model = mlflow.pyfunc.load_model("models:/titanic-model/Production")

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

    
    prediction = model.predict(sample)

    return {"prediction": int(prediction[0])}
