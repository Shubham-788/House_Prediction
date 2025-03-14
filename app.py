import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def home():
    return {"message": "California House Price Prediction API"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Convert input to array and scale
        input_data = np.array([[features.MedInc, features.HouseAge, features.AveRooms, 
                                features.AveBedrms, features.Population, features.AveOccup,
                                features.Latitude, features.Longitude]])
        input_scaled = scaler.transform(input_data)
        
        # Predict house price
        prediction = model.predict(input_scaled)[0]
        return {"predicted_price": round(prediction, 2)*100000}

    except Exception as e:
        return {"error": str(e)}

