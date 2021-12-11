from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get('/')
def home():
    return {"location": "homepage"}

@app.get("/predict")
def predict(pickup_datetime, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, passenger_count):
    pipeline = joblib.load('/app/model.joblib')
    
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    eastern = pytz.timezone("US/Eastern")
    pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    new_ride_info = {
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": float(pickup_longitude),
        "pickup_latitude": float(pickup_latitude),
        "dropoff_longitude": float(dropoff_longitude),
        "dropoff_latitude": float(dropoff_latitude),
        "passenger_count": int(passenger_count)
    }
    
    df = pd.DataFrame.from_dict(new_ride_info, orient='index').T
    
    prediction = pipeline.predict(df)
    
    return { 'fare': prediction[0] }
    
