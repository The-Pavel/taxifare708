from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from taxifare.trainer import Trainer
from taxifare.data import get_data, clean_data, set_features_targets, holdout
import pandas as pd
from datetime import datetime
import pytz

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
    trainer = Trainer('[CN] Shanghai 708 TaxiFare-API')
    df = get_data()
    # clean data
    df = clean_data(df)
    # get features and targets
    x, y = set_features_targets(df)

    # do a train/test split
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2)
    
    pipeline = trainer.train(x_train, x_test, y_train, y_test)
    
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
    
