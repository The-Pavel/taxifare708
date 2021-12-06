from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from taxifare.trainer import Trainer
from taxifare.data import get_data, clean_data, set_features_targets, holdout
from taxifare.utils import haversine_distance

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    trainer = Trainer('[CN] Shanghai 708 TaxiFare')
    print('pickup lat and long', pickup_latitude, pickup_longitude)
    print('dropoff lat and long', dropoff_latitude, dropoff_longitude)
    df = get_data()
    # clean data
    df = clean_data(df)
    # get features and targets
    x, y = set_features_targets(df)
    # do a train/test split
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2)
    
    pipeline = trainer.train(x_train, x_test, y_train, y_test)
    
    return {"distance_between_points": haversine_distance(float(pickup_latitude), float(pickup_longitude), float(dropoff_latitude), float(dropoff_longitude))}
    
    