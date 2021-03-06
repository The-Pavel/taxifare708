import pandas as pd
from sklearn.model_selection import train_test_split
### CHANGE THESE TO YOURS 👇
BUCKET_NAME="lewagon-data-708-pavel"
BUCKET_TRAIN_DATA_PATH="data/train_10k.csv"

def get_data(local=True):
    """
    Reads data from a csv file and returns a list of dictionaries.
    local=True loads data from local CSV, false loads from GCP Storage
    """
    if local:
        return pd.read_csv('./raw_data/train_10k.csv')
    return pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
                       
def clean_data(df):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    df = df[df.fare_amount.between(0, 1000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(40, 42)]
    df = df[df["pickup_longitude"].between(-74.3, -72.9 )]
    df = df[df["dropoff_latitude"].between(40, 42)]
    df = df[df["dropoff_longitude"].between(-74, -72.9)]
    # drop key column since it is not helpful and not dynamic
    df.drop('key', axis='columns', inplace=True)
    return df

def set_features_targets(df):
    y = df.pop('fare_amount')
    x = df
    return x, y

def holdout(x, y, test_size=0.3):
    """
    Splits data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return X_train, X_test, y_train, y_test
    
    