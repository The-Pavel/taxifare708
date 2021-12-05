from taxifare.data import get_data, clean_data, holdout, set_features_targets
from taxifare.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import mlflow
from mlflow.tracking import MlflowClient
from google.cloud import storage
import joblib
import pandas as pd


    
MLFLOW_URI = "https://mlflow.lewagon.co/"
### CHANGE THESE TO YOURS 👇
BUCKET_NAME = "lewagon-data-708-pavel"
STORAGE_LOCATION = "models/model.joblib"

class Trainer():
    def __init__(self, experiment_name, model_name='linreg', model=LinearRegression()):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(MLFLOW_URI)
        self.client = MlflowClient()
        self.experiment_id = None
        self.pipe = None
        self.model_name = model_name
        self.model = model
    
    def create_pipeline(self):
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        
        # create a time feature pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        
        # training pipeline
        self.pipe = Pipeline([
            ('preproc', preproc_pipe),
            (self.model_name, self.model)
        ])
    
    def evaluate_pipe(self, X_test, y_test):
        # train the pipelined model
        self.score = self.pipe.score(X_test, y_test)
        
        print(f"score with {self.model_name}", self.score)
        return self.pipe
    
    def train(self, x_train, x_test, y_train, y_test):
        # create a pipeline
        
        ## If model is not loaded from GCP, create a pipeline
        if self.pipe == None:
            self.create_pipeline()
            print(x_train.columns)
            self.pipe.fit(x_train, y_train)
        self.evaluate_pipe(x_test, y_test)
        
        ### MLflow tracking calls
        # self.create_mlflow_experiment()
        # self.create_mlflow_run()
        # self.mlflow_log_param('model', model_name)
        # if model_name == "KNN":
        #     self.mlflow_log_param('knn-neighbors', 10)
        # self.mlflow_log_metric('default score', self.score)
        return self.pipe
        
        
    ### MLFlow tracking functionss
    def mlflow_log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)
    
    def mlflow_log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)
    
    def create_mlflow_experiment(self):
        if self.experiment_id:
            return self.experiment_id
        self.experiment_id = self.client.create_experiment(self.experiment_name)
    
    def create_mlflow_run(self):
        run = self.client.create_run(self.experiment_id)
        self.run_id = run.info.run_id
        
    ### GCP Section functions
    def upload_model_to_gcp(self):
        
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')
    
    def save_model(self):
        joblib.dump(self.pipe, f'model_.joblib')
        
    def download_model(self, bucket=BUCKET_NAME):
        client = storage.Client().bucket(bucket)
        blob = client.blob(STORAGE_LOCATION)
        blob.download_to_filename('model.joblib')
        print("=> pipeline downloaded from storage")
        self.pipe = joblib.load('model.joblib')
    
        
    
if __name__ == '__main__':
    trainer = Trainer('[CN] Shanghai 708 TaxiFare')
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # get features and targets
    x, y = set_features_targets(df)
    # do a train/test split
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2)
    
    ### For training and evaluating model locally
    # models = [
    #     ('linreg', LinearRegression())
    #     ('KNN', KNeighborsRegressor(n_neighbors=10)),
    #     ('XGB', XGBRegressor())
    # ]
    # for model_name, model in models:
    #     pipe = trainer.train(x_train, x_test, y_train, y_test)  
        ## Pushing model to GCP
        # trainer.save_model()
        # trainer.upload_model_to_gcp()
        
    ### For evaluating model trained on GCP
    trainer.download_model()
    pipe = trainer.evaluate_pipe(x_test, y_test)