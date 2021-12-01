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
import memoized_property

MLFLOW_URI = "https://mlflow.lewagon.co/"

class Trainer():
    def __init__(self, experiment_name, **kwargs):
        pass
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(MLFLOW_URI)
        self.client = MlflowClient()
        self.experiment_id = None
    
    def create_pipeline(self, model_name, model):
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
            (model_name, model)
        ])
    
    def evaluate_pipe(self, X_train, X_test, y_train, y_test, model_name):
        # train the pipelined model
        self.pipe.fit(X_train, y_train)

        self.score = self.pipe.score(X_test, y_test)
        
        print(f"score with {model_name}", self.score)
    
    def train(self, x_train, x_test, y_train, y_test, model_name, model):
        # create a pipeline
        self.create_pipeline(model_name, model)
        self.evaluate_pipe(x_train, x_test, y_train, y_test, model_name)
        
        ## MLflow section calls
        self.create_mlflow_experiment()
        self.create_mlflow_run()
        self.mlflow_log_param('model', model_name)
        if model_name == "KNN":
            self.mlflow_log_param('knn-neighbors', 10)
        self.mlflow_log_metric('default score', self.score)
        
        
    ### MLFlow section
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
        
        
    
if __name__ == '__main__':
    trainer = Trainer('[CN] Shanghai 708 TaxiFare')
    # get data
    df = get_data('./raw_data/train_10k.csv')
    # clean data
    df = clean_data(df)
    # get features and targets
    x, y = set_features_targets(df)
    # do a train/test split
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2)
    models = [
        ('linreg', LinearRegression()),
        ('KNN', KNeighborsRegressor(n_neighbors=10)),
        ('XGB', XGBRegressor())
    ]
    for model_name, model in models:
        trainer.train(x_train, x_test, y_train, y_test, model_name, model)
    
    
    
    
    
    
    
    
    
    