from taxifare.data import get_data, clean_data, holdout, set_features_targets
from taxifare.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

class Trainer():
    def __init__(self):
        pass
    
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

        score = self.pipe.score(X_test, y_test)
        
        print(f"score with {model_name}", score)
        

    
    def train(self, x_train, x_test, y_train, y_test, model_name, model):
        # create a pipeline
        self.create_pipeline(model_name, model)
        self.evaluate_pipe(x_train, x_test, y_train, y_test, model_name)
        
        
    
if __name__ == '__main__':
    trainer = Trainer()
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
    
    
    
    
    
    
    
    
    
    