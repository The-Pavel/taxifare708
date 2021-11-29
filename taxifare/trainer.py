from taxifare.data import get_data, clean_data, holdout, set_features_targets

class Trainer():
    def __init__(self):
        pass
    
    
    
if __name__ == '__main__':
    trainer = Trainer()
    # get data
    df = get_data('./raw_data/train_10k.csv')
    print(df.shape)
    # clean data
    df = clean_data(df)
    print(df.shape)
    # get features and targets
    x, y = set_features_targets(df)
    # do a train/test split
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2)
    # create a pipeline
    
    
    
    
    
    
    
    