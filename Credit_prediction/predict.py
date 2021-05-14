import pandas as pd

import joblib
import config


def make_prediction(input_data):
    
    _credit_risk_ = joblib.load(filename=config.PIPELINE_NAME)
    
    results = _credit_risk_.predict_proba(input_data)[:,1]

    return results
   
if __name__ == '__main__':
    
    # test pipeline
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    data = pd.read_csv(config.TRAINING_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.1,
        random_state=0)
    
    pred = make_prediction(X_test)
    
    auc = metrics.roc_auc_score(y_test,pred)
    print(f'auc_score = {auc}')
