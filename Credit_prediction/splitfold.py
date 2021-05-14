from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv('application_train.csv')

    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.TARGET.values


    kf = model_selection.StratifiedKFold(n_splits=5)


    for fold,(train,val) in enumerate(kf.split(X= df,y = y)):
        df.loc[val,'kfold'] = fold
    
    df.to_csv("train_fold.csv")

    