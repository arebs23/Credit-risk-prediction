import copy
import pandas as pd
import argparse

from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


def mean_target_encoding(data):
    df = copy.deepcopy(data)

    nums_cols = [col for col in data.columns if data[col].dtypes !='O']

    #features = [feat for feat in df.columns if feat not in ('TARGET','kfold','SK_ID_CURR')]

    cat_features =  [feat for feat in df.columns if feat not in ('TARGET','kfold','SK_ID_CURR','num_cols')]

    for col in nums_cols:
        mapping_num = df[col].mean()
        df.loc[:,col] = df[col].fillna(mapping_num)
        
    
    for col in cat_features:
        df.loc[:,col] = df[col].astype(str).fillna('missing')

    for col in cat_features:
        mapping_dict = df.groupby(col)['TARGET'].mean().to_dict()
        df.loc[:,col] = df[col].map(mapping_dict)

    return df

def run(df,fold):

    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    features = [feat for feat in df.columns if feat not in ('TARGET','kfold','SK_ID_CURR')]

    X_train = df_train[features].values
    X_valid = df_valid[features].values

    model  = RandomForestClassifier()

    model.fit(X_train,df_train.TARGET.values)

    valid_preds = model.predict_proba(X_valid)[:,1]
   
    auc = metrics.roc_auc_score(df_valid.TARGET.values,valid_preds)

    print(f'Fold = {fold}, auc = {auc}')


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()

   # parser.add_argument("--fold",type = int)

   # parser.add_argument("--dataframe",type = pd.DataFrame)

   # args = parser.parse_args()
  

    df = pd.read_csv('train_fold.csv')

    df = mean_target_encoding(df)

    for fold in range(5):
        run(df,fold)



