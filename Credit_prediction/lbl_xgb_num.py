import pandas as pd
#import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics


def run(fold):
    df = pd.read_csv('train_fold.csv')
    ##get numerical variables
    num_cols = [col for col in df.columns if df[col].dtypes!= 'O']

    #print(num_cols)
    
    ## get the all the features in the data
    features = [feat for feat in df.columns if feat not in ('kfold','TARGET','SK_ID_CURR')]

 
    ## fill in missing numerical features
    ## first iterate thr all features, then for cat feats in all features fill na with string 'missing'
    for col in num_cols:
        df.loc[:,col] = df[col].fillna(0)

    for col in features:
        if col not in num_cols:
            df.loc[:,col] = df[col].astype(str).fillna('missing')
    
    ### encode categorical variable
    for col in features:
        if col not in num_cols:
            label_enc = preprocessing.LabelEncoder()
            label_enc.fit(df[col])

            df.loc[:,col] = label_enc.transform(df[col])

    df_train = df[df.kfold != fold].reset_index(drop = True)

    df_valid  =df[df.kfold == fold].reset_index(drop = True)

    x_train = df_train[features].values

    x_valid = df_valid[features].values

    model =  RandomForestClassifier()

    model.fit(x_train,df_train.TARGET.values)

    valid_preds = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(df_valid.TARGET.values,valid_preds)

    print(f'Fold = {fold},AUC = {auc}')

if __name__ == '__main__':
    for fold in range(5):
        run(fold)

