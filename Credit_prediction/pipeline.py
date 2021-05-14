from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import preprocessors as pp


import config


credit_risk_pipe = Pipeline(
    [
        ('numerical_inputer',
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),
        
        ('categorical_imputer',
            pp.CategoricalImputation(variables=config.CATEGORICAL_VARS_WITH_NA)),
         
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
               variables=config.CATEGORICAL_VARS)),
         
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
         
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),
         
        ('scaler', MinMaxScaler()),
        ('Linear_model', RandomForestClassifier())
    ]
)
