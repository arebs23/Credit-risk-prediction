from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np

class LogTransform(BaseEstimator,TransformerMixin):
    def __init__(self,variables = None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y = None):
        return self
    
    def transform(self,X):
        X = X.copy()

        for features in self.variables:
            X[features] = np.log(X[features])

        return X


