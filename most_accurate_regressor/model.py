from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import pandas as pd

class Model:
    def __init__(self):
        self.pipe = Pipeline([
         ("columnDropper", columnDropperTransformer([6])),('xgb', XGBRegressor(max_depth=10, n_estimators=100, learning_rate=0.18733))])

    def fit(self, X, y):
        self.model = self.pipe.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        X = pd.DataFrame(X)
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self 