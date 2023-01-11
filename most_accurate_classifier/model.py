from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
import pandas as pd
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

class Model:
    def __init__(self):
        self.model = XGBClassifier()
        
        #
        self.pipe = Pipeline([('oversample', RandomOverSampler()),('scaler',RobustScaler()),
        ('select',SelectKBest(k=12)),
        ('xgb', XGBClassifier(subsample=0.8999999999999999,n_estimators=1000,max_depth= 20,learning_rate =0.01,colsample_bytree= 0.7,colsample_bylevel= 0.5))])

    def fit(self, X, y):
        self.model = self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)
