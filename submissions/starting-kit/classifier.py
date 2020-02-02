from sklearn.base import BaseEstimator
from imblearn.ensemble import BalancedBaggingClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        self.reg = BalancedBaggingClassifier(n_estimators=50, random_state=42)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        pred = self.reg.predict(X)
        return pred
    
    def predict_proba(self, X):
        pred = self.reg.predict_proba(X)
        return pred