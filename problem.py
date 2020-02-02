import os
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

problem_title = 'Prediction of the gravity of an accident'
_target_column_name = 'grav' 
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow



class FAN(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor']):
        super(FAN, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow = FAN()

# define the score (specific score for the FAN problem)
class FAN_error(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='fan error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        balanced_score = balanced_accuracy_score(y_true, y_pred.round())
        
        return balanced_score

score_types = [
    FAN_error(name='Accuracy error', precision=2),
]

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X,y, groups=X.index)

def _read_data(path, f_name1,f_name2,f_name3,f_name4):
    data1 = pd.read_csv(os.path.join(path, 'data', f_name1), low_memory=False)
    data2 = pd.read_csv(os.path.join(path, 'data', f_name2), low_memory=False)
    data3 = pd.read_csv(os.path.join(path, 'data', f_name3), low_memory=False)
    data4 = pd.read_csv(os.path.join(path, 'data', f_name4), low_memory=False, encoding='latin-1')
    dataMerge = pd.merge(data1,data2, on='Num_Acc', how='left')
    dataMerge1 =  pd.merge(dataMerge,data3, on='Num_Acc', how='left')
    dataMerge2 = pd.merge(dataMerge1,data4, on='Num_Acc', how='left')
    encoder = LabelEncoder()
    y_array = dataMerge2[_target_column_name].values
    y_array = encoder.fit_transform(y_array)
    X_df = dataMerge2.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name1 = 'usagers-2017.csv'
    f_name2 = 'lieux-2017.csv'
    f_name3 = 'vehicules-2017.csv'
    f_name4 = 'caracteristiques-2017.csv'
    return _read_data(path, f_name1,f_name2,f_name3,f_name4)

def get_test_data(path='.'):
    f_name1 = 'usagers-2018.csv'
    f_name2 = 'lieux-2018.csv'
    f_name3 = 'vehicules-2018.csv'
    f_name4 = 'caracteristiques-2018.csv'
    return _read_data(path, f_name1,f_name2,f_name3,f_name4)