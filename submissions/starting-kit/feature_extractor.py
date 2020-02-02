import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


class FeatureExtractor(object):
    def __init__(self):
            pass

    def fit(self, X_df, y_array):
            pass

    def transform(self, X_df):
        
        def to_numeric(X):
            tmp = X.apply(lambda x: pd.factorize(x)[0])
            return tmp

        to_numeric_transformer = FunctionTransformer(to_numeric, validate=False)

        def infer_num(X):
            return X.apply(pd.to_numeric, errors='coerce')

        infer_num_transformer = FunctionTransformer(infer_num, validate=False)
        
        cat_cols = np.array(X_df.select_dtypes(exclude=["number","bool_"]).columns)
        num_cols = np.array(X_df.select_dtypes(exclude=["object"]).columns)
        to_drop_cols = ['voie','v1','v2','pr','pr1','lartpc','larrout','locp','actp','etatp']
        
        preprocessor_comp = ColumnTransformer(
                transformers=[
            ('num', make_pipeline(infer_num_transformer, SimpleImputer(strategy='median')), num_cols),
            ('cat', make_pipeline(to_numeric_transformer, SimpleImputer(strategy="median")), cat_cols),
            ('drop cols', 'drop', to_drop_cols)
        ])

        X_array = preprocessor_comp.fit_transform(X_df) 
        
        return X_array