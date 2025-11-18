import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class ReleaseDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date='2025-03-31'):#database is from march 2025
        self.reference_date=pd.to_datetime(reference_date) 
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.to_datetime(X, errors='coerce')
        X_df = pd.DataFrame({
            'release_year': X.dt.year,
            'release_month': X.dt.month,
            'days_since_release': (self.reference_date - X).dt.days
        })
        return X_df
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['release_year', 'release_month', 'days_since_release'])


class MLBTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False, min_freq=0.01):
        self.sparse_output = sparse_output
        self.min_freq = min_freq
        self.mlb=None
        self.allowed_labels= None
    
    def fit(self, X, y=None):
        X = pd.Series(X)
        all_labels = X.explode().value_counts(normalize=True)
        self.allowed_labels=list(all_labels[all_labels >= self.min_freq].index)

        #filter the list
        X_filtered = X.apply(self._filter_labels)

        #now we can fit the MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer(sparse_output=self.sparse_output)
        self.mlb.fit(X_filtered)
        return self

    def _filter_labels(self, lst):
        filtered = [x for x in lst if x in self.allowed_labels] or ['Other']
        return filtered
    
    def transform(self, X):
        X = pd.Series(X)
        X_filtered = X.apply(self._filter_labels)
        return self.mlb.transform(X_filtered)

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_


class FreqEnc(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.freq = np.log1p(X.explode().value_counts())
        self.input_col = X.name if hasattr(X, 'name') else None #takes the name of the variable if it has an attribute called 'name'
        return self
    
    def transform(self, X):
        output = X.apply(lambda lst: max([self.freq.get(x,0) for x in lst]))
        return output.to_numpy().reshape(-1,1) #reshaping is necessary to get it to be a column
    
    def get_feature_names_out(self, input_features=None):
        return np.array([self.input_col or 'freq_encoded']) #returns the feature name or 'freq_encoded' if name is None