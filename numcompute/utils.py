import numpy as np

class BaseEstimator:
  
    def __init__(self):
        self._is_fitted = False

    def fit(self, X, y=None):

        # Validation logic goes here
        
        self._is_fitted = True # ensures .fit() has been called prior to other action
        return self

class BaseTransformer(BaseEstimator):

    
    def transform(self, X):

        if not self._is_fitted:
            raise RuntimeError("Estimator must be fitted before transforming.")
        return X

    def fit_transform(self, X, y=None):
        """Fits and transforms in a single vectorized call."""
        return self.fit(X, y).transform(X)
