class PassthroughScaler:
    """
    A dummy scaler that mimics the fit/transform interface of scikit-learn scalers
    but does not modify the data. This is used to maintain a consistent pipeline
    for fingerprint data, which should not be scaled.
    """
    def fit(self, X, y=None):
        # Does nothing
        return self

    def transform(self, X, y=None):
        # Returns the data unmodified
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)