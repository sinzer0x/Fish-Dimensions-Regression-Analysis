from sklearn.base import BaseEstimator, TransformerMixin


class VolumeAdder(BaseEstimator, TransformerMixin):
    """Добавляет производный геометрический признак V_proxy = Length3 * Height * Width."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["V_proxy"] = X_copy["Length3"] * X_copy["Height"] * X_copy["Width"]
        return X_copy