"""Pure ML helpers (no Django imports) so joblib can load the saved models anywhere.

Edit the feature lists below if you retrain on your own department's columns.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Final grade G3 is on a 0-20 scale. Below this mark a student counts as "at risk".
PASS_MARK = 10

NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2",
]
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Early-warning variant: no period grades, so it works before any assessment exists.
EARLY_FEATURES = [f for f in ALL_FEATURES if f not in ("G1", "G2")]


def read_table(source):
    """Read a CSV from a path or file-like object. Auto-detects ';' vs ',' separators."""
    if hasattr(source, "read"):
        raw = source.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    else:
        text = Path(source).read_text(encoding="utf-8-sig")
    first_line = text.splitlines()[0] if text.strip() else ""
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric column to [Q1 - k*IQR, Q3 + k*IQR] learned on the training data.

    Columns with IQR == 0 (e.g. 'failures', mostly 0) are left alone, otherwise the
    clipping would flatten the whole feature.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = np.where(iqr > 0, q1 - self.factor * iqr, -np.inf)
        self.upper_ = np.where(iqr > 0, q3 + self.factor * iqr, np.inf)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_pipeline(estimator, features):
    """Preprocessing + model in ONE pipeline so nothing is fitted outside the CV folds."""
    numeric = [f for f in features if f in NUMERIC_FEATURES]
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clip", IQRClipper()),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([("prep", prep), ("model", estimator)])
