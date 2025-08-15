# -*- coding: utf-8 -*-
# acs_core/models/reactive.py
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def fit_reactive_head(
    X_fast: pd.DataFrame,
    y: pd.Series,
    time_index: pd.DatetimeIndex,
    C=3.0,
    lambda_decay=0.03,
):
    """对近期样本加权的“反应头”"""
    dt_days = (time_index.max() - time_index).days.values
    sw = np.exp(-lambda_decay * dt_days)
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=200),
            ),
        ]
    )
    mask = X_fast.notna().any(axis=1) & y.notna()
    pipe.fit(X_fast.loc[mask].values, y.loc[mask].values, clf__sample_weight=sw[mask])
    proba = pd.Series(np.nan, index=time_index)
    proba.loc[mask.index] = pipe.predict_proba(X_fast.values)[:, 1]
    return proba
