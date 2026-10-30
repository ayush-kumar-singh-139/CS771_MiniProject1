"""Our final Part-2 model: average of a scaled ridge and an ExtraTrees forest, both predicting the
change from persistence (target - reading at issue time). The hyperparameters are chosen on the
validation set in mini_project_1.ipynb (section 2.5) and passed in; the defaults below are those
chosen values (the notebook asserts that they agree).
"""
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import ALL_GROUPS, predict_residual_ensemble

# the ablation (notebook section 2.6) showed the 'trend' group does not earn its place
FINAL_GROUPS = tuple(g for g in ALL_GROUPS if g != "trend")
RIDGE_ALPHA = 1.0
ET_PARAMS = dict(n_estimators=100, min_samples_leaf=50, max_features=0.3)


def make_final_models(alpha=RIDGE_ALPHA, et_params=None, seed=C.RANDOM_STATE):
    """Fresh (unfitted) members. The StandardScaler lives inside the ridge pipeline, so it is fitted
    on exactly the rows passed to .fit() - the training rows - and never on validation rows."""
    et_params = ET_PARAMS if et_params is None else et_params
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    forest = ExtraTreesRegressor(**et_params, n_jobs=-1, random_state=seed)
    return [ridge, forest]


def fit_final_models(X, y, persistence, alpha=RIDGE_ALPHA, et_params=None, seed=C.RANDOM_STATE):
    """Fit both members on the residual target y - persistence."""
    models = make_final_models(alpha, et_params, seed)
    for m in models:
        m.fit(X, y - persistence)
    return models


def predict_final(models, X, persistence):
    return predict_residual_ensemble(models, X, persistence)
