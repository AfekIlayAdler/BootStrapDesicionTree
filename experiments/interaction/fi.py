import shap
import numpy as np
import pandas as pd


def compute_mse(model_name, model, X, y, transform_x=True, categorical_features=None,):
    return np.mean(np.square(y - model.predict(X)))


cat_features = [0]


def permutation_feature_importance(model_name, model, X, y, categorical_features=None):
    results = {}
    mse = compute_mse(model_name, model, X, y, categorical_features=categorical_features)
    for col in X.columns:
        temp_x = X.copy()
        temp_x[col] = np.random.permutation(temp_x[col])
        new_mse = compute_mse(model_name, model, temp_x, y, transform_x=False,
                              categorical_features=categorical_features)
        results[col] = new_mse - mse
    results = pd.Series(results)
    return results / results.sum()


def get_x1_shap_value(model, x, columns=None):
    if columns is None:
        columns = x.columns
    abs_shap_values = pd.DataFrame(shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(x),
                                   columns=columns).apply(np.abs)
    return (abs_shap_values.mean() / abs_shap_values.mean().sum())['x1']
