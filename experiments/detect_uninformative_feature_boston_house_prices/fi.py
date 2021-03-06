import shap
import numpy as np
import pandas as pd
from catboost import Pool
import xgboost as xgb

from experiments.detect_uninformative_feature_simulated_data.config import N_PERMUTATIONS, CATEGORY_COLUMN_NAME


def compute_mse(model_name, model, X, y, transform_x=True, categorical_features=None):
    if transform_x:
        temp_x = X.copy()
        if model_name == 'xgboost':
            temp_x = xgb.DMatrix(temp_x)
        elif model_name == 'catboost':
            if categorical_features:
                temp_x = Pool(temp_x, cat_features=categorical_features)
            else:
                temp_x = Pool(temp_x)
        return np.mean(np.square(y - model.predict(temp_x)))
    return np.mean(np.square(y - model.predict(X)))


def permutation_feature_importance(model_name, model, X, y, categorical_features=None):
    results = {}
    mse = compute_mse(model_name, model, X, y, categorical_features=categorical_features)
    for col in X.columns:
        permutated_x = X.copy()
        random_feature_mse = []
        for i in range(N_PERMUTATIONS):
            permutated_x[col] = np.random.permutation(permutated_x[col])
            if model_name == 'xgboost':
                temp_x = xgb.DMatrix(permutated_x)
            elif model_name == 'catboost':
                temp_x = Pool(permutated_x, cat_features=categorical_features) if categorical_features else Pool(
                    permutated_x)
            else:
                temp_x = permutated_x
            random_feature_mse.append(
                compute_mse(model_name, model, temp_x, y, transform_x=False, categorical_features=categorical_features))
        results[col] = np.mean(np.array(random_feature_mse)) - mse
    results = pd.Series(results)
    fi = results/results.sum()
    return np.sum([fi[col] for col in X.columns if not col.startswith(CATEGORY_COLUMN_NAME)])


def get_shap_value(model, x, columns= None) -> object:
    if columns is None:
        columns = x.columns
    abs_shap_values = pd.DataFrame(shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(x),
                                   columns=columns).apply(np.abs)
    shap_values = abs_shap_values.mean() / abs_shap_values.mean().sum()
    return np.sum([shap_values[col] for col in shap_values.index if not col.startswith(CATEGORY_COLUMN_NAME)])
