import shap
import numpy as np
import pandas as pd
from catboost import Pool
import xgboost as xgb

N_PERMUTATIONS = 5


def compute_mse(model_name,model,X,y, transform_x = True, categorical_features = None):
    if transform_x:
        temp_x = X.copy()
        if model_name == 'xgboost':
            temp_x = xgb.DMatrix(temp_x)
        elif model_name == 'catboost':
            if categorical_features:
                temp_x = Pool(temp_x , cat_features = categorical_features)
            else:
                temp_x = Pool(temp_x)
        return np.mean(np.square(y - model.predict(temp_x)))
    return np.mean(np.square(y - model.predict(X)))


def permutation_feature_importance(model_name,model,X,y, categorical_features = None):
    results = {}
    mse = compute_mse(model_name, model,X,y, categorical_features = categorical_features)
    for col in X.columns:
        temp_x = X.copy()
        random_feature_mse = 0
        for i in N_PERMUTATIONS:
            temp_x[col] = np.random.permutation(temp_x[col])
            if model_name == 'xgboost':
                temp_x = xgb.DMatrix(temp_x)
            elif model_name == 'catboost':
                temp_x = Pool(temp_x , cat_features = categorical_features) if categorical_features else Pool(temp_x)
            random_feature_mse += compute_mse(model_name, model,temp_x,y, transform_x = False, categorical_features = categorical_features)
        results[col] = random_feature_mse - mse
    results = pd.Series(results)
    return results/results.sum()


def get_x1_shap_value(model, x, columns=None):
    if columns is None:
        columns = x.columns
    abs_shap_values = pd.DataFrame(shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(x),
                                   columns=columns).apply(np.abs)
    return (abs_shap_values.mean() / abs_shap_values.mean().sum())['x1']
