import multiprocessing

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from experiments.interaction.config import RESULTS_DIR, EXP_NAME, MODELS_DIR, MAX_DEPTH, N_ESTIMATORS, \
    LEARNING_RATE
from experiments.interaction.config import VAL_RATIO
from experiments.interaction.fi import permutation_feature_importance, get_x1_shap_value
from experiments.interaction.utils import create_x_y, create_one_hot_x_x_val, create_mean_imputing_x_x_val
from experiments.interaction.utils import make_dirs, all_experiments, \
    n_experiments


def xgboost_worker(exp_number, category_size, a, predictor_name, _):
    model_type= 'xgboost'
    np.random.seed(exp_number)
    exp_name = F"{EXP_NAME}_{model_type}_{predictor_name}_exp_{exp_number}_category_size_{category_size}_a_{a}"
    exp_results_path = RESULTS_DIR / F"{exp_name}.csv"
    if exp_results_path.exists(): return
    X, y = create_x_y(category_size, float(a))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO, random_state=42)
    if predictor_name == 'one_hot':
        X_train, X_test = create_one_hot_x_x_val(X_train, X_test)
    elif predictor_name == 'mean_imputing':
        X_train, X_test = create_mean_imputing_x_x_val(X_train, y_train, X_test)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    param = {'max_depth': MAX_DEPTH - 1, 'eta': LEARNING_RATE, 'objective': 'reg:squarederror'}
    bst = xgb.train(param, dtrain, N_ESTIMATORS)
    fi_gain = pd.Series(bst.get_score(importance_type='gain'))
    fi_gain /= fi_gain.sum()
    fi_permutation_train = permutation_feature_importance(model_type, bst, X_train, y_train)
    fi_permutation_test = permutation_feature_importance(model_type, bst, X_test, y_test)
    temp_results = pd.DataFrame([[F"{model_type}_{predictor_name}", category_size, a, fi_gain['x1'], fi_permutation_train['x1'], fi_permutation_test['x1'], get_x1_shap_value(bst, X_train),get_x1_shap_value(bst, X_test)]])
    temp_results.to_csv(exp_results_path)


if __name__ == '__main__':
    make_dirs([MODELS_DIR, RESULTS_DIR])
    args = []
    for exp_number, category_size, a in tqdm(all_experiments(), total=n_experiments):
        np.random.seed(exp_number)
        for predictor_name, predictor in {'one_hot': GradientBoostingRegressor,
                                  'mean_imputing': GradientBoostingRegressor}.items():
            args.append((exp_number, category_size, a, predictor_name, predictor))
    print(F"# of experiments is {len(args)}")
    with multiprocessing.Pool(4) as process_pool:
        process_pool.starmap(xgboost_worker, args)
