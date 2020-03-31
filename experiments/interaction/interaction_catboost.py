import multiprocessing

import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from experiments.interaction.config import RESULTS_DIR, EXP_NAME, MODELS_DIR, MAX_DEPTH, N_ESTIMATORS, \
    LEARNING_RATE
from experiments.interaction.config import VAL_RATIO
from experiments.interaction.fi import permutation_feature_importance, get_x1_shap_value
from experiments.interaction.utils import create_x_y, create_mean_imputing_x_x_val
from experiments.interaction.utils import make_dirs, all_experiments, \
    n_experiments


def catboost_worker(exp_number, category_size, a, predictor_name):
    model_type = 'catboost'
    np.random.seed(exp_number)
    exp_name = F"{EXP_NAME}_{model_type}_{predictor_name}_exp_{exp_number}_category_size_{category_size}_a_{a}"
    exp_results_path = RESULTS_DIR / F"{exp_name}.csv"
    X, y = create_x_y(category_size, float(a))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO, random_state=42)
    if predictor_name == 'mean_imputing':
        X_train, X_test = create_mean_imputing_x_x_val(X_train, y_train, X_test)
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_test, y_test)
    else:
        train_pool = Pool(X_train, y_train, cat_features=[0])
        val_pool = Pool(X_test, y_test, cat_features=[0])

    model = CatBoostRegressor(iterations=N_ESTIMATORS,
                              depth=MAX_DEPTH,
                              learning_rate=LEARNING_RATE,
                              loss_function='RMSE', logging_level='Silent')
    model.fit(train_pool)

    fi_gain = pd.Series(model.feature_importances_, index=model.feature_names_)
    fi_gain /= fi_gain.sum()
    cat_feature = [0] if predictor_name == 'vanilla' else None
    fi_permutation_train = permutation_feature_importance(model_type, model, X_train, y_train,
                                                          categorical_features=cat_feature)
    fi_permutation_test = permutation_feature_importance(model_type, model, X_test, y_test,
                                                         categorical_features=cat_feature)
    temp_results = pd.DataFrame([[F"{model_type}_{predictor_name}", category_size, a, fi_gain['x1'],
                                  fi_permutation_train['x1'], fi_permutation_test['x1'],
                                  get_x1_shap_value(model, train_pool, columns=X_train.columns),
                                  get_x1_shap_value(model, val_pool, columns=X_test.columns)]])
    temp_results.to_csv(exp_results_path)


if __name__ == '__main__':
    make_dirs([MODELS_DIR, RESULTS_DIR])
    args = []
    for exp_number, category_size, a in tqdm(all_experiments(), total=n_experiments):
        np.random.seed(exp_number)
        for predictor_name in ["vanilla", "mean_imputing"]:
            args.append((exp_number, category_size, a, predictor_name))
    print(F"# of experiments is {len(args)}")
    with multiprocessing.Pool(4) as process_pool:
        process_pool.starmap(catboost_worker, args)
