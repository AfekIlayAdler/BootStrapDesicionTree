import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import Pool, CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from experiments.detect_uninformative_feature_boston_house_prices.config import RESULTS_DIR, VAL_RATIO, N_ESTIMATORS, \
    MAX_DEPTH, LEARNING_RATE, MODELS_DIR, CATEGORY_COLUMN_NAME
from experiments.detect_uninformative_feature_boston_house_prices.fi import permutation_feature_importance, \
    get_shap_value
from experiments.detect_uninformative_feature_boston_house_prices.utils import create_x_y, create_one_hot_x_x_val, \
    create_mean_imputing_x_x_val, all_experiments, make_dirs
from gradient_boosting_trees.gradient_boosting_regressor import CartGradientBoostingRegressorKfold, \
    CartGradientBoostingRegressor


def worker(model_name, variant, exp_number, category_size, path):
    exp_name = F"{model_name}_{variant}_exp_{exp_number}_category_size_{category_size}.csv"
    dir = RESULTS_DIR / model_name
    exp_results_path = dir / exp_name
    if exp_results_path.exists():
        return
    np.random.seed(exp_number)
    X, y = create_x_y(category_size,path)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=VAL_RATIO, random_state=42)
    if variant == 'one_hot':
        X_train, X_test = create_one_hot_x_x_val(X_train, X_test)
    if variant == 'mean_imputing':
        X_train, X_test = create_mean_imputing_x_x_val(X_train, y_train, X_test)
    if model_name == 'ours':
        model = CartGradientBoostingRegressorKfold if variant == 'Kfold' else CartGradientBoostingRegressor
        reg = model(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                    learning_rate=LEARNING_RATE, min_samples_leaf=5)
        reg.fit(X_train, y_train)
    elif model_name == 'sklearn':
        reg = GradientBoostingRegressor(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                        learning_rate=LEARNING_RATE)
        reg.fit(X_train, y_train)
    elif model_name == 'xgboost':
        dtrain = xgb.DMatrix(X_train, label=y_train)
        param = {'max_depth': MAX_DEPTH, 'eta': LEARNING_RATE, 'objective': 'reg:squarederror'}
        reg = xgb.train(param, dtrain, N_ESTIMATORS)
    else:  # model_name == 'catboost
        if variant == 'mean_imputing':
            train_pool = Pool(X_train, y_train)
            val_pool = Pool(X_test, y_test)
            cat_feature = None
        else:
            cat_feature = [3, 8, 13]
            train_pool = Pool(X_train, y_train, cat_features=cat_feature)
            val_pool = Pool(X_test, y_test, cat_features=cat_feature)
        reg = CatBoostRegressor(iterations=N_ESTIMATORS,
                                depth=MAX_DEPTH,
                                learning_rate=LEARNING_RATE,
                                loss_function='RMSE', logging_level='Silent')
        reg.fit(train_pool)
    results_df = pd.DataFrame(
        columns=['model', 'categories', 'exp', 'gain', 'permutation_train', 'permutation_test', 'shap_train',
                 'shap_test'])
    results_df.loc[0, ['model', 'categories', 'exp']] = [F"{model_name}_{variant}", category_size, exp_number]
    # fi gain
    if model_name == 'ours':
        fi_gain = pd.Series(reg.compute_feature_importance(method='gain'))
    elif model_name == 'sklearn':
        fi_gain = pd.Series(reg.feature_importances_, index=X_train.columns)
    elif model_name == 'xgboost':
        fi_gain = pd.Series(reg.get_score(importance_type='gain'))
    else:  # model_name == 'catboost'
        fi_gain = pd.Series(reg.feature_importances_, index=reg.feature_names_)
    fi_gain /= fi_gain.sum()
    results_df.loc[0, 'gain'] = np.sum([fi_gain[col] for col in fi_gain.index if not col.startswith(CATEGORY_COLUMN_NAME)])
    if model_name == 'catboost':
        fi_permutation_train = permutation_feature_importance(model_name, reg, X_train, y_train,
                                                              categorical_features=cat_feature)
        fi_permutation_test = permutation_feature_importance(model_name, reg, X_test, y_test,
                                                             categorical_features=cat_feature)
    else:
        fi_permutation_train = permutation_feature_importance(model_name, reg, X_train, y_train)
        fi_permutation_test = permutation_feature_importance(model_name, reg, X_test, y_test)

    results_df.loc[0, 'permutation_train'] = fi_permutation_train
    results_df.loc[0, 'permutation_test'] = fi_permutation_test
    if model_name in ['sklearn', 'xgboost']:
        fi_shap_train = get_shap_value(reg, X_train)
        fi_shap_test = get_shap_value(reg, X_test)
    elif model_name == 'catboost':
        fi_shap_train = get_shap_value(reg, train_pool, columns=X_train.columns),
        fi_shap_test = get_shap_value(reg, val_pool, columns=X_test.columns)
    else:
        fi_shap_train, fi_shap_test = None, None
    results_df.loc[0, ['shap_train', 'shap_test']] = [fi_shap_train, fi_shap_test]
    results_df.to_csv(exp_results_path)


dtypes = {'CRIM': 'float64',
          'ZN': 'float64',
          'INDUS': 'float64',
          'CHAS': 'category',
          'NOX': 'float64',
          'RM': 'float64',
          'AGE': 'float64',
          'DIS': 'float64',
          'RAD': 'category',
          'TAX': 'float64',
          'PTRATIO': 'float64',
          'B': 'float64',
          'LSTAT': 'float64',
          'y': 'float64'}

sorted_index = ['AGE', 'B', 'CHAS', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'NOX', 'PTRATIO',
                'RAD', 'RANDOM_CATEGORY', 'RM', 'TAX', 'ZN']

if __name__ == '__main__':
    N_PROCESS = 4
    input_path = Path.cwd().parent.parent / "Datasets/boston_house_prices/boston_house_prices.csv"
    df = pd.read_csv(input_path)
    make_dirs([MODELS_DIR, RESULTS_DIR])
    models = {
        'xgboost': ['one_hot', 'mean_imputing'],
        'catboost': ['vanilla', 'mean_imputing'],
        'sklearn': ['one_hot', 'mean_imputing'],
        'ours': ['Kfold', 'CartVanilla']}

    n_experiments = len(all_experiments())
    print(f"n experimets for each model: {n_experiments}")
    for model_name, model_variants in models.items():
        print(f'Working on experiment : {model_name}')
        args = []
        make_dirs([RESULTS_DIR / model_name])
        for exp_number, category_size in all_experiments():
            for variant in model_variants:
                # worker(model_name, variant, exp_number, category_size, input_path)
                args.append((model_name, variant, exp_number, category_size,input_path))

        with multiprocessing.Pool(N_PROCESS) as process_pool:
            process_pool.starmap(worker, args)
        # with concurrent.futures.ThreadPoolExecutor(4) as executor:
        #     results = list(tqdm(executor.map(lambda x: worker(*x), args), total=len(args)))
