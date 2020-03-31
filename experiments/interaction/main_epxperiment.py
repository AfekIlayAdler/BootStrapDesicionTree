import multiprocessing

from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import xgboost as xgb

from experiments.detect_uninformative_feature_simulated_data.utils import all_experiments, create_x_y, make_dirs, \
    create_one_hot_x_x_val, create_mean_imputing_x_x_val
from experiments.interaction.config import RESULTS_DIR, EXP_NAME, MODELS_DIR, MAX_DEPTH, N_ESTIMATORS, \
    LEARNING_RATE, VAL_RATIO, Y_COL_NAME
from experiments.interaction.fi import permutation_feature_importance, get_x1_shap_value
from gradient_boosting_trees.gradient_boosting_regressor import CartGradientBoostingRegressorKfold, \
    CartGradientBoostingRegressor, GradientBoostingRegressor


def worker(model_name, variant, exp_number, category_size, a):
    exp_name = F"{EXP_NAME}_{model_name}_{variant}_exp_{exp_number}_category_size_{category_size}_a_{a}"
    exp_results_path = RESULTS_DIR / F"{model_name}/{exp_name}.csv"
    if exp_results_path.exists():
        return
    np.random.seed(exp_number)
    X, y = create_x_y(category_size, a)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=VAL_RATIO, random_state=42)
    if variant == 'one_hot':
        X_train, X_test = create_one_hot_x_x_val(X_train, X_test)
    if variant == 'mean_imputing':
        X_train, X_test = create_mean_imputing_x_x_val(X_train, y_train, X_test)
    if model_name == 'ours':
        X_train[Y_COL_NAME] = y_train
        model = CartGradientBoostingRegressorKfold if variant == 'kfold' else CartGradientBoostingRegressor
        reg = model(Y_COL_NAME, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                    learning_rate=LEARNING_RATE, min_samples_leaf=5)
        reg.fit(X)
    elif model_name == 'sklearn':
        reg = GradientBoostingRegressor(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                                        learning_rate=LEARNING_RATE)
        reg.fit(X_train, y_train)
    elif model_name == 'xgboost':
        dtrain = xgb.DMatrix(X_train, label=y_train)
        param = {'max_depth': MAX_DEPTH - 1, 'eta': LEARNING_RATE, 'objective': 'reg:squarederror'}
        reg = xgb.train(param, dtrain, N_ESTIMATORS)
    else:  # model_name == 'catboost
        if variant == 'mean_imputing':
            train_pool = Pool(X_train, y_train)
            val_pool = Pool(X_test, y_test)
            cat_feature = None
        else:
            cat_feature = [0]
            train_pool = Pool(X_train, y_train, cat_features=cat_feature)
            val_pool = Pool(X_test, y_test, cat_features=cat_feature)
        reg = CatBoostRegressor(iterations=N_ESTIMATORS,
                                depth=MAX_DEPTH,
                                learning_rate=LEARNING_RATE,
                                loss_function='RMSE', logging_level='Silent')
        reg.fit(train_pool)
    results_df = pd.DataFrame(columns=['gain', 'permutation_train', 'permutation_test', 'shap_train', 'shap_test'])
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
    results_df.loc[0, 'gain'] = fi_gain['x1']
    if model_name == 'catboost':
        fi_permutation_train = permutation_feature_importance(model_name, reg, X_train, y_train,
                                                              categorical_features=cat_feature)
        fi_permutation_test = permutation_feature_importance(model_name, reg, X_test, y_test,
                                                             categorical_features=cat_feature)
    else:
        fi_permutation_train = permutation_feature_importance(model_name, reg, X_train, y_train)
        fi_permutation_test = permutation_feature_importance(model_name, reg, X_test, y_test)

    results_df.loc[0, 'permutation_train'] = fi_permutation_train['x1']
    results_df.loc[0, 'permutation_test'] = fi_permutation_test['x1']
    if model_name in ['sklearn', 'xgboost']:
        fi_shap_train = get_x1_shap_value(reg, X_train)
        fi_shap_test = get_x1_shap_value(reg, X_test)
    elif model_name == 'catboost':
        fi_shap_train = get_x1_shap_value(model, train_pool, columns=X_train.columns),
        fi_shap_test = get_x1_shap_value(model, val_pool, columns=X_test.columns)
    else:
        fi_shap_train, fi_shap_test = None, None
    results_df.loc[0, ['shap_train', 'shap_test']] = fi_shap_train, fi_shap_test
    results_df.to_csv(exp_results_path)


if __name__ == '__main__':
    N_PROCESS = 4
    make_dirs([MODELS_DIR, RESULTS_DIR])
    args = []
    models = {'xgboost': ['one_hot', 'mean_imputing'],
              'ours': ['Kfold', 'CartVanilla'],
              'catboost': ['vanilla', 'mean_imputing'],
              'sklearn': ['one_hot', 'mean_imputing']}
    n_experiments = len(all_experiments())
    for model_name, model_variants in models.items():
        make_dirs[(RESULTS_DIR / model_name)]
        args = []
        for exp_number, category_size, a in tqdm(all_experiments(), total=n_experiments):
            for variant in model_variants:
                args.append(model_name, variant, exp_number, category_size, a)

        print(F"# of experiments is {len(args)}")
        with multiprocessing.Pool(N_PROCESS) as process_pool:
            process_pool.starmap(worker, args)
