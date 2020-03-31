import multiprocessing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from experiments.interaction.config import RESULTS_DIR, EXP_NAME, Y_COL_NAME, MODELS_DIR
from experiments.interaction.config import VAL_RATIO
from experiments.interaction.fi import permutation_feature_importance
from experiments.interaction.utils import create_x_y
from experiments.interaction.utils import make_dirs, all_experiments, \
    get_fitted_model, n_experiments
from gradient_boosting_trees.gradient_boosting_regressor import CartGradientBoostingRegressor, \
    CartGradientBoostingRegressorKfold


def our_worker(exp_number, category_size, a, predictor_name, predictor):
    model_type = 'ours'
    exp_name = F"{EXP_NAME}__{predictor_name}_exp_{exp_number}_category_size_{category_size}_a_{a}"
    model_path = MODELS_DIR / F"{exp_name}.pkl"
    exp_name = F"{EXP_NAME}_{model_type}_{predictor_name}_exp_{exp_number}_category_size_{category_size}_a_{a}"
    exp_results_path = RESULTS_DIR / F"{exp_name}.csv"
    if exp_results_path.exists():
        return
    X, y = create_x_y(category_size, float(a))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO, random_state=42)
    X_train[Y_COL_NAME] = y_train
    try:
        model = get_fitted_model(model_path, predictor, X, Y_COL_NAME)
    except:
        print(model_path)
        return
    fi_gain = pd.Series(model.compute_feature_importance(method='gain')).sort_index()
    fi_gain /= fi_gain.sum()
    fi_permutation_train = permutation_feature_importance(model_type, model, X_train, y_train)
    fi_permutation_test = permutation_feature_importance(model_type, model, X_test, y_test)
    temp_results = pd.DataFrame([[F"{model_type}_{predictor_name}", category_size, a, fi_gain['x1'],
                                  fi_permutation_train['x1'], fi_permutation_test['x1'], 0,0]])
    temp_results.to_csv(exp_results_path)


if __name__ == '__main__':
    make_dirs([MODELS_DIR, RESULTS_DIR])
    args = []
    for exp_number, category_size, a in tqdm(all_experiments(), total=n_experiments):
        np.random.seed(exp_number)
        for predictor_name, predictor in {'Kfold': CartGradientBoostingRegressorKfold,
                                          'CartVanilla': CartGradientBoostingRegressor}.items():
            args.append((exp_number, category_size, a, predictor_name, predictor))
    print(F"# of experiments is {len(args)}")
    with multiprocessing.Pool(4) as process_pool:
        process_pool.starmap(our_worker, args)