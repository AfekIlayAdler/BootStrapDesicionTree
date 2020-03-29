import multiprocessing

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from experiments.detect_uninformative_feature_simulated_data.config import RESULTS_DIR, EXP_NAME, Y_COL_NAME, \
    MODELS_DIR, VAL_RATIO
from experiments.detect_uninformative_feature_simulated_data.utils import make_dirs, all_experiments, \
    get_fitted_model, save_model, n_experiments, create_x_y
from gradient_boosting_trees.gradient_boosting_regressor import CartGradientBoostingRegressor, \
    CartGradientBoostingRegressorKfold


def worker(model_path, predictor, X, Y_COL_NAME):
    fitted_model = get_fitted_model(model_path, predictor, X, Y_COL_NAME)
    save_model(model_path, fitted_model)


if __name__ == '__main__':
    make_dirs([MODELS_DIR, RESULTS_DIR])
    args = []
    for exp_number, category_size in tqdm(all_experiments(), total=n_experiments):
        np.random.seed(exp_number)
        for predictor_name, predictor in {'Ffold': CartGradientBoostingRegressorKfold,
                                          'CartVanilla': CartGradientBoostingRegressor}.items():
            exp_name = F"{EXP_NAME}__{predictor_name}_exp_{exp_number}_category_size_{category_size}"
            model_path = MODELS_DIR / F"{exp_name}.pkl"
            exp_results_path = RESULTS_DIR / F"{exp_name}.csv"
            X, y = create_x_y(category_size)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO, random_state=42)
            X_train[Y_COL_NAME] = y_train
            args.append((model_path, predictor, X_train, Y_COL_NAME))
    with multiprocessing.Pool(4) as process_pool:
        process_pool.starmap(worker, args)


