import numpy as np
from tqdm import tqdm

from experiments.detect_uninformative_feature_simulated_data.config import RESULTS_DIR, EXP_NAME, Y_COL_NAME, MODELS_DIR
from experiments.detect_uninformative_feature_simulated_data.utils import make_dirs, all_experiments, create_x, \
    get_fitted_model, save_model, n_experiments
from gradient_boosting_trees.gradient_boosting_regressor import CartGradientBoostingRegressor, \
    CartGradientBoostingRegressorKfold

if __name__ == '__main__':
    make_dirs([MODELS_DIR, RESULTS_DIR])
    for exp_number, category_size in tqdm(all_experiments(), total=n_experiments):
        np.random.seed(exp_number)
        for predictor_name, predictor in {'Ffold': CartGradientBoostingRegressorKfold,
                                          'CartVanilla': CartGradientBoostingRegressor}.items():
            exp_name = F"{EXP_NAME}__{predictor_name}_exp_{exp_number}_category_size_{category_size}"
            model_path = MODELS_DIR / F"{exp_name}.pkl"
            exp_results_path = RESULTS_DIR / F"{exp_name}.csv"
            X = create_x(category_size)
            fitted_model = get_fitted_model(model_path, predictor, X, Y_COL_NAME)
            save_model(model_path, fitted_model)
