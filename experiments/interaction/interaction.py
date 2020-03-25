import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from experiments.interaction.config import RESULTS_DIR, EXP_NAME, Y_COL_NAME, MODELS_DIR
from experiments.interaction.config import VAL_RATIO
from experiments.interaction.utils import create_x_y
from experiments.interaction.utils import make_dirs, all_experiments, \
    get_fitted_model, save_model, n_experiments
from gradient_boosting_trees.gradient_boosting_regressor import CartGradientBoostingRegressor, \
    CartGradientBoostingRegressorKfold

if __name__ == '__main__':
    make_dirs([MODELS_DIR, RESULTS_DIR])
    for exp_number, category_size, a in tqdm(all_experiments(), total=n_experiments):
        np.random.seed(exp_number)
        for predictor_name, predictor in {'Kfold': CartGradientBoostingRegressorKfold,
                                          'CartVanilla': CartGradientBoostingRegressor}.items():
            exp_name = F"{EXP_NAME}__{predictor_name}_exp_{exp_number}_category_size_{category_size}_a_{a}"
            model_path = MODELS_DIR / F"{exp_name}.pkl"
            exp_results_path = RESULTS_DIR / F"{exp_name}.pkl"
            X, y = create_x_y(category_size, a)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO, random_state=42)
            X_train[Y_COL_NAME] = y_train
            fitted_model = get_fitted_model(model_path, predictor, X_train, Y_COL_NAME)
            save_model(model_path, fitted_model)
