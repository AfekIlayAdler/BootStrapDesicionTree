from os import mkdir
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from gradient_boosting_trees.gradient_boosting_regressor import CartGradientBoostingRegressor, \
    CartGradientBoostingRegressorKfold

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
    N_EXPERIMENTS = 1
    SEED = 10
    np.random.seed(SEED)
    CATEGORIES = np.arange(20, 220, 20)
    MAX_DEPTH = 4
    N_ESTIMATORS = 100
    LEARNING_RATE = 0.01
    RANDOM_CATEGORY_NAME = 'RANDOM_CATEGORY'
    input_path = Path.cwd().parent.parent / "Datasets/boston_house_prices/boston_house_prices.csv"
    EXP_NAME = F"boston_house_prices_max_depth_{MAX_DEPTH}_nestimators_{N_ESTIMATORS}_learning_rate_{LEARNING_RATE}"
    RESULTS_DIR = Path(F"results/")
    if not RESULTS_DIR.exists():
        mkdir(RESULTS_DIR)
    reg_results = pd.DataFrame(index=sorted_index)
    kfold_results = pd.DataFrame(index=sorted_index)
    df = pd.read_csv(input_path, dtype=dtypes)
    for category_size in tqdm(CATEGORIES, total=len(CATEGORIES)):
        for exp in range(N_EXPERIMENTS):
            kfold_exp_name = F"{EXP_NAME}__kfold_exp_{exp}_category_size_{category_size}"
            reg_exp_name = F"{EXP_NAME}__reg_exp_{exp}_category_size_{category_size}"
            kfold_path = RESULTS_DIR / F"{kfold_exp_name}.csv"
            reg_path = RESULTS_DIR / F"{reg_exp_name}.csv"
            if kfold_path.exists() or reg_path.exists():
                continue
            df[RANDOM_CATEGORY_NAME] = np.random.randint(0, category_size, df.shape[0])
            df[RANDOM_CATEGORY_NAME] = df[RANDOM_CATEGORY_NAME].astype('category')
            # reg
            regular_gbm = CartGradientBoostingRegressor("y", max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
            regular_gbm.fit(df)
            pd.Series(regular_gbm.compute_feature_importance()).sort_index().to_csv(F"{reg_path.parent}/{reg_path.name}", header=True)
            # kfold
            kfold_gbm = CartGradientBoostingRegressorKfold("y", max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
            kfold_gbm.fit(df)
            pd.Series(kfold_gbm.compute_feature_importance()).sort_index().to_csv(kfold_path, header=True)
            with open(RESULTS_DIR / F"{reg_exp_name}.pkl", 'wb') as output:
                pickle.dump(kfold_gbm, output, pickle.HIGHEST_PROTOCOL)