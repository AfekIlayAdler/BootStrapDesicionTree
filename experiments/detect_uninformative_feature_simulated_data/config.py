from pathlib import Path
from numpy import arange

# exp
CATEGORIES = list(arange(2, 10, 2)) + list(arange(10, 200, 10))
N_EXPERIMENTS = 5

# data
A1 = 3
A2 = 2
SIGMA = 10
N_ROWS = 10 ** 3
CATEGORY_COLUMN_NAME = 'random_category'
Y_COL_NAME = 'y'

# gbm
MAX_DEPTH = 4
N_ESTIMATORS = 100
LEARNING_RATE = 0.1

# io
MODELS_DIR = Path(F"results/models/")
RESULTS_DIR = Path(F"results/experiments_results/")
EXP_NAME = F"detect_uninformative_feature_{MAX_DEPTH}_nestimators_{N_ESTIMATORS}_learning_rate_{LEARNING_RATE}"
