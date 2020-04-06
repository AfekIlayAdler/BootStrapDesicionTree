from pathlib import Path
from numpy import arange

# exp
CATEGORIES = list(arange(2, 10, 2)) + list(arange(10, 110, 10))
N_EXPERIMENTS = 15
N_PERMUTATIONS = 20

# data
A1 = 3
A2 = 2
SIGMA = 10
N_ROWS = 10 ** 3
CATEGORY_COLUMN_NAME = 'random_category'
VAL_RATIO = 0.15

# gbm
MAX_DEPTH = 3
N_ESTIMATORS = 100
LEARNING_RATE = 0.1

# io
MODELS_DIR = Path(F"results/models/")
RESULTS_DIR = Path(F"results/experiments_results/")
EXP_NAME = F"detect_uninformative_feature_max_depth_{MAX_DEPTH}_nestimators_{N_ESTIMATORS}_learning_rate_{LEARNING_RATE}"
