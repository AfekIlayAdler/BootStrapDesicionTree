from pathlib import Path
from numpy import arange

# exp
CATEGORIES = list(arange(2, 10, 2)) + list(arange(10, 110, 10))
N_EXPERIMENTS = 1
N_PERMUTATIONS = 20

# data
CATEGORY_COLUMN_NAME = 'random_category'
VAL_RATIO = 0.15
CATEGORICAL_COLUMNS = [CATEGORY_COLUMN_NAME, 'CHAS', 'RAD']
# gbm
MAX_DEPTH = 3
N_ESTIMATORS = 100
LEARNING_RATE = 0.1

# io
MODELS_DIR = Path(F"results/models/")
RESULTS_DIR = Path(F"results/experiments_results/")
EXP_NAME = F"detect_uninformative_feature_max_depth_{MAX_DEPTH}_nestimators_{N_ESTIMATORS}_learning_rate_{LEARNING_RATE}"
