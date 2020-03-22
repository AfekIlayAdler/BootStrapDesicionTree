from pathlib import Path
from numpy import arange

# exp
CATEGORIES = list(arange(2, 10, 2)) + list(arange(10, 200, 10))
A1_VALUES = [2, 10, 50, 100, 200]
N_EXPERIMENTS = 2

# data
A1 = 2
A2 = 5
SIGMA = 3
N_ROWS = 10 ** 3
VAL_RATIO = 0.15
CATEGORY_COLUMN_NAME = 'category'
Y_COL_NAME = 'y'

# gbm
MAX_DEPTH = 4
N_ESTIMATORS = 100
LEARNING_RATE = 0.1

# io
MODELS_DIR = Path(F"results/models/")
RESULTS_DIR = Path(F"results/experiments_results/")
EXP_NAME = F"interaction_{MAX_DEPTH}_nestimators_{N_ESTIMATORS}_learning_rate_{LEARNING_RATE}"
