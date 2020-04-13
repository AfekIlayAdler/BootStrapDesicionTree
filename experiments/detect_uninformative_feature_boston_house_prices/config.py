from numpy import arange

# exp
CATEGORIES = list(arange(2, 10, 2)) + list(arange(10, 110, 10))
N_EXPERIMENTS = 1

# data
CATEGORICAL_COLUMNS = [CATEGORY_COLUMN_NAME, 'CHAS', 'RAD']
EXP_NAME = F"detect_uninformative_feature_max_depth_{MAX_DEPTH}_nestimators_{N_ESTIMATORS}_learning_rate_{LEARNING_RATE}"
