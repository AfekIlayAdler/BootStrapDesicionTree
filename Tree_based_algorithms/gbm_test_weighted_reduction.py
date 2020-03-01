from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Tree.tree import CartRegressionTree
from Tree.tree_feature_importance import weighted_variance_reduction_feature_importance
from Tree.tree_visualizer import TreeVisualizer

np.random.seed(3)
input_path = Path.cwd().parent / "Datasets/boston_house_prices/boston_house_prices.csv"
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

df = pd.read_csv(input_path, dtype=dtypes)
X_train, X_test = train_test_split(df, test_size=0.1)
tree = CartRegressionTree("y", max_depth=4)
# tree = CartRegressionTreeKFold("y", max_depth=4)
tree.build(X_train)
print(tree.predict(X_test.to_dict('records')))
tree_vis = TreeVisualizer()
tree_vis.plot(tree.root)
fi = weighted_variance_reduction_feature_importance(tree)
print(pd.Series(fi) / pd.Series(fi).sum())
