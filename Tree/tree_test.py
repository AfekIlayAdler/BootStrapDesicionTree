from pathlib import Path
import numpy as np
import pandas as pd

from Tree.tree import CartRegressionTree, CartClassificationTree
from Tree.tree_feature_importance import weighted_variance_reduction_feature_importance
from Tree.tree_visualizer import TreeVisualizer

if __name__ == '__main__':
    # TODO : chech way feature importance is negative
    CHECK_TYPE_REGRESSION = True
    np.random.seed(3)
    input_path = Path.cwd().parent / "Datasets\house_prices_regrssion\house_pricing_moc_dataset.csv"
    df = pd.read_csv(input_path, dtype={'OverallCond': 'category', 'HouseStyle': 'category'})
    if CHECK_TYPE_REGRESSION:
        df['SalePrice'] /= 10000
        tree = CartRegressionTree("SalePrice", max_depth=4)
    else:
        df['SalePrice'] = np.random.randint(0, 2, df.shape[0])
        tree = CartClassificationTree("SalePrice", max_depth=4)
    tree.build(df)
    # test = {'LotArea': 8450, 'YearBuilt': 2003, 'OverallCond': 'medium', 'HouseStyle': '2Story'}
    # tree.predict(test)
    tree_vis = TreeVisualizer()
    tree_vis.plot(tree.root)
    fi = weighted_variance_reduction_feature_importance(tree)
    print(fi)