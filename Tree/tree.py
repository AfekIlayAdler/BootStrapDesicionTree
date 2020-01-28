from pathlib import Path

import numpy as np
import pandas as pd

from Tree.get_node import GetNode
from Tree.node import Leaf, InternalNode
from Tree.splitters.cart_splitter import CartRegressionSplitter, CartTwoClassClassificationSplitter
from Tree.tree_feature_importance import weighted_variance_reduction_feature_importance
from Tree.tree_visualizer import TreeVisualizer
from Tree.utils import get_cols_dtypes, impurity_dict, get_col_type


class BaseTree:
    def __init__(self, splitter, label_col_name, max_depth, min_impurity_decrease=0., min_samples_split=2):
        self.label_col_name = label_col_name
        self.splitter = splitter
        self.root = None
        self.column_dtypes = None
        self.min_impurity_decrease = min_impurity_decrease
        self.impurity = impurity_dict.get(self.splitter.type)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def calculate_impurity(self, y) -> float:
        return self.impurity(y)

    def get_node(self, data: pd.DataFrame, depth: int) -> [InternalNode, Leaf]:
        # min_samples_split
        impurity = self.calculate_impurity(data[self.label_col_name])
        # TODO : if impurity = 0: return leaf
        n_samples = data.shape[0]
        leaf_prediction = data[self.label_col_name].mean()
        if impurity == 0:
            return Leaf(leaf_prediction, "pure_leaf", n_samples, impurity)
        if n_samples <= self.min_samples_split:
            return Leaf(leaf_prediction, "min_samples_split", n_samples, impurity)
        # max_depth
        if depth == self.max_depth:
            return Leaf(leaf_prediction, "max_depth", n_samples, impurity)
        best_node, best_node_score = None, np.inf
        for col, col_type in self.column_dtypes.items():
            col_type = get_col_type(col_type)
            node_getter = GetNode(self.splitter, col, self.label_col_name, col_type)
            col_best_node, col_split_purity_score = node_getter.get(data[[col, self.label_col_name]])
            if col_best_node is None:
                continue
            if col_split_purity_score < best_node_score:
                best_node = col_best_node
                best_node_score = col_split_purity_score
        if best_node is None:
            # all x values are the same
            return Leaf(leaf_prediction, "pure_node", n_samples, impurity)
        # min impurity increase
        print(impurity, best_node_score)
        if (impurity - best_node_score) < self.min_impurity_decrease:
            return Leaf(leaf_prediction, "min_impurity_increase", n_samples, impurity)
        best_node.purity = impurity
        best_node.add_child_data(data)
        best_node.add_depth(depth)
        return best_node

    def split(self, node: [InternalNode, Leaf]):
        children_data = node.children_data
        node.children_data = None
        for child_name, child_data in children_data.items():
            child_node = self.get_node(child_data, node.depth + 1)
            node.add_child_nodes(child_name, child_node)
            if isinstance(child_node, InternalNode):
                self.split(child_node)

    def build(self, data: pd.DataFrame):
        self.column_dtypes = get_cols_dtypes(data, self.label_col_name)
        root = self.get_node(data, 1)
        if isinstance(root, InternalNode):
            self.split(root)
        self.root = root

    def predict(self, row):
        node = self.root
        while isinstance(node, InternalNode):
            value = row[node.field]
            node = node.get_child(value)
        return node.prediction


class CartRegressionTree(BaseTree):
    def __init__(self, label_col_name, min_samples_leaf=1, max_depth=np.inf):
        super().__init__(CartRegressionSplitter(min_samples_leaf), label_col_name, max_depth=max_depth)


class CartClassificationTree(BaseTree):
    def __init__(self, label_col_name, min_samples_leaf=1, max_depth=np.inf):
        super().__init__(CartTwoClassClassificationSplitter(min_samples_leaf), label_col_name, max_depth=max_depth)


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
