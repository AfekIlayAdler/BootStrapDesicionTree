from pathlib import Path

import numpy as np
import pandas as pd

from Tree.get_node import GetNode
from Tree.node import Leaf, InternalNode
from Tree.splitters.cart_splitter import CartRegressionSplitter
from Tree.utils import get_cols_dtypes, impurity_dict, get_col_type


class BaseTree:
    def __init__(self, splitter, label_col_name, thr, min_samples_split=2,max_depth = np.inf):
        self.label_col_name = label_col_name
        self.splitter = splitter
        self.root = None
        self.column_dtypes = None
        self.thr = thr
        self.impurity = impurity_dict.get(self.splitter.type)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def calculate_impurity(self, y) -> float:
        return self.impurity(y)

    def get_node(self, df: pd.DataFrame, depth: int) -> [InternalNode, Leaf]:
        # min_samples_split
        if df.shape[0] <= self.min_samples_split:
            return Leaf(df[self.label_col_name].mean(), "min_samples_split")
        # max_depth
        if depth == self.max_depth:
            return Leaf(df[self.label_col_name].mean(), "max_depth")
        impurity = self.calculate_impurity(df[self.label_col_name])
        best_node, best_node_score = None, np.inf
        for col, col_type in self.column_dtypes.items():
            col_type = get_col_type(col_type)
            node_getter = GetNode(self.splitter, col, self.label_col_name, col_type)
            col_best_node = node_getter.get(df[[col, self.label_col_name]])
            if col_best_node is None:
                continue
            if col_best_node.purity < best_node_score:
                best_node = col_best_node
                best_node_score = col_best_node.purity
        if best_node is None:
            # all x values are the same
            return Leaf(df[self.label_col_name].mean(), "pure_node")
        # min impurity increase
        # print(f"purity increse: {(impurity - best_node.purity)}")
        if (impurity - best_node.purity) < self.thr:
            return Leaf(df[self.label_col_name].mean(), "min_impurity_increase")
        best_node.add_child_data(df)
        best_node.add_depth(depth)
        return best_node

    def split(self, node: [InternalNode, Leaf]):
        children_data = node.children_data
        node.children_data = None
        for child_name, child_data in children_data.items():
            child_node = self.get_node(child_data, node.depth+1)
            node.add_child_nodes(child_name, child_node)
            # TODO: Understand if both leaf and internal node are getting True
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
    def __init__(self, label_col_name, thr=0.01):
        super().__init__(CartRegressionSplitter(), label_col_name, thr)


if __name__ == '__main__':
    input_path = Path.cwd().parent / "Datasets\house_prices_regrssion\house_pricing_moc_dataset.csv"
    df = pd.read_csv(input_path, dtype={'OverallCond': 'category', 'HouseStyle': 'category'})
    tree = CartRegressionTree("SalePrice")
    tree.build(df)
    test = {'LotArea': 8450, 'YearBuilt': 2003, 'OverallCond': 'medium', 'HouseStyle': '2Story'}
    print(tree.predict(test))
