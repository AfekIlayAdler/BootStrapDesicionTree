from pathlib import Path

import pandas as pd

from Tree.config import PANDAS_CATEGORICAL_COLS
from Tree.get_node import GetNode
from Tree.node import Leaf, InternalNode
from Tree.splitters.cart_splitter import CartRegressionSplitter
from Tree.utils import get_cols_dtypes, impurity_dict


def get_col_type(col_type):
    if col_type in PANDAS_CATEGORICAL_COLS:
        return 'categorical'
    return 'numeric'


class BaseTree:
    def __init__(self, splitter, label_col_name):
        self.label_col_name = label_col_name
        self.splitter = splitter
        self.root = None
        self.column_dtypes = None
        self.thr = None
        self.impurity = impurity_dict.get(self.splitter.type)

    def calculate_impurity(self, y) -> float:
        return self.impurity(y)

    def get_node(self, df: pd.DataFrame) -> [InternalNode, Leaf]:
        # find the best split in order to create and return a node
        # TODO - add:
        # max depth
        # min_samples_split
        # min_samples_leaf
        # min impurity increase
        # TODO - maybe we don't need purity
        purity = self.calculate_impurity(df[self.label_col_name])
        best_node, best_node_score = None, 0
        for col, col_type in self.column_dtypes.items():
            col_type = get_col_type(col_type)
            node_getter = GetNode(self.splitter, col, self.label_col_name, col_type)
            col_best_node = node_getter.get(df[[col, self.label_col_name]])
            if col_best_node.purity > best_node_score:
                best_node = col_best_node
                best_node_score = col_best_node.purity
        if (purity - best_node.purity) <= self.thr:
            return Leaf(self.impurity.predict_on_leaf(df[self.label_col_name]))
        best_node.add_child_data(df)
        return best_node

    def split(self, node: [InternalNode, Leaf]):
        if isinstance(node, Leaf):
            raise Exception('method split got a leaf node as parameter')
        children_data = node.children_data
        node.children_data = None
        for child_name, child_data in children_data.items():
            child_node = self.get_node(child_data)
            node.add_child_nodes(child_name, child_node)
            if not isinstance(child_node, Leaf):
                self.split(child_node)

    def build(self, data: pd.DataFrame):
        self.column_dtypes = get_cols_dtypes(data, self.label_col_name)
        root = self.get_node(data)
        self.split(root)
        self.root = root

    def predict(self, row):
        node = self.root
        while not node.is_leaf:
            value = row[node.field]
            node = node.get_child(value)
        assert node.is_leaf, "arrived the end of the tree but node is not leaf"
        return node.prediction


class CartRegressionTree(BaseTree):
    def __init__(self, label_col_name):
        super().__init__(CartRegressionSplitter(), label_col_name)


if __name__ == '__main__':
    input_path = Path.cwd().parent / "Datasets\house_prices_regrssion\house_pricing_moc_dataset.csv"
    df = pd.read_csv(input_path, dtype={'OverallCond': 'category', 'HouseStyle': 'category'})
    tree = CartRegressionTree("SalePrice")
    tree.build(df)
    a = 5
