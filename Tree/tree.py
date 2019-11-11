from Tree.node import Node
from Tree.find_best_split import FindBestSplit
from Tree.predictor import Predictor
from Tree.utils import get_cols_dtypes

import pandas as pd


class BaseTree:
    def __init__(self, n_splitter: FindBestSplit, c_splitter: FindBestSplit,predictor: Predictor):
        self.numeric_splitter = n_splitter
        self.categorical_splitter = c_splitter
        self.predictor = predictor
        self.root = None
        self.label_column = None
        self.column_dtypes = None
        self.thr = None

    def calculate_purity(self, y) -> float:
        return self.predictor.calc_impurity(y)

    def get_split(self, train: pd.DataFrame) -> Node:
        """
        creates a node, saves the data of it's children
        """
        purity = self.calculate_purity(train[self.label_column])
        split, best_split_score = None, 0
        for col, col_type in self.column_dtypes.items():
            if col_type in ('category', 'bool'):
                col_best_split = self.categorical_splitter.get_split(train[col], train[self.label_column])
            else:
                col_best_split = self.numeric_splitter.get_split(train[col], train[self.label_column])
            if col_best_split.score > best_split_score:
                split = col_best_split
        if purity - split.score <= self.thr:
            # TODO : create leaf
            return split.create_leaf(train[self.label_column])
        # TODO : do split
        return split.do_split(train)

    def split(self, node: Node):
        children_data = node.children_data
        node.children = None
        for child_name, child_data in children_data.items():
            child_node = self.get_split(child_data)
            node.children[child_name] = child_node
            if not child_node.is_leaf:
                self.split(child_node)

    def build(self, train):
        self.column_dtypes = get_cols_dtypes(train)
        root = self.get_split(train)
        self.split(root)
        self.root = root

    def predict(self, row):
        node = self.root
        while not node.is_leaf:
            value = row[node.field]
            node = node.go_down(value)
        assert node.is_leaf, "arrived the end of the tree but node is not leaf"
        return node.prediction


if __name__ == '__main__':
    df = pd.read_csv('desicion_tree_data.csv')
    df['gender'] = df['gender'].astype('bool')
    df['city'] = df['city'].astype('category')
    pass
