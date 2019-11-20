import pandas as pd

from Tree.config import PANDAS_CATEGORICAL_COLS
from Tree.find_best_split import FindBestSplit
from Tree.node import Node, Leaf, InternalNode
from Tree.predictor import Predictor
from Tree.utils import get_cols_dtypes


class BaseTree:
    def __init__(self, n_splitter: FindBestSplit, c_splitter: FindBestSplit, predictor: Predictor):
        self.numeric_splitter = n_splitter
        self.categorical_splitter = c_splitter
        self.predictor = predictor
        self.root = None
        self.label_column = None
        self.column_dtypes = None
        self.thr = None

    def calculate_purity(self, y) -> float:
        return self.predictor.calc_impurity(y)

    def get_splitter(self, col_type):
        if col_type in PANDAS_CATEGORICAL_COLS:
            return self.categorical_splitter
        return self.numeric_splitter

    def get_node(self, df: pd.DataFrame) -> [InternalNode, Leaf]:
        # find the best split in order to create and return a node
        # TODO - add:
        # max depth
        # min_samples_split
        # min_samples_leaf
        # min impurity increase
        # TODO - maybe we don't need purity
        purity = self.calculate_purity(df[self.label_column])
        best_node, best_node_score = None, 0
        for col, col_type in self.column_dtypes.items():
            splitter = self.get_splitter(col_type)
            col_best_node = splitter.get_split(df[[col, self.label_column]], n = df.shape[0], col = col)
            if col_best_node.purity > best_node_score:
                best_node = col_best_node
                best_node_score = col_best_node.purity
        if (purity - best_node.purity) <= self.thr:
            return Leaf(self.predictor.predict_on_leaf(df[self.label_column]))
        best_node.add_child_data(df)
        return best_node

    def split(self, node: [InternalNode, Leaf]):
        if isinstance(node, Leaf):
            raise Exception('method split got a leaf node as parameter')
        children_data = node.children_data
        node.children = None
        for child_name, child_data in children_data.items():
            child_node = self.get_node(child_data)
            node.add_child_nodes(child_node)
            if not isinstance(child_node, Leaf):
                self.split(child_node)

    def build(self, data: pd.DataFrame):
        self.column_dtypes = get_cols_dtypes(data, self.label_column)
        root = self.get_node(data)
        self.split(root)
        self.root = root

    def predict(self, row):
        # TODO : fix function
        node = self.root
        while not node.is_leaf:
            value = row[node.field]
            node = node.get_child(value)
        assert node.is_leaf, "arrived the end of the tree but node is not leaf"
        return node.prediction


if __name__ == '__main__':
    df = pd.read_csv('desicion_tree_data.csv')
    df['gender'] = df['gender'].astype('bool')
    df['city'] = df['city'].astype('category')
    pass
