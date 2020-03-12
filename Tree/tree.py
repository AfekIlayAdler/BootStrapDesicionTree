from typing import List, Dict

import numpy as np
import pandas as pd
from Tree.get_node import GetNode
from Tree.kfold_get_node import KFoldGetNode
from Tree.node import Leaf, InternalNode
from Tree.splitters.cart_splitter import CartRegressionSplitter, CartTwoClassClassificationSplitter
from Tree.utils import get_cols_dtypes, impurity_dict, get_col_type

MAX_DEPTH = np.inf
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 2
MIN_IMPURITY_DECREASE = 0.


class BaseTree:
    def __init__(self, node_getter, splitter, label_col_name,
                 max_depth, min_impurity_decrease, min_samples_split):
        self.node_getter = node_getter
        self.splitter = splitter
        self.label_col_name = label_col_name
        self.min_impurity_decrease = min_impurity_decrease
        self.impurity = impurity_dict.get(self.splitter.type)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.column_dtypes = None
        self.n_leaves = 0

    def calculate_impurity(self, y) -> float:
        return self.impurity(y)

    def get_node(self, data: pd.DataFrame, depth: int) -> [InternalNode, Leaf]:
        # min_samples_split
        impurity = self.calculate_impurity(data[self.label_col_name])
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
            node_getter = self.node_getter(self.splitter, col, self.label_col_name, col_type)
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
        # print(impurity, best_node_score)
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
            else:
                self.n_leaves += 1

    def build(self, data: pd.DataFrame):
        self.column_dtypes = get_cols_dtypes(data, self.label_col_name)
        root = self.get_node(data, 1)
        if isinstance(root, InternalNode):
            self.split(root)
        self.root = root

    def predict(self, records: List[Dict]) -> np.array:
        results = np.zeros(len(records))
        for i, row in enumerate(records):
            node = self.root
            while isinstance(node, InternalNode):
                value = row[node.field]
                node = node.get_child(value)
            results[i] = node.prediction
        return results


class CartRegressionTree(BaseTree):
    def __init__(self, label_col_name,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(node_getter=GetNode,
                         splitter=CartRegressionSplitter(min_samples_leaf),
                         label_col_name=label_col_name,
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split)


class CartClassificationTree(BaseTree):
    def __init__(self, label_col_name,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(node_getter=GetNode,
                         splitter=CartTwoClassClassificationSplitter(min_samples_leaf),
                         label_col_name=label_col_name,
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split)


class CartRegressionTreeKFold(BaseTree):
    def __init__(self, label_col_name,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(node_getter=KFoldGetNode,
                         splitter=CartRegressionSplitter(min_samples_leaf),
                         label_col_name=label_col_name,
                         max_depth=max_depth,
                         min_impurity_decrease = min_impurity_decrease,
                         min_samples_split=min_samples_split)

