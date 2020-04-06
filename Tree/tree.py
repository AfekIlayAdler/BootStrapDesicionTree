from typing import List, Dict

import numpy as np
import pandas as pd

from Tree.get_node import GetNode
from Tree.kfold_get_node import KFoldGetNode
from Tree.node import InternalNode, Leaf
from Tree.splitters.cart_splitter import CartRegressionSplitter, CartTwoClassClassificationSplitter
from Tree.utils import impurity_dict, get_cols_dtypes, get_col_type

MAX_DEPTH = np.inf
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 2
MIN_IMPURITY_DECREASE = 0.


class BaseTree:
    def __init__(self, node_getter, splitter,
                 max_depth, min_impurity_decrease, min_samples_split):
        self.node_getter = node_getter
        self.splitter = splitter
        self.min_impurity_decrease = min_impurity_decrease
        self.impurity = impurity_dict.get(self.splitter.type)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.column_dtypes = None
        self.column_to_index = None
        self.n_leaves = 0

    def calculate_impurity(self, y) -> float:
        return self.impurity(y)

    def _best_split(self, X, y):
        best_node, best_node_score, best_indices = None, np.inf, None
        # TODO add indices left and indices right
        for col, col_type in self.column_dtypes.items():
            col_type = get_col_type(col_type)
            node_getter = self.node_getter(self.splitter, col, col_type)
            col_best_node, col_split_purity_score, indices = node_getter.get(X[:, self.column_to_index[col]], y)
            if col_best_node is None:
                continue
            if col_split_purity_score < best_node_score:
                best_node = col_best_node
                best_node_score = col_split_purity_score
                best_indices = indices
        return best_node, best_node_score, best_indices

    def _grow_tree(self, X: np.array, y: np.array, depth: int) -> [InternalNode, Leaf]:
        impurity = self.calculate_impurity(y)
        n_samples = X.shape[0]
        leaf = Leaf(y.mean(), "_", n_samples, impurity)
        is_leaf = (impurity == 0) or (n_samples <= self.min_samples_split) or (depth == self.max_depth)
        if is_leaf:
            return leaf
        node, node_score, indices = self._best_split(X, y)
        is_leaf = (node is None) or ((impurity - node_score) < self.min_impurity_decrease)
        if is_leaf:
            return leaf
        node.purity = impurity
        node.left = self._grow_tree(X[indices['left']], y[indices['left']], depth + 1)
        node.right = self._grow_tree(X[indices['right']], y[indices['right']], depth + 1)
        self.n_leaves += 2
        node.add_depth(depth)
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.column_dtypes = get_cols_dtypes(X)
        self.column_to_index = {col: i for i,col in enumerate(X.columns)}
        X, y = X.values, y.values
        self.root = self._grow_tree(X, y, 0)

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
    def __init__(self,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(node_getter=GetNode,
                         splitter=CartRegressionSplitter(min_samples_leaf),
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split)


class CartClassificationTree(BaseTree):
    def __init__(self,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(node_getter=GetNode,
                         splitter=CartTwoClassClassificationSplitter(min_samples_leaf),
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split)


class CartRegressionTreeKFold(BaseTree):
    def __init__(self,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(node_getter=KFoldGetNode,
                         splitter=CartRegressionSplitter(min_samples_leaf),
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split)
