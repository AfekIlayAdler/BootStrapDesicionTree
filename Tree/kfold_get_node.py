from sklearn.model_selection import KFold

from Tree.get_node import GetNode
from Tree.node import InternalNode, CategoricalBinaryNode, NumericBinaryNode

import numpy as np
import pandas as pd


class KFoldGetNode(GetNode):
    def __init__(self, splitter, col_name, label_col_name, col_type, k_folds=5):
        super().__init__(splitter, col_name, label_col_name, col_type)
        self.k_folds = k_folds

    def calculate_fold_error(self, node: [CategoricalBinaryNode, NumericBinaryNode], train: pd.DataFrame,
                             val: pd.DataFrame) -> float:
        if self.splitter.type == 'regression':
            if self.col_type == 'numeric':
                # TODO: we can avoid left_mean & right_mean calculation by passing it from the splitter (node) object
                left_mean, right_mean = train[self.col_name <= node.thr][self.label_col_name].mean(), \
                                        train[self.col_name <= node.thr][self.label_col_name].mean()
                val_left_error = np.mean(np.square(val[self.col_name <= node.thr][self.label_col_name] - left_mean))
                val_right_error = np.mean(np.square(val[self.col_name > node.thr][self.label_col_name] - left_mean))
                return val_left_error + val_right_error
            else:  # self.col_type == 'categorical'
                left_mean, right_mean = train[self.col_name.isin(node.left_values)][self.label_col_name].mean(), \
                                        train[~self.col_name.isin(node.left_values)][self.label_col_name].mean()
                val_left_error = np.mean(
                    np.square(val[self.col_name.isin(node.left_values)][self.label_col_name] - left_mean))
                val_right_error = np.mean(
                    np.square(val[~self.col_name.isin(node.left_values)][self.label_col_name] - right_mean))
                return val_left_error + val_right_error
        else:
            if self.col_type == 'numeric':
                left_response, right_response = val[self.col_name <= node.thr][self.label_col_name].values, \
                                                val[self.col_name <= node.thr][self.label_col_name].values
            else:  # self.col_type == 'categorical':
                left_response, right_response = val[self.col_name.isin(node.left_values)][self.label_col_name].values, \
                                                val[~self.col_name.isin(node.left_values)][self.label_col_name].values
            left_n, right_n = left_response.size, right_response.size
            left_p, right_p = left_response.mean(), right_response.mean()
            return left_n * left_p * (1 - left_p) + right_n * right_p(1 - right_p)

    def get(self, df):
        """
        the validation score has to be on the same scale as the purity score (good practice),
        so we sum it up snd not divide it
        """
        best_node = self.__get(df)
        if not best_node:
            return None, None
        # now we will calculate a real estimate for this impurity using kfold
        validation_error = 0
        kf = KFold(n_splits=self.k_folds)
        for train_index, validation_index in kf.split(df):
            train, validation = df[train_index], df[validation_index]
            temp_kfold_node = self.__get(train)
            validation_error += self.calculate_fold_error(temp_kfold_node, train, validation)
            # TODO: what happens when in validation there are missing values or values that do not appear in train, maybe avoidable by imputing
        mean_error = validation_error
        return best_node, mean_error
