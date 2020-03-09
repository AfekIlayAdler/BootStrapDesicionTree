from sklearn.model_selection import KFold

from Tree.get_node import GetNode
from Tree.node import CategoricalBinaryNode, NumericBinaryNode

import numpy as np
import pandas as pd


class KFoldGetNode(GetNode):
    def __init__(self, splitter, col_name, label_col_name, col_type, k_folds=5):
        super().__init__(splitter, col_name, label_col_name, col_type)
        self.k_folds = k_folds

    def get_left_right_values(self, data, node):
        if self.col_type == 'numeric':
            left_filtered_data = data[data[self.col_name] <= node.thr]
            right_filtered_data = data[data[self.col_name] > node.thr]
        else:
            left_filtered_data = data[data[self.col_name].isin(node.left_values)]
            right_filtered_data = data[~data[self.col_name].isin(node.left_values)]
        return left_filtered_data[self.label_col_name].values, right_filtered_data[self.label_col_name].values

    def calculate_fold_error(self, node: [CategoricalBinaryNode, NumericBinaryNode], train: pd.DataFrame,
                             val: pd.DataFrame) -> float:
        left_train_response, right_train_response = self.get_left_right_values(train, node)
        left_val_response, right_val_response = self.get_left_right_values(val, node)
        # TODO: we can avoid left_mean & right_mean calculation by passing it from the splitter (node) object
        if self.splitter.type == 'regression':
            # error = sum of squared errors from the prediction
            left_train_mean, right_train_mean = np.mean(left_train_response), np.mean(right_train_response)
            left_var = np.sum(np.square(left_val_response - left_train_mean))
            right_var = np.sum(np.square(right_val_response - right_train_mean))
            return left_var + right_var
        else:
            left_n, right_n = left_val_response.size, right_val_response.size
            left_p, right_p = left_val_response.mean(), right_val_response.mean()
            return left_n * left_p * (1 - left_p) + right_n * right_p(1 - right_p)

    def get(self, df):
        """ the validation score has to be on the same scale as the purity score (good practice),
        so we sum it up snd not divide it """
        # TODO: what happens when in validation there are missing values or values that do not appear in train, maybe avoidable by imputing
        best_node = self._get(df)
        if not best_node:
            return None, None
        # now we will calculate a real estimate for this impurity using kfold
        validation_error = 0
        kf = KFold(n_splits=self.k_folds)
        for train_index, validation_index in kf.split(df):
            train, validation = df.iloc[train_index], df.iloc[validation_index]
            temp_kfold_node = self._get(train)
            # TODO: check if those next two lines makes sense-
            if not temp_kfold_node:
                return None, None
            validation_error += self.calculate_fold_error(temp_kfold_node, train, validation)
        return best_node, validation_error
