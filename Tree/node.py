from typing import Optional


class Node:
    def __init__(self, field=None, is_leaf=False, prediction=None, splitting_point=None):
        self.splitting_point = splitting_point
        self.is_leaf = is_leaf
        self.field = field
        self.prediction = prediction
        self.children = {}

    def go_down(self, column, value):
        raise NotImplementedError

    # def predict(self, column: str, value: [str, float, int]):
    #     if self.is_leaf:
    #         return self
    #     else:
    #         return self.split(column, value)


class NumericBinaryNode(Node):
    def __init__(self, thr: Optional[float] = None, field=None, is_leaf=False, prediction=None, splitting_point=None):
        super().__init__(field, is_leaf, prediction, splitting_point)
        self.thr = thr

    def go_down(self, column, value):
        pass
