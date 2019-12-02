import pandas as pd


class Node:
    pass


class Leaf(Node):
    def __init__(self, prediction):
        self.prediction = prediction


class InternalNode(Node):
    def __init__(self, purity, field):
        self.field = field
        self.purity = purity
        self.children_data = {}
        self.children = {}

    def add_child_data(self, df: pd.DataFrame):
        raise NotImplementedError

    def add_child_nodes(self, child_name: str, child_node):
        self.children.update({child_name: child_node})

    def get_child(self, column, value):
        raise NotImplementedError


class NumericBinaryNode(InternalNode):
    """
    left child is smaller:
    child names are left and right
    """
    def __init__(self, field, purity, splitting_point: float):
        super().__init__(purity, field)
        self.thr = splitting_point

    def add_child_data(self, df):
        left_child = df[df[self.field] <= self.thr]
        right_child = df[df[self.field] > self.thr]
        self.children_data.update(left=left_child, right=right_child)

    def get_child(self, column, value):
        if value <= self.thr:
            return self.children['left']
        return self.children['right']


class CategoricalBinaryNode(InternalNode):
    """
    left child is smaller:
    child names are left and right
    """
    def __init__(self, field,purity, left_values, right_values):
        super().__init__(purity, field)
        self.left_values = left_values
        self.right_values = right_values

    def add_child_data(self, df):
        left_child = df[df[self.field].isin(self.left_values)]
        right_child = df[df[self.field].isin(self.right_values)]
        self.children_data.update(left=left_child, right=right_child)

    def get_child(self, column, value):
        if value in self.left_values:
            return self.children['left']
        return self.children['right']


class CategoricalMultiNode(InternalNode):

    def __init__(self, purity, field):
        super().__init__(purity, field)

    def add_child_data(self, df):
        pass

    def get_child(self, column, value):
        pass
