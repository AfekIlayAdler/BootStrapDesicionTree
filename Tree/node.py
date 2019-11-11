class Node:
    raise NotImplementedError


class Leaf(Node):
    def __init__(self, prediction):
        self.prediction = prediction


class InternalNode(Node):
    def __init__(self,purity, field):
        self.field = field
        self.purity = purity
        self.children_data = {}
        self.children = {}

    def add_child_data(self, df):
        NotImplementedError

    def add_child_nodes(self, df):
        NotImplementedError

    def get_child(self, column, value):
        raise NotImplementedError


class NumericBinaryNode(InternalNode):
    def __init__(self, field, purity,splitting_point: float):
        super().__init__(purity,field)
        self.thr = splitting_point



class CategorialBinaryNode(InternalNode):
    def __init__(self, purity, field,left_values,right_values):
        super().__init__(purity, field)
        self.left_values = left_values
        self.right_values = right_values


class CategorialMultiNode(InternalNode):
    def __init__(self, purity, field,left_values,right_values):
        super().__init__(purity, field)
        self.left_values = left_values
        self.right_values = right_values

