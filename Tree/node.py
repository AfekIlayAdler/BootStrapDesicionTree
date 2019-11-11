class Node:
    raise NotImplementedError


class Leaf(Node):
    def __init__(self, prediction):
        self.prediction = prediction


class InternalNode(Node):
    def __init__(self, field=None):
        self.field = field
        self.children = {}

    def go_down(self, column, value):
        raise NotImplementedError


class NumericBinaryNode(InternalNode):
    def __init__(self, field, splitting_point: float):
        super().__init__(field)
        self.thr = splitting_point

    def go_down(self, column, value):
        pass
