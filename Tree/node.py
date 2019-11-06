class Node:
    def __init__(self):
        self.is_root = False
        self.is_leaf = False
        self.prediction = None
        self.data = None
        self.field = None
        self.splitting_point = None
        self.children = {}
        self.father_score = None

    def go_down(self,column,value):
        raise NotImplementedError

    def predict(self, column: str, value: [str, float, int]):
        if self.is_leaf:
            return self
        else:
            return self.split(column, value)


