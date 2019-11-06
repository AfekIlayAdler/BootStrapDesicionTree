from Tree.node import Node
from Tree.split_evaluetr import Splitter


class BaseTree:
    def __init__(self, split_evaluater: Splitter):
        self.split_evaluater = split_evaluater
        self.root = None

    def get_split(self, train,y) -> Node:
        pass

    def split(self, node):
        pass

    def build(self, train):
        root = self.get_split(train)
        root.is_root = True
        self.split(root)
        self.root = root

    def predict(self, row):
        node = self.root
        while not node.is_leaf:
            value = row[node.field]
            node = node.go_down(value)
        assert node.is_leaf, "arrived the end of the tree but node is not leaf"
        return node.prediction


class CartClfTree(BaseTree):
    def __init__(self,split_evaluater):
        super(CartClfTree, self).__init__(split_evaluater)

    def get_split(self, train, y) -> Node:
        """
        cols_dtypes = get_cols_dtypes()
        best_split, best_split_score  = None
        for col in train.columns:
            col_dtype = cols_dtypes
            split_score, split = self.split_evaluater.get_split(col_dtype,train[col],y)
            if split_score > best_split_score:
                best_split = split
        # HERE WE NEED TO RETURN THE TYPE OF NODE
        """
        pass

    def split(self, node):
        pass




if __name__ == '__main__':
    pass
