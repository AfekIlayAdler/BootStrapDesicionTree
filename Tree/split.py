from Tree.node import Node
import pandas as pd


class Split:
    def __init__(self):
        raise NotImplementedError

    def do_split(self, train: pd.DataFrame) -> Node:
        raise NotImplementedError
