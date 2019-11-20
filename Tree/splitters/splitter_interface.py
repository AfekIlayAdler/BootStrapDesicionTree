import pandas as pd

from Tree.node import InternalNode


class Splitter:
    def _evaluate(self, x, y):
        pass

    def get_split(self, df: pd.DataFrame, n: int, col: str) -> InternalNode:
        raise NotImplementedError


