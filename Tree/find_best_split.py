from Tree.split import Split, CategoricalBinarySplit


class FindBestSplit:
    def _evaluate(self, x, y):
        raise NotImplementedError

    def get_split(self, x, y) -> Split:
        raise NotImplementedError


class NumericBestSplitter(FindBestSplit):
    def _evaluate(self, x, y):
        pass

    def get_split(self, x, y) -> Split:
        pass


class CategorialBestSplitter(FindBestSplit):
    def _evaluate(self, x, y):
        pass

    def get_split(self, x, y) -> Split:
        pass


class MoocCategorialBinaryBestSplitter(FindBestSplit):
    def _evaluate(self, x, y):
        pass

    def get_split(self, x, y) -> Split:
        col_name = x.name
        score = 1
        right_values = x.unique()[:1]
        left_values = x.unique()[1:]
        return CategoricalBinarySplit(col_name, score, right_values, left_values)