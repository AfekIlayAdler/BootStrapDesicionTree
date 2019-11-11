from Tree.split import Split


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