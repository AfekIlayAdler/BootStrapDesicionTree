
class Splitter:
    def _evaluate(self,x,y):
        raise NotImplementedError

    def get_split(self, x,y):
        raise NotImplementedError