from Tree.splitters.splitter_abstract import Splitter


def get_categorical_node(splitter, series, n, col_name):
    series = Splitter.group_by_mean_response_value(series)
    col_values, split_index, impurity, thr = splitter.get_split(series, n)
    left_values, right_values = col_values[:split_index], col_values[split_index:]
    return splitter.node(impurity, col_name, left_values, right_values)


def get_numeric_node(splitter, series, n, col_name):
    col_values, split_index, impurity, thr = splitter.get_split(series, n)
    return splitter.node(col_name, impurity, thr)