from Tree.node import InternalNode, Leaf


def weighted_variance_reduction_feature_importance(tree, method='gain'):
    # based on https://stats.stackexchange.com/questions/162162/relative-variable-importance-for-boosting
    queue = [[tree.root]]
    feature_importance = {feature: 0 for feature in tree.column_dtypes.keys()}
    if isinstance(tree.root, Leaf):
        return feature_importance
    tree_depth = 0
    while queue:
        level_nodes = queue.pop(0)
        next_level_nodes = []
        for node in level_nodes:
            children_nodes_weighted_purity = 0
            for _, child in node.children.items():
                if isinstance(child, InternalNode):
                    next_level_nodes.append(child)
                children_nodes_weighted_purity += child.purity
            node_mean_purity_reduction = (node.purity - children_nodes_weighted_purity) / node.n_examples
            # actually node.n_examples is not needed here.
            if method == 'gain':
                feature_importance[node.field] += (node.n_examples / tree.root.n_examples) * node_mean_purity_reduction
            else:  # method == number of modes
                feature_importance[node.field] += 1
        if next_level_nodes:
            queue.append(next_level_nodes)
        tree_depth += 1
    feature_importance = {i: v for i, v in feature_importance.items()}
    return feature_importance
