from Tree.node import InternalNode


def weighted_variance_reduction_feature_importance(tree):
    # TODO: chech if bark
    queue = [[tree.root]]
    feature_importance = {feature: 0 for feature in tree.column_dtypes.keys()}
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
            feature_importance[node.field] += (node.n_examples/tree.root.n_examples)*(node.purity/node.n_examples - children_nodes_weighted_purity/node.n_examples)
        if next_level_nodes:
            queue.append(next_level_nodes)
        tree_depth += 1
    feature_importance = {i: v for i, v in feature_importance.items()}
    return feature_importance

