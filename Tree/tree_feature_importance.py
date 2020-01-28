from Tree.node import InternalNode


def weighted_variance_reduction_feature_importance(tree):
    # TODO: chech if bark
    queue = [[tree.root]]
    feature_importance = {feature: 0 for feature in tree.column_dtypes.keys()}
    counter = 0
    while queue:
        print(counter)
        level_nodes = queue.pop(0)
        next_level_nodes = []
        for node in level_nodes:
            children_nodes_weighted_purity = 0
            if node.field == 'YearBuilt':
                print(F"node examples {node.n_examples},purity - {node.purity}")
            for _, child in node.children.items():
                if isinstance(child, InternalNode):
                    next_level_nodes.append(child)
                children_nodes_weighted_purity += child.n_examples*child.purity # (child.n_examples/node.n_examples) * child.purity
                if node.field == 'YearBuilt':
                    print(F"child examples {child.n_examples},purity - {child.purity}")
            feature_importance[node.field] += node.n_examples*node.purity - children_nodes_weighted_purity
        if next_level_nodes:
            queue.append(next_level_nodes)
        counter += 1
    feature_importance = {i: v / tree.root.n_examples for i, v in feature_importance.items()}
    return feature_importance
