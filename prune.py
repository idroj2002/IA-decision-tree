from DataStructures import DecisionNode

def prune_tree(root: DecisionNode, threshold: float):
    """
    Prune the tree given a certain root and threshold
    """
    if not is_leaf(root.tb):
        prune_tree(root.tb, threshold)
    if not is_leaf(root.fb):
        prune_tree(root.fb, threshold)
    if both_children_leaf(root) and root.split_quality < threshold:
        combine_leaves(root)


def is_leaf(node: DecisionNode):
    """
    Check if a node is non-leaf
    """
    return node.results is not None


def both_children_leaf(node: DecisionNode):
    """
    Check if both children of a node are leaf nodes
    """
    return is_leaf(node.tb) and is_leaf(node.fb)


def combine_leaves(node: DecisionNode):
    """
    Combine the leaves of a node
    """
    node.results = {**(node.tb.results or {}), **(node.fb.results or {})}
    node.tb = None
    node.fb = None
    node.split_quality = 0