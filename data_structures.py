from collections import deque


class DecisionNode:
    """
    Decision node structure to build the decision tree
    """
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None, split_quality=0):
        """
        t8: We have 5 member variables:
        - col is the column index which represents the
          attribute we use to split the node
        - value corresponds to the answer that satisfies
          the question
        - tb and fb are internal nodes representing the
          positive and negative answers, respectively
        - results is a dictionary that stores the result
          for this branch. Is None except for the leaves
        """

        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        self.split_quality = split_quality  # How good is the split


class Stack:
    """
    Stack data structure
    """

    def __init__(self):
        self.stack = deque()

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        return self.stack.pop()

    def is_empty(self):
        return len(self.stack) == 0
