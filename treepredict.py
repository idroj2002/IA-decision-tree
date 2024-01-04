#!/usr/bin/env python3
import collections
from prune import prune_tree
import evaluation
import random
import sys
from math import log2
from typing import List, Tuple, Union

from data_structures import Stack, DecisionNode

# Used for typing
Data = List[List]


def read(file_name: str, separator1: str = ",", separator2: str = "\t") -> Tuple[List[str], Data]:
    """
    t3: Load the data into a bidimensional list.
    Return the headers as a list, and the data
    """
    header = None
    data = []
    with open(file_name, "r") as fh:
        for line in fh:
            line = line.replace(separator2, separator1)
            values = line.strip('').split(separator1)
            if header is None:
                header = values
                continue
            data.append([_parse_value(v) for v in values])
    return header, data


def _parse_value(v: str):
    try:
        if float(v) == int(v):
            return int(v)
        else:
            return float(v)
    except ValueError:
        return v
    # try:
    #     return float(v)
    # except ValueError:
    #     try:
    #         return int(v)
    #     except ValueError:
    #         return v


def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result)
    """
    return dict(collections.Counter(row[-1] for row in part))

    # results = collections.Counter()
    # for row in part:
    #     c = row[-1]
    #     results[c] += 1
    # return dict(results)

    # results = {}
    # for row in part:
    #     c = row[-1]
    #     if c not in results:
    #         results[c] = 0
    #     results[c] += 1
    # return results


def gini_impurity(part: Data):
    """
    t5: Computes the Gini index of a node
    """
    total = len(part)
    if total == 0:
        return 0

    results = unique_counts(part)
    imp = 1
    for v in results.values():
        imp -= (v / total) ** 2
    return imp


def entropy(part: Data):
    """
    t6: Entropy is the sum of p(x)log(p(x))
    across all the different possible results
    """
    total = len(part)
    results = unique_counts(part)

    probs = (v / total for v in results.values())
    return -sum(p * log2(p) for p in probs)

    # imp = 0
    # for v in results.values():
    #     p = v / total
    #     imp -= p * log2(p)
    # return imp


def _split_numeric(prototype: List, column: int, value):
    return prototype[column] >= value


def _split_categorical(prototype: List, column: int, value: str):
    return prototype[column] == value


def divide_eset(part: Data, column: int, value) -> Tuple[Data, Data]:
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    """if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical
    #...
    return (set1, set2)"""

    if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical
        # Split "part" according "split_function"
    set1, set2 = [], []
    for row in part:  # For each row in the dataset
        if split_function(row, column, value):  # If it matches the criteria
            set1.append(row)  # Add it to the first set
        else:
            set2.append(row)  # Add it to the second set
    return set1, set2  # Return both sets


def build_tree(part: Data, score_f=entropy, beta=0):
    """
    t9: Define a new function build_tree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s/Delta i(s,t) < /beta
    """

    if len(part) == 0:
        return DecisionNode()

    current_score = score_f(part)
    if current_score == 0:
        return DecisionNode(results=unique_counts(part))  # Pure node

    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(part[0]) - 1  # -1 because the last column is the label
    for col in range(0, column_count):  # Search the best parameters to use
        column_values = set()
        for row in part:
            column_values.add(row[col])

        for value in column_values:
            (set1, set2) = divide_eset(part, col, value)
            p = float(len(set1)) / len(part)
            gain = current_score - p * score_f(set1) - (1 - p) * score_f(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > beta:
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=build_tree(best_sets[0]), fb=build_tree(best_sets[1]), split_quality=best_gain)
    else:
        return DecisionNode(results=unique_counts(part), split_quality=best_gain)


def iterative_build_tree(part: Data, score_f=entropy, beta=0):
    """
    t10: Define the iterative version of the function build_tree
    """

    if len(part) == 0:
        return DecisionNode()

    current_score = score_f(part)
    if current_score == 0:
        return DecisionNode(results=unique_counts(part))  # Pure node

    stack = Stack()
    node_stack = Stack()
    stack.push((False, part, None, 0))
    while not stack.is_empty():
        connection_node, data, criteria, split_quality = stack.pop()
        if not connection_node:
            current_score = score_f(data)
            if current_score == 0:
                node_stack.push(DecisionNode(results=unique_counts(data), split_quality=0))  # Pure node
            else:
                best_gain = 0.0
                best_criteria = None
                best_sets = None
                column_count = len(data[0]) - 1
                for col in range(0, column_count):  # Search for the best parameters
                    column_values = set()
                    for row in data:
                        column_values.add(row[col])
                    for value in column_values:
                        (set1, set2) = divide_eset(data, col, value)
                        p = float(len(set1)) / len(data)
                        gain = current_score - p * score_f(set1) - (1 - p) * score_f(set2)
                        if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                            best_gain = gain
                            best_criteria = (col, value)
                            best_sets = (set1, set2)
                if best_gain > beta:
                    stack.push((True, data, best_criteria, best_gain))
                    stack.push((False, best_sets[0], best_criteria, best_gain))
                    stack.push((False, best_sets[1], best_criteria, best_gain))
                else:
                    node_stack.push(DecisionNode(results=unique_counts(data)))
        else:
            true_branch = node_stack.pop()
            false_branch = node_stack.pop()
            node_stack.push(DecisionNode(col=criteria[0], value=criteria[1], tb=true_branch, fb=false_branch,
                                         split_quality=split_quality))
            if len(data) == len(part):
                return node_stack.pop()  # Return the root node


def classify(tree: DecisionNode, values: List):
    if tree.results is not None:
        maximum = max(tree.results.values())
        labels = [k for k, v in tree.results.items() if v == maximum]
        return random.choice(labels)
    if isinstance(tree.value, (int, float)):
        if _split_numeric(values, tree.col, tree.value):
            return classify(tree.tb, values)
        else:
            return classify(tree.fb, values)
    else:
        if _split_categorical(values, tree.col, tree.value):
            return classify(tree.tb, values)
        else:
            return classify(tree.fb, values)


def print_tree(tree: DecisionNode, headers: List[str] = None, indent=""):
    """
    t11: Include the following function
    """
    if tree is None:
        print("None")
        return

    # Is this a leaf node?
    if tree.results is not None:
        print(tree.results)
    else:
        # Print the criteria
        criteria = tree.col
        if headers:
            criteria = headers[criteria]
        print(f"{indent}{criteria}: {tree.value}?")

        # Print the branches
        print(f"{indent}T->")
        print_tree(tree.tb, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.fb, headers, indent + "  ")


def print_trees(headers: List[str], data: Data):
    print("----- TREES -----")
    tree = build_tree(data)
    print("   - RECURSIVE -   ")
    print_tree(tree, headers)
    print("")
    print("   - ITERATIVE -   ")
    it_tree = iterative_build_tree(data)
    print_tree(it_tree, headers)
    print("   - PRUNED TREE -   ")
    prune_tree(tree, 0.8)
    print_tree(tree, headers)


def print_data(headers: List[str], data: Data):
    col_size = 15
    print('-' * ((col_size + 1) * len(headers) + 1))
    print("|", end="")
    for header in headers:
        print(header.center(col_size), end="|")
    print("")
    print('-' * ((col_size + 1) * len(headers) + 1))
    for row in data:
        print("|", end="")
        for value in row:
            if isinstance(value, (int, float)):
                print(str(value).rjust(col_size), end="|")
            else:
                print(value.ljust(col_size), end="|")
        print("")
    print('-' * ((col_size + 1) * len(headers) + 1))


def predict_data(data: Data, test_size: Union[float, int] = 0.2):
    print("----- PREDICTIONS -----")
    train, test = evaluation.train_test_split(data, test_size)
    tree = build_tree(train)
    for row in test:
        prediction = classify(tree, row[:-1])
        print("Prediction for ", row, "is: ", prediction)


def testing(data: Data, test_size: Union[float, int] = 0.2):
    print("----- TESTING -----")
    train, test = evaluation.train_test_split(data, test_size)
    tree = build_tree(train)
    print("Data split between train and test with ", test_size, " test size")
    train_accuracy = evaluation.get_accuracy(tree, train)
    print("Accuracy with training data: " + "{:.2f}".format(train_accuracy * 100) + " %")
    test_accuracy = evaluation.get_accuracy(tree, test)
    print("Accuracy with testing data: " + "{:.2f}".format(test_accuracy * 100) + " %")


def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "iris.csv"

    # header, data = read(filename)
    # print_data(header, data)

    # print(unique_counts(data))

    # print(gini_impurity(data))
    # print(gini_impurity([]))
    # print(gini_impurity([data[0]]))

    # print(entropy(data))
    # print(entropy([]))
    # print(entropy([data[0]]))

    headers, data = read(filename)
    print_data(headers, data)

    print_trees(headers, data)
    predict_data(data, 0.5)
    testing(data, 0.5)


if __name__ == "__main__":
    main()
