import random
import treepredict
from typing import Union, List, Callable

Data = List[List]


def train_test_split(dataset, test_size: Union[float, int], seed=None):
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(test_size) != int(test_size):
        test_size = int(n_rows * test_size)  # We need an integer number of rows

    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=test_size)

    test = [row for (i, row) in enumerate(dataset) if i in test_rows]
    train = [row for (i, row) in enumerate(dataset) if i not in test_rows]

    return train, test


def get_accuracy(classifier: treepredict.DecisionNode, dataset: List):
    correct = 0
    for row in dataset:
        if treepredict.classify(classifier, row[:-1]) == row[-1]:
            correct += 1
    return correct / len(dataset)


def mean(values: List[float]):
    return sum(values) / len(values)


def cross_validation(dataset: Data = treepredict.Data, k: int = 1, agg: Callable = mean, scoref = treepredict.entropy, beta = 0, threshold = 0, seed: int = None) -> int:
    if seed:
        random.seed(seed)
    random.shuffle(dataset)  # shuffle the dataset for randomization of the data
    partitions = _create_partitions(dataset, k)
    accuracy_scores = []
    for part in range(k):
        train_data = [data for j, partition in enumerate(partitions) for data in partition if j != part]
        test_data = partitions[part]
        decision_tree = treepredict.build_tree(train_data, scoref, beta)
        # pruning.prune_tree(decision_tree, threshold)
        accuracy_scores.append(get_accuracy(decision_tree, test_data))
    return agg(accuracy_scores)


def _create_partitions(dataset, k):
    partition_size = len(dataset) // k
    partitions = [dataset[i * partition_size:(i + 1) * partition_size] for i in range(k)]
    return partitions
