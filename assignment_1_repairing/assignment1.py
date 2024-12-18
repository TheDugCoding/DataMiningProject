"""
Annebelle Olminkhof
4822048
a.n.olminkhof@students.uu.nl
Luca Dughera
1738380
l.dughera@students.uu.nl
Riccardo Campanella
8175721
r.campanella@students.uu.nl
"""

import numpy as np
import pandas as pd
import statistics
from numpy import genfromtxt
import time


from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

"""
Uncomment this part to run the examples (need to download the datasets listed in the README file)
credit_data_with_headers = pd.read_csv('data/credit.txt', delimiter=',')
indians = pd.read_csv('data/pima.txt', delimiter=',', names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
"""

#True use multiprocessing, False don't use multiprocessing
MULTIPROCESSING = True

def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    :param x: rows of the dataset used for creating the tree
    :param y: rows of the target features used for creating the tree
    :param nmin: the number of observations that a node must contain at least, for it to be allowed to be split
    :param minleaf: minimum number of observations required for a leaf node
    :param nfeat: number of features that should be considered for each split
    :return: the function returns a binary tree
    """
    if x.size > 0:
        root = Node(np.arange(x.shape[0]))
        nodelist = [root]
        leaves = []
        i = 0

        # tree grow stops when we split all the nodes
        while nodelist:
            # visit all the nodes in the list, for optimization we don't use 'pop' we just iterate over the nodes
            current_node = nodelist.pop(0)

            # store the node instances
            current_node_instances = current_node.instances

            # store node in the tree before splitting
            labels = y[current_node_instances]

            # avoid splitting leaf nodes with zero impurity and check that there are enough observations for a split
            if impurity(labels) > 0 and len(current_node.instances) >= nmin:

                # random sample nfeat number of columns
                if nfeat < x.shape[1]:
                    candidate_features = np.random.choice(x.shape[1], size=nfeat, replace=False)
                else:
                    candidate_features = np.arange(nfeat)

                # calculate best split and impurity reduction to get child nodes, (if a split is not found feature = None)
                left, right, feature, threshold = best_split(x[np.ix_(current_node_instances, candidate_features)], labels, minleaf)

                # store current node info, if it is not a leaf
                if feature is not None:
                    current_node.left = Node(current_node_instances[left])
                    current_node.right = Node(current_node_instances[right])
                    # update list
                    nodelist.append(current_node.left)
                    nodelist.append(current_node.right)
                    current_node.threshold = threshold
                    current_node.feature = candidate_features[feature]
                else:
                    leaves.append(current_node)
                current_node.class_distribution = np.bincount(labels.astype(int))
                current_node.predicted_class = np.argmax(current_node.class_distribution)


            else:
                # return the final prediction of the leaf node
                current_node.class_distribution = np.bincount(labels.astype(int))
                current_node.predicted_class = np.argmax(current_node.class_distribution)
                leaves.append(current_node)
        return Tree(root, leaves)
    else:
        raise ValueError("x is empty")

def tree_pred(x, tr):
    """
    :param x: rows of the dataset used to make a prediction
    :param tr: the tree that we are using for the prediction
    :return: a single dimensional array containing the prediction of the tree
    """
    predicted_labels = []
    for index, row in enumerate(x):
        current_node = tr.root
        # a leaf node doesn't contain a feature
        while current_node.feature is not None:
            # left branch
            if row[current_node.feature] < current_node.threshold:
                current_node = current_node.left
            else:
                # right branch
                current_node = current_node.right
        predicted_labels.append(current_node.predicted_class)

    return predicted_labels

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    """
    :param x: rows of the dataset used for creating the tree
    :param y: name of the feature used for classification
    :param nmin: the number of observations that a node must contain at least, for it to be allowed to be split.
    :param minleaf: minimum number of observations required for a leaf node
    :param nfeat: number of features used to select the best split
    :param m: number of trees to be created
    :return: the function return a list of trees
    """
    trees = []
    random_num = np.arange(x.shape[0])
    random_indexes_with_replacement = np.random.choice(random_num, size=(m,len(x)), replace=True)

    if MULTIPROCESSING:
        with ProcessPoolExecutor() as executor:
            futures = []
            # using parallelization to speed up the process, in case parallelization doesn't work change value of the variable MULTIPROCESSING to False
            for i in range(m):
                future = executor.submit(tree_grow, x[random_indexes_with_replacement[i]], y[random_indexes_with_replacement[i]], nmin, minleaf, nfeat)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures)):
                trees.append(future.result())

    else:
        for i in tqdm(range(m)):
            tree = tree_grow(x[random_indexes_with_replacement[i]], y[random_indexes_with_replacement[i]], nmin, minleaf, nfeat)
            trees.append(tree)


    return trees

def tree_pred_b(x, tr):
    """
    :param x: rows of the dataset used to make a prediction
    :param tr: a list of trees that we are using for predictions
    :return: a list containing the final predictions, where each prediction is made
    by selecting the class that received the most votes.
    """
    majority_votes = {}
    predicted_labels = []

    for tree in tqdm(tr, desc="Processing Trees Predictions", unit="tree"):
        predicted_labels.append(tree_pred(x, tree))

    # loop over the list of predicted labels (one list for each tree)
    for tree_predictions in tqdm(predicted_labels, desc="Processing Predictions", unit="set"):
        # loop over the individual predictions in a tree
        for i in range(len(tree_predictions)):
            if i not in majority_votes:
                majority_votes[i] = 0
            # add 1 for '1', subtract 1 for '0'
            majority_votes[i] += 1 if tree_predictions[i] == 1 else -1

    # convert the majority votes dictionary to a list of predictions
    final_predictions = [1 if majority_votes[i] > 0 else 0 for i in range(len(majority_votes))]

    return final_predictions

def best_split(x, y, minleaf):
    """
    :param x: rows of the dataset used to select the best split
    :param y: rows of the target variable
    :param minleaf: minimum number of observations required for a leaf node
    :return: the functions returns the indexes of the elements that belong to the left and right children nodes
    (if they are not leaves node), the feature used for the split, and the value used for splitting the node
    """
    best_impurity_reduction = float('-inf')
    best_value = 0
    best_left_child_indexes = []
    best_right_child_indexes = []
    best_split = None
    impurity_father = impurity(y)
    elements_in_y = len(y)

    if len(x) == elements_in_y:
        for split in np.arange(x.shape[1]):
            # check how many unique values there are
            split_values = x[:, split]
            sorted_values = np.unique(split_values)


            #check that we have enough different values for a split
            if len(sorted_values) > 1:
                for value_index in range(len(sorted_values) -1):
                    # follows the x < c instructions, the variable avg is the average of two consecutive numbers
                    avg = (sorted_values[value_index] + sorted_values[value_index + 1]) / 2
                    # select all the indexes where x < c (left child), then select indexes for the right child
                    mask = split_values < avg
                    indexes_left_child = np.where(mask)[0]
                    indexes_right_child = np.where(~mask)[0]
                    if len(indexes_left_child) >= minleaf and len(indexes_right_child) >= minleaf:
                        # calculate impurity reduction
                        impurity_reduction = impurity_father - (
                                ((len(y[indexes_left_child]) / elements_in_y) * impurity(y[indexes_left_child])) +
                                ((len(y[indexes_right_child]) / elements_in_y) * impurity(y[indexes_right_child])))
                        # if the impurity reduction obtained with these values is the best one yet, save it
                        if impurity_reduction > best_impurity_reduction:
                            best_impurity_reduction = impurity_reduction
                            best_split = split
                            best_value = avg
                            best_left_child_indexes = indexes_left_child
                            best_right_child_indexes = indexes_right_child
        return best_left_child_indexes, best_right_child_indexes, best_split, best_value
    else:
        raise ValueError("Arrays must have the same size")

def impurity(x):
    """
    :param x: A series of values, the values must be 0 or 1
    :return: the functions returns the Gini impurity
    """
    if len(x) > 0:
        prob_0 = statistics.fmean(x)
        prob_1 = 1-prob_0
        return prob_0 * prob_1
    else:
        return 0

class Node:
    def __init__(self, instances, feature=None, threshold=None, left=None, right=None, predicted_class=None, class_distribution=None):
        self.instances = instances
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_distribution = class_distribution
        self.predicted_class = predicted_class

class Tree:
    def __init__(self, root = None, leaves = None):
        self.root = root
        self.leaves = leaves

# Function for testing single tree

def single_test(x, y, nmin, minleaf, nfeat, n):
    acc = np.zeros(n)
    for i in range(0, n):
        tr = tree_grow(x, y, nmin, minleaf, nfeat)
        pred = tree_pred(x, tr)
        acc[i] = sum(pred == y) / len(y)
    return [np.mean(acc), np.std(acc)]


# Function for testing bagging/random forest

def rf_test(x, y, nmin, minleaf, nfeat, m, n):
    acc = np.zeros(n)
    for i in range(0, n):
        tr_list = tree_grow_b(x, y, nmin, minleaf, nfeat, m)
        pred = tree_pred_b(x, tr_list)
        acc[i] = sum(pred == y) / len(y)
    return [np.mean(acc), np.std(acc)]


if __name__ == '__main__':

    # Basic test on credit data. Prediction should be perfect.

    credit_data = genfromtxt('data/credit.txt', delimiter=',', skip_header=True)
    credit_x = credit_data[:, 0:5]
    credit_y = credit_data[:, 5]
    credit_tree = tree_grow(credit_x, credit_y, 2, 1, 5)
    credit_pred = tree_pred(credit_x, credit_tree)
    print(pd.crosstab(np.array(credit_y), np.array(credit_pred)))

    # Single tree on pima data

    pima_data = genfromtxt('data/pima.txt', delimiter=',')
    pima_x = pima_data[:, 0:8]
    pima_y = pima_data[:, 8]
    pima_tree = tree_grow(pima_x, pima_y, 20, 5, 8)
    pima_pred = tree_pred(pima_x, pima_tree)

    # confusion matrix should be: 444,56,54,214 (50/50 leaf nodes assigned to class 0)
    # or: 441,59,51,217 (50/50 leaf nodes assigned to class 1)

    print(pd.crosstab(np.array(pima_y), np.array(pima_pred)))

    # Compute average and standard deviation of accuracy for single tree

    print(single_test(pima_x, pima_y, 20, 5, 2, 25))
    print(single_test(pima_x, pima_y, 20, 5, 8, 25))

    # Compute average and standard deviation of accuracy for bagging/random forest

    print(rf_test(pima_x, pima_y, 20, 5, 2, 25, 25))
    print(rf_test(pima_x, pima_y, 20, 5, 8, 25, 25))

    # Measure time for training and prediction with random forest

    start = time.time()
    print(rf_test(pima_x, pima_y, 20, 5, 8, 25, 25))
    end = time.time()
    print("The execution time is :", (end - start), "seconds")







