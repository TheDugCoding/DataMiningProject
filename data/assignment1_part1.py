# Part 1 of assignment 1 which needs to be handed in
import numpy as np
import pandas as pd
import statistics

from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

data = pd.read_csv('data/credit.txt', delimiter=',')

class Node:
    def __init__(self, instances, feature=None, threshold=None, left=None, right=None, predicted_class=None):
        self.instances = instances
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.predicted_class = predicted_class

class Tree:
    def __init__(self, root = None, leaves = None):
        self.root = root
        self.leaves = leaves

def impurity(x):
    """
    :param x: A series of values, the values must be 0 or 1
    :return: the functions returns the Gini impurity
    """
    if len(x) > 0:
        sum = 0
        for i in x:
            sum += i
        prob_0 = sum/len(x)
        prob_1 = 1-prob_0
        return prob_0 * prob_1
    else:
        return 0
    
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
    best_split = ''
    impurity_father = impurity(y)
    elements_in_y = len(y)

    if len(x) == elements_in_y:
        for split in x.columns:
            # check how many unique values there are
            sorted_values = np.sort(np.unique(x[split]))
            split_values = x[split]

            #check that we have enough different values for a split
            if len(sorted_values) > 1:
                for value_index in range(len(sorted_values) -1):
                    # follows the x < c instructions, the variable avg is the average of two consecutive numbers
                    avg = (sorted_values[value_index] + sorted_values[value_index + 1]) / 2
                    # select all the indexes where x < c (left child), then select indexes for the right child
                    indexes_left_child = split_values[split_values <= avg].index
                    indexes_right_child = split_values[split_values > avg].index
                    if len(indexes_left_child) > minleaf and len(indexes_right_child) > minleaf:
                        # calculate impurity reduction
                        impurity_reduction = impurity_father - (
                                ((len(y[indexes_left_child]) / elements_in_y) * impurity(y[indexes_left_child])) +
                                ((len(y[indexes_right_child]) / elements_in_y) * impurity(y[indexes_right_child])))
                        # if the impurity reduction obtained with this values is the best one yet, save it
                        if impurity_reduction > best_impurity_reduction:
                            best_impurity_reduction = impurity_reduction
                            best_split = split
                            best_value = avg
                            best_left_child_indexes = indexes_left_child
                            best_right_child_indexes = indexes_right_child
        return best_left_child_indexes, best_right_child_indexes, best_split, best_value
    else:
        raise ValueError("Arrays must have the same size")

def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    :param x: rows of the dataset used for creating the tree
    :param y: rows of the target features used for creating the tree
    :param nmin: the number of observations that a node must contain at least, for it to be allowed to be split
    :param minleaf: minimum number of observations required for a leaf node
    :param nfeat: number of features that should be considered for each split
    :return: the function returns a binary tree
    """
    if not x.empty:
        root = Node(x.index)
        nodelist = [root]
        leaves = []

        # tree grow stops when we split all the nodes, the nodes that cannot be split are removed from the list
        while nodelist:
            # visit the first node
            current_node = nodelist[0]

            # store the node instances
            current_node_instances = current_node.instances

            # store node in the tree before splitting
            labels = y.iloc[current_node_instances]

            nodelist.pop(0)

            # avoid splitting leaf nodes with zero impurity and check that there are enough observations for a split
            if impurity(labels) > 0 and len(current_node.instances) >= nmin:

                # random sample nfeat number of columns
                candidate_features = np.random.choice(x.columns, size=nfeat, replace=False)

                # calculate best split and impurity reduction to get child nodes
                left, right, feature, threshold = best_split(x.loc[current_node_instances, candidate_features], labels, minleaf)

                # store current node info
                if feature:
                    current_node.left = Node(left)
                    current_node.right = Node(right)
                    # update list
                    nodelist.append(current_node.left)
                    nodelist.append(current_node.right)
                    current_node.threshold = threshold
                    current_node.feature = feature
                else:
                    leaves.append(current_node)
                current_node.predicted_class = statistics.mode(labels)

            else:
                # return the final prediction of the leaf node
                current_node.predicted_class = statistics.mode(labels)
                leaves.append(current_node)
        return Tree(root, leaves)
    else:
        raise ValueError("x is empty")
    
def tree_grow_b(x, target_feature, nmin, minleaf, nfeat, m):
    """
    :param x: rows of the dataset used for creating the tree
    :param target_feature: name of the feature used for classification
    :param nmin: the number of observations that a node must contain at least, for it to be allowed to be split.
    :param minleaf: minimum number of observations required for a leaf node
    :param nfeat: number of features used to select the best split
    :param m: number of trees to be created
    :return: the function return a list of trees
    """
    trees = []
    results = []
    random_indexes_with_replacement = np.random.choice(x.index.tolist(), size=(m,len(x)), replace=True)

    pool = Pool(processes=(cpu_count() - 1))

    # tqdm progress bar for asynchronous task completion
    pbar = tqdm(total=m, desc="Growing Trees", unit=" tree")

    def collect_result(result):
        """Callback to collect result and update progress bar."""
        trees.append(result)
        pbar.update()

    # using parallelization to speed up the process, in case parallelization doesn't work use the commented instruction instead
    for i in range(m):
        #trees.append(tree_grow(x.loc[random_indexes_with_replacement[i], x.columns != target_feature].reset_index(drop=True), x.loc[random_indexes_with_replacement[i], target_feature].reset_index(drop=True), nmin, minleaf, nfeat))
        pool.apply_async(tree_grow, args=(x.loc[random_indexes_with_replacement[i], x.columns != target_feature].reset_index(drop=True), x.loc[random_indexes_with_replacement[i], target_feature].reset_index(drop=True), nmin, minleaf, nfeat), callback=collect_result)
        #results.append(result)
    pool.close()
    pool.join()

    return trees

def tree_pred(x, tr):
    """
    :param x: rows of the dataset used to make a prediction
    :param tr: the tree that we are using for the prediction
    :return: a single dimensional array containing the prediction of the tree
    """
    predicted_labels = []
    for index, row in x.iterrows():
        current_node = tr.root
        # a leaf node doesn't contain a feature
        while current_node.feature:
            # left branch
            if row[current_node.feature] < current_node.threshold:
                current_node = current_node.left
            else:
                # right branch
                current_node = current_node.right
        predicted_labels.append(current_node.predicted_class)

    return predicted_labels

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

    # Loop over the list of predicted labels (one list for each tree)
    for tree_predictions in tqdm(predicted_labels, desc="Processing Predictions", unit="set"):
        # Loop over the individual predictions in a tree
        for i in range(len(tree_predictions)):
            if i not in majority_votes:
                majority_votes[i] = 0
            # Add 1 for '1', subtract 1 for '0'
            majority_votes[i] += 1 if tree_predictions[i] == 1 else -1

    # Convert the majority_votes dictionary to a list of predictions
    final_predictions = [1 if majority_votes[i] > 0 else 0 for i in range(len(majority_votes))]

    return final_predictions
    
if __name__ == '__main__':
    single_tree = tree_grow(data.loc[:, data.columns != 'class'], data['class'], 2, 2, 5)
    ensamble_tree = tree_grow_b(data, 'class', 2, 2, 5, 10)

    #test prediction
    print('\n\n--prediction single tree')
    print(tree_pred(data.loc[:, data.columns != 'class'], single_tree))

    #test prediction_b
    print('\n\n--prediction all trees')
    predictions = tree_pred_b(data.loc[:, data.columns != 'class'].iloc[-2:], ensamble_tree)
