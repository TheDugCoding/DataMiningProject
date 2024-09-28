import numpy as np
import random
from collections import Counter
import pandas as pd
import statistics

credit_data_with_headers = pd.read_csv('data/credit.txt', delimiter=',')

def impurity_reduction_calc(y, indexes_left_child, indexes_right_child):
    return gini_index_calc(y) - (
            len(y[indexes_left_child]) / len(y) * gini_index_calc(y[indexes_left_child]) + len(
        y[indexes_right_child]) / len(y) * gini_index_calc(y[indexes_right_child]))

def count_class_occurences(x, indexes):
    if len(x) > 0:
        number_of_items_per_class = sum(x[i] == 1 for i in indexes)
        # return 1 if more than half of the elements are 1, else return 0
        return 1 if number_of_items_per_class > len(indexes) / 2 else 0
    else:
        raise ValueError("Dataset is empty")

def best_split(x, y, minleaf):
    best_impurity_reduction_overall = float('inf')
    best_value_overall = 0
    best_split_overall = ''
    best_left_child_indexes_overall = []
    best_right_child_indexes_overall = []

    if len(x) == len(y):
        for split in x.columns:
            best_impurity_reduction = float('inf')
            best_value = 0
            best_left_child_indexes = []
            best_right_child_indexes = []
            sorted_values = np.sort(np.unique(x[split]))
            #check that we have enough different values for a split
            if len(sorted_values) > 1:
                # check if there are only 2 values, then we don't need to calculate the average
                if len(sorted_values) == 2:
                    best_left_child_indexes = x[split][x[split] == sorted_values[0]].index.tolist()
                    best_right_child_indexes = list(set(x[split].index) - set(best_left_child_indexes))
                    # check that both children have enough elements
                    if len(best_left_child_indexes) > minleaf and len(
                        best_right_child_indexes) > minleaf:
                        best_value = sorted_values[0]
                        #calculate impurity reduction
                        best_impurity_reduction = impurity_reduction_calc(y, best_left_child_indexes, best_right_child_indexes)
                else:
                    for value_index in range(len(sorted_values - 1)):
                        # follows the x < c instructions, the variable avg is the average of two consecutive numbers
                        avg = sum(sorted_values[value_index:value_index + 2]) / len(
                            sorted_values[value_index:value_index + 2])
                        # select all the indexes where x < c (left child), then select indexes for the right child
                        indexes_left_child = x[split][x[split] <= avg].index.tolist()
                        indexes_right_child = list(set(x[split].index)- set(indexes_left_child))
                        # calculate impurity reduction
                        impurity_reduction = impurity_reduction_calc(y, indexes_left_child, indexes_right_child)
                        if impurity_reduction < best_impurity_reduction and len(indexes_left_child) > minleaf and len(
                                indexes_right_child) > minleaf:
                            best_impurity_reduction = impurity_reduction
                            best_value = avg
                            best_left_child_indexes = indexes_left_child
                            best_right_child_indexes = indexes_right_child
                if best_impurity_reduction < best_impurity_reduction_overall:
                    best_impurity_reduction_overall = best_impurity_reduction
                    best_value_overall = best_value
                    best_split_overall = split
                    best_left_child_indexes_overall = best_left_child_indexes
                    best_right_child_indexes_overall = best_right_child_indexes
        return best_left_child_indexes_overall, best_right_child_indexes_overall, best_split_overall, best_value_overall
    else:
        raise ValueError("Arrays must have the same size")

def gini_index_calc(x):
    gini_index = 1
    for class_name, value in Counter(x).items():
        gini_index *= (value/len(x))
    return gini_index

def impurity(x):
    if len(x) > 0:
        sum = 0
        for i in x:
            sum += i
        prob_0 = sum/len(x)
        prob_1 = 1-prob_0
        return prob_0 * prob_1
    else:
        return 0

class Node:
    def __init__(self, instances, feature=None, threshold=None, left=[], right=[], predicted_class=None):
        self.instances = instances
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.predicted_class = predicted_class


def tree_grow(x, y, nmin, minleaf, nfeat):
    root = Node(x)
    nodelist = [root]

    # tree grow stops when we split all the nodes, the nodes that cannot be split are removed from the list
    while nodelist:
        # visit the first node
        current_node = nodelist[0]

        # store the node instances
        current_node_instances = current_node.instances

        # store node in the tree before splitting
        labels = y.iloc[current_node_instances.index]

        nodelist.pop(0)

        # avoid splitting leaf nodes with zero impurity
        if impurity(labels) > 0:

            # early stopping: pure node
            if current_node.instances.shape[0] >= nmin:
                print
                # random sample nfeat number of columns
                candidate_features = current_node.instances.sample(n=nfeat, axis='columns')

                # calculate best split and impurity reduction to get child nodes
                left, right, feature, threshold = best_split(candidate_features, labels, minleaf)

                # store current node info
                current_node.left = Node(x.iloc[left], feature, threshold)
                current_node.right = Node(x.iloc[right], feature, threshold)
                current_node.threshold = threshold
                current_node.feature = feature
                current_node.predicted_class = statistics.mode(labels)

                # update list
                nodelist.append(current_node.left)
                nodelist.append(current_node.right)

        elif len(current_node_instances) > 0:
            # return the final prediction of the leaf node
            current_node.predicted_class = statistics.mode(labels)

    return root

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    # assignment states trees must be in list
    trees = []
    for i in range(m):
        trees.append(tree_grow(x, y, nmin, minleaf, nfeat))
    return trees

def tree_pred(x, tr):
    predicted_labels = []
    for index, row in x.iterrows():
        current_node = tr
        while current_node.feature:
            if row[current_node.feature] < current_node.threshold:
                current_node = current_node.left
                print('left')
            else:
                current_node = current_node.right
                print('right')
        predicted_labels.append([index, current_node.predicted_class])

    return predicted_labels


#print(best_split(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data_with_headers['class'], 2))

single_tree = tree_grow(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data_with_headers['class'], 2, 2, 5)
print(single_tree)

ensamble_tree = tree_grow_b(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data_with_headers['class'], 2, 2, 5, 6)
print(ensamble_tree)

#test prediction
tree_pred(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'].iloc[-2:], single_tree)