import numpy as np
import random
from collections import Counter
import pandas as pd

credit_data_with_headers = pd.read_csv('data/credit.txt', delimiter=',')

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
                    best_right_child_indexes = list(set(x[split].index)- set(best_left_child_indexes))
                    best_value = sorted_values[0]
                    gini_index_left_child = gini_index_calc(y[best_left_child_indexes])
                    gini_index_right_child = gini_index_calc(y[best_right_child_indexes])
                    best_impurity_reduction = gini_index_calc(y) - (
                            len(y[best_left_child_indexes]) / len(y) * gini_index_left_child + len(
                        y[best_right_child_indexes]) / len(y) * gini_index_right_child)
                else:
                    for value_index in range(len(sorted_values - 1)):
                        # follows the x < c instructions, the variable avg is the average of two consecutive numbers
                        avg = sum(sorted_values[value_index:value_index + 2]) / len(
                            sorted_values[value_index:value_index + 2])
                        # select all the indexes where x < c (left child), then select indexes for the right child
                        indexes_left_child = x[split][x[split] <= avg].index.tolist()
                        indexes_right_child = list(set(x[split].index)- set(indexes_left_child))
                        # calculate gini index for the current split, for both children
                        gini_index_left_child = gini_index_calc(y[indexes_left_child])
                        gini_index_right_child = gini_index_calc(y[indexes_right_child])
                        # calculate impurity reduction, lecture 2 slide 12
                        impurity_reduction = gini_index_calc(y) - (
                                len(y[indexes_left_child]) / len(y) * gini_index_left_child + len(
                            y[indexes_right_child]) / len(y) * gini_index_right_child)
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
        return best_left_child_indexes_overall, best_right_child_indexes_overall
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

def tree_grow(x, y, nmin, minleaf, nfeat):
    node = 1
    nodelist = [x]
    Tree = {}
    # possible nodes to check must exist
    while len(nodelist) > 0:
        current_node = nodelist[0]
        classes = y.iloc[current_node.index]
        # classes = classes.to_list()
        nodelist.pop(0)
        # check if impurity of class labels is not 0, else it cannot be split and is leaf node
        if impurity(classes) > 0:
            if current_node.shape[0] >= nmin:
                # randomly select nfeat number of columns
                candidate_splits = current_node.sample(n=nfeat, axis='columns')
                # calculate best split and impurity reduction to get child nodes
                child_node_left, child_node_right = best_split(candidate_splits, classes, minleaf)
                # add rows of child nodes to be checked to nodelist
                nodelist.append(x.iloc[child_node_left])
                nodelist.append(x.iloc[child_node_right])
                if (len(child_node_left) + len(child_node_right)) == 0:
                    Tree[node, "leaf"] = current_node.index.to_list()
                else:
                    Tree[node] = current_node.index.to_list()
                node += 1
        else:
            if len(current_node) > 0:
                Tree[node, "leaf"] = current_node.index.to_list()
                node += 1
    return Tree

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    # assignment states trees must be in list
    trees = []
    for i in range(m):
        trees.append(tree_grow(x, y, nmin, minleaf, nfeat))
    return trees

def tree_pred():
    print('tree')

#print(best_split(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data_with_headers['class'], 2))
Tree = tree_grow_b(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data_with_headers['class'], 2, 2, 2, 6)
print(Tree)
