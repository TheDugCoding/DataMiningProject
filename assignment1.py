import numpy as np
import random
from collections import Counter
import pandas as pd

credit_data_with_headers = pd.read_csv('data/credit.txt', delimiter=',')

def best_split(x,y, minleaf):
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
            for value_index in range(len(sorted_values - 1)):
                # follows the x < c instructions, the variable avg is the average of two consecutive numbers
                avg = sum(sorted_values[value_index:value_index + 2]) / len(sorted_values[value_index:value_index + 2])
                # select all the indexes where x < c (left child), then select indexes for the right child
                indexes_left_child = [i for i, value in enumerate(x[split]) if value <= avg]
                indexes_right_child = list(set(range(len(x[split]))) - set(indexes_left_child))
                # calculate gini index for the current split, for both children
                gini_index_left_child = gini_index_calc(y[indexes_left_child])
                gini_index_right_child = gini_index_calc(y[indexes_right_child])
                # calculate impurity reduction, lecture 2 slide 12
                impurity_reduction = gini_index_calc(y) - (
                            len(y[indexes_left_child]) / len(y) * gini_index_left_child + len(
                        y[indexes_right_child]) / len(y) * gini_index_right_child)
                if impurity_reduction < best_impurity_reduction and len(indexes_left_child)>minleaf and len(indexes_left_child)<minleaf:
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
        return best_value_overall, best_split_overall, best_left_child_indexes_overall, best_right_child_indexes_overall
    else:
        raise ValueError("Arrays must have the same size")
    
def gini_index_calc(x):
    gini_index = 1
    for class_name, value in Counter(x).items():
        gini_index *= (value/len(x))
    return gini_index

def impurity(x):
    sum = 0
    for i in x:
        sum += i
    prob_0 = sum/len(x)
    prob_1 = 1-prob_0
    return prob_0 * prob_1

def tree_grow(x, y, nmin, minleaf, nfeat):
    nodelist = [x]
    Tree = []
    # possible nodes to check must exist
    while len(nodelist) > 0:
        current_node = nodelist[0]
        nodelist = nodelist.remove(current_node)
        print(nodelist)
        # check if impurity of class labels is not 0, else it cannot be split and is leaf node
        if impurity(y) > 0:
            if current_node.index.size >= nmin:
                # randomly select nfeat number of columns
                candidate_splits = current_node.sample(n=nfeat, axis='columns')
                # calculate best split and impurity reduction
                value, split, child_node_left, child_node_right = best_split(candidate_splits, y, minleaf)
                # add child nodes to be checked to nodelist
                nodelist = nodelist.append(child_node_left, ignore_index=True)
                nodelist = nodelist.append(child_node_right, ignore_index=True)
        else:
            Tree.append(current_node)
    return Tree

def tree_pred():
    print('tree')

print(best_split(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data_with_headers['class']))
tree_grow(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data_with_headers['class'], 2, 2, 2)
# tree_grow(data_matrix, labels, 2, 2, 2)