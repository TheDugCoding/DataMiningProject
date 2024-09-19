import numpy as np
import random
from collections import Counter

def best_split(x, y):
    best_impurity_reduction = 1.1
    best_value = 0
    if len(x) == len(y):
        sorted_values = np.sort(np.unique(x[:, 3]))
        print(sorted_values)
        for value_index in range(len(sorted_values-1)):
            #follows the x < c instructions, the variable avg is the average of two consecutive numbers
            avg = sum(sorted_values[value_index:value_index+2])/len(sorted_values[value_index:value_index+2])
            #select all the indexes where x < c (left child), then select indexes for the right child
            indexes_left_child = [i for i, value in enumerate(x) if value <= avg]
            indexes_right_child = list(set(range(len(x))) - set(indexes_left_child))
            #calculate gini index for the current split, for both children
            gini_index_left_child = gini_index_calc(y[indexes_left_child])
            gini_index_right_child = gini_index_calc(y[indexes_right_child])
            #calculate impurity reduction, lecture 2 slide 12
            impurity_reduction = gini_index_calc(y) - (len(y[indexes_left_child])/len(y) * gini_index_left_child + len(y[indexes_right_child])/len(y) * gini_index_right_child)
            if impurity_reduction < best_impurity_reduction:
                best_impurity_reduction = impurity_reduction
                best_value = avg
        return best_value, best_impurity_reduction
    else:
        raise ValueError("Arrays must have the same size")
    
def gini_index_calc(x):
    gini_index = 1
    for class_name, value in Counter(x).items():
        gini_index *= (value/len(x))
    print(gini_index)
    return gini_index

def impurity(x):
    sum = 0
    for i in x:
        sum += i
    prob_0 = sum/len(x)
    prob_1 = 1-prob_0
    return prob_0 * prob_1

def tree_grow(x, y, nmin, minleaf, nfeat):
    nodelist = x
    tree = {}
    level = 0
    # possible nodes to check must exist
    while len(nodelist) > 0:
        for i in nodelist:
            # remove current node from nodes to check
            nodelist = nodelist.remove(i)
            # check if impurity of current node is not 0, else it cannot be split and is leaf node
            if impurity(i) > 0:
                if len(i) >= nmin:
                    # randomly select n number of candidate splits
                    candidate_splits = random.sample(i, nfeat)
                    # calculate best split and impurity reduction
                    child_node_left, child_node_right, reduction = best_split(candidate_splits, y)
                    # add child nodes to be checked to nodelist
                    nodelist = nodelist.append(child_node_left, child_node_right)
                # return nodes in tree (WIP)
                tree[level] = child_node_left, child_node_right
            else:
                tree[level] = i
    return tree

def tree_pred():
    print('tree')

data_matrix = [[1,0,1,1],[1,0,0,1],[0,1,0,1]]
labels = [[0,1],[1,1]]
tree_grow(data_matrix, labels, 2, 2, 2)
