""" 
Arda Canser Adali (0385433)
Keren Dogan (1696637)
Sofia Bianchini (7273746)
"""

import numpy as np
import pandas as pd
import random
import math
from collections import Counter

class Node():
    def __init__(self, index=None, split=None, leaf_class=None):
        self.index = index
        self.split = split
        self.leaf_class = leaf_class
        self.left = None
        self.right = None

def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    Name: tree_grow
    Inputs:
        x: 2-D Numpy array of individuals with attributes.
        y: 1-D array of class labels.
        nmin: Minimum number of individuals at a node to consider splitting.
        minleaf: Minimum number of individuals required to form a leaf node.
        nfeat: Number of features to consider for splitting.
    Returns:
        A node object representing a node of the decision tree.
    Description: Grows a decision tree by recursively finding the best splits until stopping conditions are met.
    """
    l = len(y)
    splits = []
    impurity_reductions = []
    feature = None
    best_red = 0
    best_split = None
    if l < nmin:
        leaf_class = np.argmax(np.bincount(y))
        return Node(leaf_class=leaf_class)

    n_f = range(x.shape[1])
    if nfeat < len(n_f):
        n_f = np.random.choice(n_f, nfeat, replace=False)

    for col in n_f:
        split, impurity_reduction = bestsplit(x[:, col], y)
        if split is None or impurity_reduction == 0:
            continue
        splits.append(split)
        impurity_reductions.append(impurity_reduction)

        left_branch = y[x[:, col] >= split]
        right_branch = y[x[:, col] < split]
        reduction = weighted_impurity(y, left_branch, right_branch)

        if best_red < reduction:
            best_red = reduction
            best_split = split
            feature = col

    if best_split is None or not isinstance(best_split, (int, float)) or best_red == 0:
        leaf_class = np.argmax(np.bincount(y))
        return Node(leaf_class=leaf_class)

    node = Node(index=feature, split=best_split)
    left_id = x[:, feature] >= best_split
    right_id = x[:, feature] < best_split

    if len(x[left_id]) >= minleaf:
        node.left = tree_grow(x[left_id], y[left_id], nmin, minleaf, nfeat)
    else:
        leaf_class = np.argmax(np.bincount(y))
        node.left = Node(leaf_class=leaf_class)

    if len(x[right_id]) >= minleaf:
        node.right = tree_grow(x[right_id], y[right_id], nmin, minleaf, nfeat)
    else:
        leaf_class = np.argmax(np.bincount(y))
        node.right = Node(leaf_class=leaf_class)

    return node

def tree_pred(x, tree):
    """
    Name: tree_pred
    Inputs:
        x: A 2-D Numpy array of individuals.
        tree: The root node of the decision tree.
    Returns:
        A vector with the predicted labels of the cases in the input array
    Description: Makes predictions for each individual in the input array using the given decision tree.
    """
    predictions = []
    for X in x:
        pred = tree_pred_sing(tree, X)
        predictions.append(pred)

    return predictions

def tree_grow_b(X, y, nmin, minleaf, nfeat, m):
    """
    Name: tree_grow_b
    Inputs:
        x: 2-D Numpy array of individuals with attributes.
        y: 1-D array of class labels.
        nmin: Minimum number of individuals at a node to consider splitting.
        minleaf: Minimum number of individuals required to form a leaf node.
        nfeat: Number of features to consider for splitting.
        m: Number of bootstrap samples to create.
    Returns:
        list: A list of Node objects representing the trained decision trees.
    Description: Trains an ensemble of decision trees using bootstrap sampling.
    """
    models = []
    n_samples = X.shape[0]
    
    for _ in range(m):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_sample, y_sample = X[indices], y[indices]
        tree = tree_grow(X_sample, y_sample, nmin, minleaf, nfeat)
        models.append(tree)
    
    return models

def tree_pred_b(x, forest):
    """
    Name: tree_pred_b
    Inputs:
        x: 2-D Numpy array of individuals to make predictions on.
        forest: A list of decision trees.
    Returns:
        predictions: An array of predicted class labels for each individual in x.
    Description: Makes predictions for each individual in the input array using an ensemble of decision trees.
    """
    predictions = []
    for i in range(x.shape[0]):
        votes = []
        for tree in forest:
            vote = tree_pred_sing(tree, x[i])
            votes.append(vote)

        votes_count = Counter(votes)
        most_common_class = votes_count.most_common(1)[0][0]
        predictions.append(most_common_class)

    return np.array(predictions)

def gini_impurity(x):
    """
    Name: gini_impurity
    Inputs:
        An array of class labels.
    Returns:
        gini: Gini impurity of the input array.
    Description: Calculates the probability of each class and uses this formula for the gini index: 1 - sum (p)^2 (generalized for more than binary classes)
    """
    l = len(x)
    if l == 0:
        return 0.0
    classes = np.unique(x)
    probs = np.array([np.sum(x == label) / l for label in classes])
    gini = 1 - np.sum(probs ** 2)
    if len(classes) == 1:
        gini = 0.0

    return gini

def bestsplit(x, y):
    """
    Name: bestsplit
    Inputs:
        x: 1-D array of feature values.
        y: 1-D array of class labels.
    Returns:
        best_split: A float showing the best split point.
        best_impurity_reduction: A float showing the impurity reduction
    Description: Finds the best split point for a feature based on Gini impurity reduction.
    """
    income_sorted = np.sort(np.unique(x))
    income_splitpoints = (income_sorted[:-1] + income_sorted[1:]) / 2
    total_impurity = gini_impurity(y)
    best_impurity_reduction = 0
    best_split = 0
    for i in income_splitpoints:
        first_branch = y[x >= i]
        second_branch = y[x < i]
        weighted_impurity = (len(first_branch) / len(y)) * gini_impurity(first_branch) + (len(second_branch) / len(y)) * gini_impurity(second_branch)
        impurity_reduction = total_impurity - weighted_impurity
        if impurity_reduction > best_impurity_reduction:
            best_impurity_reduction = impurity_reduction
            best_split = i
            
    return best_split, best_impurity_reduction

def weighted_impurity(parent, left_branch, right_branch) -> float:
    """
    Name: weighted_impurity
    Inputs:
        parent: 1-D array of class labels for the parent node.
        left_branch: 1-D array of class labels for the left child node.
        right_branch: 1-D array of class labels for the right child node.
    Returns:
        impurity_reduction: The amount of impurity reduction after the split.
    Description: Calculates the weighted impurity reduction resulting from a split based on Gini impurity.
    """
    total_impurity = gini_impurity(parent)
    left_branch_imp = gini_impurity(left_branch)
    right_branch_imp = gini_impurity(right_branch)
    
    weighted_impurity = (len(left_branch) / len(parent)) * left_branch_imp + (len(right_branch) / len(parent)) * right_branch_imp
    impurity_reduction = total_impurity - weighted_impurity

    return impurity_reduction

def print_tree(node, level=0):
    """
    Name: print_tree
    Inputs:
        node : The root node of the decision tree to print.
        level: The current level in the tree (used for indentation).
    Returns:
        This function prints the tree structure.
    Description: Recursively prints the structure of the decision tree, including feature splits and class labels.
    """
    if node.leaf_class is not None:
        print("    " * level + f"Leaf: Class: {node.leaf_class}")
        return
    
    print("    " * level + f"Feature #{node.index}, split at {node.split}")
    
    if node.left:
        print("    " * (level + 1) + "Left:")
        print_tree(node.left, level + 2)
    
    if node.right:
        print

def tree_pred_sing(tr,x):
    """
    Name: tree_pred_sing
    Inputs:
        tr: A node of the decision tree
        x: 1-D array of feature values
    Returns:
        tr.leaf_class: if the stopping criteria is met, the predicted class label corresponding to the leaf node is returned
    Description:
        Recursively searches through the leaf nodes and returns the predicted class label for the array of feature values
    """
    if tr.leaf_class is not None:
        return tr.leaf_class
    if x[tr.index] >= tr.split: 
        return tree_pred_sing(tr.left, x)
    else:
        return tree_pred_sing(tr.right, x)