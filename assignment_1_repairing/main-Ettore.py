### DATA MINING ASSIGNMENT 1 2024
### Dimitra Tsolka 1304585
### Ettore Cesari 1657399
### Samuele Milanese 1907816


import dataclasses
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
from tqdm import tqdm

MULTIPROCESSING = True


def impurity(x: np.ndarray) -> float:
    """
    Computes the impurity of a vector (of arbitrary length) of class labels using the gini-index as impurity measure
    for a classification problem with only 2 classes that are labeled 0 and 1 respectively.

    @param x: 1-dimensional Bernoulli array
    @return: Gini impurity of the array
    """
    # The impurity in this case is (1 - p_0t) * (1 - p_1t).
    # Since (1 - p_0t) * (1 - p_1t) == p_1t * (1. - p_1t), we can only compute p_1t.
    # Furthermore p_1t = len(x[x == 1]) / len(x).
    # Finally, since numpy implementation in C is faster than python
    # and len(x[x == 1]) == np.sum(x) for Bernoulli arrays,
    # we have p_1t = np.sum(x) / len(x), which is the mean of the array.
    p_1t = np.mean(x)
    return p_1t * (1. - p_1t)


def bestsplit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Computes the best split value on a numeric attribute x,
    assuming there are only two classes in y, coded as 0 and 1

    @param x: vector of numeric values
    @param y: Bernoulli vector of class labels
    @return: best split threshold and best split quality
    """
    x_values = np.unique(x)  # Get the sorted unique elements of x.
    candidates = (x_values[:-1] + x_values[1:]) / 2  # Splitpoint candidates
    splits = [(np.where(x <= c)[0], np.where(x > c)[0]) for c in candidates]
    qualities = np.array([
        impurity(y[low_split]) * len(low_split) +
        impurity(y[high_split]) * len(high_split)
        for low_split, high_split in splits
    ])  # * -1 / x.shape[0] + impurity(y)  # No need to compute to compare the qualities
    best_index = np.argmin(qualities)  # Using argmin instead of argmax because it's missing the '* -1'
    best_candidate = candidates[best_index]
    quality = impurity(y) - float(qualities[best_index]) / len(x)
    return best_candidate, quality


@dataclasses.dataclass(slots=True)
class Node:
    depth: int = 0  # Depth of the node in the tree
    is_leaf: bool = True  # Whether the node is a leaf
    nsamples: int = 0  # Number of samples managed
    prediction: int = None  # The prediction of the leaf node
    feature: int = None  # The feature on which the node splits
    threshold: float = None  # The threshold for which the node splits
    left: 'Node' = None  # The node on the left child branch
    right: 'Node' = None  # The node on the right child branch

    def split(self, feature: int, threshold: float) -> tuple['Node', 'Node']:
        """
        Splits the current node in two branches

        @param feature: the feature for which the split is to be done
        @param threshold: the threshold for which the split is to be done
        @return: a tuple with the children nodes
        """
        self.is_leaf = False
        self.feature = feature
        self.threshold = threshold
        self.left = Node()
        self.right = Node()
        self.left.depth = self.right.depth = self.depth + 1
        return self.left, self.right

    def __str__(self):
        if self.is_leaf:
            return f"{'  ' * self.depth}-> {self.prediction}\n"
        return (
            f"{'  ' * self.depth}{self.feature} < {self.threshold:.2f}\n"
            f"{self.left}"
            f"{self.right}"
        )


def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int) -> 'Node':
    """
    Creates a decision tree, assuming the attribute values are numerical and the target values are binary (0 or 1)

    @param x: 2-dimensional array containing the attribute values
    @param y: vector of class labels
    @param nmin: min number of indices to create a split
    @param minleaf: min number of leafs to create a split
    @param nfeat: number of features for random forest
    @return: root node of the decision tree
    """
    root = Node()
    y = y.astype(bool).astype(int)
    nodelist: list[tuple[Node, np.ndarray]] = [
        (root, np.arange(len(y)))  # Start splitting the whole dataset
    ]

    while nodelist:  # Stops when nodelist is empty
        node, indices = nodelist.pop(0)

        # Defaults the node to a leaf with the majority class as prediction
        targets = y[indices]
        majority_class = np.argmax(np.bincount(targets))
        node.prediction = majority_class
        node.nsamples = len(indices)

        # Checks the requirements to create a split
        if (node.nsamples < nmin
                or len(np.unique(targets)) == 1
                or impurity(targets) == 0.0):
            continue

        # Selects feature to check for the split
        if nfeat < x.shape[1]:
            features = np.random.choice(x.shape[1], nfeat, replace=False)
        else:
            features = np.arange(nfeat)

        # Finds the best feature to compute the split on
        best_feature = None
        best_threshold = None
        best_quality = 0
        nsamples = node.nsamples

        for feature in features:
            values = x[indices, feature]

            if len(np.unique(values)) == 1:  # The feature is not informative
                continue

            threshold, quality = bestsplit(values, targets)
            cond = values <= threshold
            splits = indices[cond], indices[~cond]

            if any(len(samples) < minleaf for samples in splits):
                continue

            if quality > best_quality:
                best_feature = feature
                best_threshold = threshold
                best_quality = quality

        # If no feature can perform a split, keep the node as a leaf
        if best_feature is None:
            continue

        # Creates left and right branches and add the child nodes to the nodelist
        left_child, right_child = node.split(best_feature, best_threshold)
        cond = x[indices, best_feature] <= best_threshold
        nodelist.append((left_child, indices[cond]))
        nodelist.append((right_child, indices[~cond]))

    return root


def tree_pred(x: np.ndarray, tr: 'Node') -> np.ndarray:
    """
    Predicts class values from the attribute values based on a decision tree,
    assuming the attribute values are numerical and the target values are binary (0 or 1)

    @param x: 2-dimensional array containing the attribute values
    @param tr: decision tree
    @return: vector with tree predictions
    """

    def row_pred(row):
        node = tr
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    return np.apply_along_axis(row_pred, axis=1, arr=x)


def tree_grow_b(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int):
    """
    Creates multiple decision trees,
    assuming the attribute values are numerical and the target values are binary (0 or 1)

    @param x: 2-dimensional array containing the attribute values
    @param y: vector of class labels
    @param nmin: min number of indices to create a split
    @param minleaf: min number of leafs to create a split
    @param nfeat: number of features for random forest
    @param m: number of trees to grow
    @return: root nodes of the decision trees
    """
    trees: list[Node] = []
    n_samples = x.shape[0]

    indices = []
    while len(indices) != m:
        # Generating all the random indices at once is faster with numpy for bagging
        indices = np.random.choice(n_samples, (m, n_samples), replace=True)
        indices = np.unique(indices, axis=0)  # To check if all the indices are different

    if MULTIPROCESSING:
        futures = []
        with ProcessPoolExecutor() as pool:
            for idx in indices:
                x_bootstrap, y_bootstrap = x[idx], y[idx]
                futures.append(pool.submit(tree_grow, x_bootstrap, y_bootstrap, nmin, minleaf, nfeat))
            for future in tqdm(as_completed(futures), total=m):
                trees.append(future.result())
    else:
        for idx in tqdm(indices):
            x_bootstrap, y_bootstrap = x[idx], y[idx]
            tree = tree_grow(x_bootstrap, y_bootstrap, nmin, minleaf, nfeat)
            trees.append(tree)

    return trees


def tree_pred_b(tr: list['Node'], x: np.ndarray) -> np.ndarray:
    """
    Predicts class values from the attribute values based on multiple decision trees,
    assuming the attribute values are numerical and the target values are binary (0 or 1)

    @param tr: decision trees
    @param x: 2-dimensional array containing the attribute values
    @return: vector with tree predictions
    """
    if MULTIPROCESSING:
        with ProcessPoolExecutor() as pool:
            futures = [pool.submit(tree_pred, x, tree) for tree in tr]
            predictions = [future.result() for future in futures]
    else:
        predictions = [tree_pred(x, tree) for tree in tr]
    return np.argmax(  # 2D majority class
        np.apply_along_axis(np.bincount, axis=0, arr=np.array(predictions)), axis=0
    ).astype(bool)

#Test evaluation script
from numpy import genfromtxt
import numpy as np
import pandas as pd
import time

# Basic test on credit data. Prediction should be perfect.

credit_data = genfromtxt('../credit.txt', delimiter=',', skip_header=True)
credit_x = credit_data[:,0:5]
credit_y = credit_data[:,5]
credit_tree = tree_grow(credit_x,credit_y,2,1,5)
credit_pred = tree_pred(credit_x, credit_tree)
pd.crosstab(np.array(credit_y),np.array(credit_pred))

# Single tree on pima data

pima_data = genfromtxt('../pima.txt', delimiter=',')
pima_x = pima_data[:,0:8]
pima_y = pima_data[:,8]
pima_tree = tree_grow(pima_x,pima_y,20,5,8)
pima_pred = tree_pred(pima_x,pima_tree)

# confusion matrix should be: 444,56,54,214 (50/50 leaf nodes assigned to class 0)
# or: 441,59,51,217 (50/50 leaf nodes assigned to class 1)

pd.crosstab(np.array(pima_y),np.array(pima_pred))

# Function for testing single tree

def single_test(x,y,nmin,minleaf,nfeat,n):
  acc = np.zeros(n)
  for i in range(0, n):
    tr = tree_grow(x,y,nmin,minleaf,nfeat)
    pred = tree_pred(x,tr)
    acc[i] = sum(pred == y)/len(y)
  return [np.mean(acc), np.std(acc)]

# Function for testing bagging/random forest

def rf_test(x,y,nmin,minleaf,nfeat,m,n):
  acc = np.zeros(n)
  for i in range(0,n):
    tr_list = tree_grow_b(x,y,nmin,minleaf,nfeat,m)
    pred = tree_pred_b(tr_list,x)
    acc[i] = sum(pred == y)/len(y)
  return [np.mean(acc), np.std(acc)]

# Compute average and standard deviation of accuracy for single tree

single_test(pima_x,pima_y,20,5,2,25)
single_test(pima_x,pima_y,20,5,8,25)

# Compute average and standard deviation of accuracy for bagging/random forest

rf_test(pima_x,pima_y,20,5,2,25,25)
rf_test(pima_x,pima_y,20,5,8,25,25)

# Measure time for training and prediction with random forest

start = time.time()
rf_test(pima_x,pima_y,20,5,8,25,25)
end = time.time()
print("The execution time is :", (end-start), "seconds")
