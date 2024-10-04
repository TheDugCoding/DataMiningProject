import numpy as np
import pandas as pd
import statistics
from joblib import Parallel, delayed
from multiprocessing import Pool
import multiprocessing

from statsmodels.graphics.tukeyplot import results
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

credit_data_with_headers = pd.read_csv('data/credit.txt', delimiter=',')
indians = pd.read_csv('data/indians.txt', delimiter=',', names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
eclipse_2 = pd.read_csv('data/eclipse-metrics-packages-2.0.csv', delimiter=';')
eclipse_3 = pd.read_csv('data/eclipse-metrics-packages-3.0.csv', delimiter=';')
diabetes = pd.read_csv('data/diabetes.csv', delimiter=',')

# clean data for part 2
features_data = ['pre', 'post', 'FOUT', 'MLOC', 'NBD', 'PAR', 'VG', 'NOF', 'NOM', 'NSF', 'NSM', 'ACD', 'NOI', 'NOT', 'TLOC', 'NOCU']
keep_col_list = []
for feature in features_data:
    for col in eclipse_2.columns: 
        if col.startswith(feature):
            keep_col_list.append(col)
training_data = eclipse_2[keep_col_list]
test_data = eclipse_3[keep_col_list]
training_data.loc[training_data['post'] > 0, 'post'] = 1
test_data.loc[test_data['post'] > 0, 'post'] = 1
training_features = training_data.drop('post', axis=1)
test_features = test_data.drop('post', axis=1)

def impurity_reduction_calc(y, indexes_left_child, indexes_right_child):
    return impurity(y) - (
        ((len(y[indexes_left_child]) / len(y)) * impurity(y[indexes_left_child])) +
        ((len(y[indexes_right_child]) / len(y)) * impurity(y[indexes_right_child])))

def best_split(x, y, minleaf):
    best_impurity_reduction_overall = float('-inf')
    best_value_overall = 0
    best_split_overall = ''
    best_left_child_indexes_overall = []
    best_right_child_indexes_overall = []

    if len(x) == len(y):
        for split in x.columns:
            best_impurity_reduction = float('-inf')
            best_value = 0
            best_left_child_indexes = []
            best_right_child_indexes = []
            sorted_values = np.sort(np.unique(x[split]))
            impurity_father = impurity(y)
            #check that we have enough different values for a split
            if len(sorted_values) > 1:
                # check if there are only 2 values, do the split by selecting one of the two values
                for value_index in range(len(sorted_values) -1):
                    # follows the x < c instructions, the variable avg is the average of two consecutive numbers
                    avg = sum(sorted_values[value_index:value_index + 2]) / len(
                        sorted_values[value_index:value_index + 2])
                    # select all the indexes where x < c (left child), then select indexes for the right child
                    indexes_left_child = x[split][x[split] <= avg].index.tolist()
                    indexes_right_child = list(set(x[split].index)- set(indexes_left_child))
                    # calculate impurity reduction
                    impurity_reduction = impurity_father - (
                        ((len(y[indexes_left_child]) / len(y)) * impurity(y[indexes_left_child])) +
                        ((len(y[indexes_right_child]) / len(y)) * impurity(y[indexes_right_child])))
                    if impurity_reduction > best_impurity_reduction and len(indexes_left_child) > minleaf and len(
                            indexes_right_child) > minleaf:
                        best_impurity_reduction = impurity_reduction
                        best_value = avg
                        best_left_child_indexes = indexes_left_child
                        best_right_child_indexes = indexes_right_child
            if best_impurity_reduction > best_impurity_reduction_overall:
                best_impurity_reduction_overall = best_impurity_reduction
                best_value_overall = best_value
                best_split_overall = split
                best_left_child_indexes_overall = best_left_child_indexes
                best_right_child_indexes_overall = best_right_child_indexes
        return best_left_child_indexes_overall, best_right_child_indexes_overall, best_split_overall, best_value_overall
    else:
        raise ValueError("Arrays must have the same size")

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
    def __init__(self, instances, feature=None, threshold=None, left=[], right=[], father=None, predicted_class=None):
        self.instances = instances
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.father = father
        self.predicted_class = predicted_class

class Tree:
    def __init__(self, root = None, leaves = None):
        self.root = root
        self.leaves = leaves


def tree_grow(x, y, nmin, minleaf, nfeat):

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

                # random sample nfeat number of columns (should we create the condition for the random forest?)
                candidate_features = np.random.choice(x.columns, size=nfeat, replace=False)

                # calculate best split and impurity reduction to get child nodes
                left, right, feature, threshold = best_split(x.loc[current_node_instances, candidate_features], labels, minleaf)

                # store current node info
                if feature:
                    current_node.left = Node(left, father=current_node)
                    current_node.right = Node(right, father=current_node)
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
    # assignment states trees must be in list
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

    for i in range(m):
        #trees.append(tree_grow(x.loc[random_indexes_with_replacement[i], x.columns != target_feature].reset_index(drop=True), x.loc[random_indexes_with_replacement[i], target_feature].reset_index(drop=True), nmin, minleaf, nfeat))
        result = pool.apply_async(tree_grow, args=(x.loc[random_indexes_with_replacement[i], x.columns != target_feature].reset_index(drop=True), x.loc[random_indexes_with_replacement[i], target_feature].reset_index(drop=True), nmin, minleaf, nfeat), callback=collect_result)
        results.append(result)
    pool.close()
    pool.join()

    return trees

def tree_pred(x, tr):
    predicted_labels = []
    for index, row in x.iterrows():
        current_node = tr.root
        # leaf node doesn't contain a feature
        while current_node.feature:
            if row[current_node.feature] < current_node.threshold:
                current_node = current_node.left
                #print('left')
            else:
                current_node = current_node.right
                #print('right')
        predicted_labels.append(current_node.predicted_class)

    return predicted_labels

def tree_pred_b(x, tr):
    majority_votes = {}
    predicted_labels = []
    for tree in tqdm(tr, desc="Processing Trees", unit="tree"):
        predicted_labels.append(tree_pred(x, tree))

    # Loop over the list of predicted labels (one list for each tree)
    for tree_predictions in tqdm(predicted_labels, desc="Processing Predictions", unit="set"):
        # Loop over the individual predictions in a tree
        for i in range(len(tree_predictions)):
            if i not in majority_votes:
                majority_votes[i] = 0
            # Add 1 for '1', subtract 1 for '0'
            majority_votes[i] += 1 if tree_predictions[i] == 1 else -1

        # Return 1 if the sum is positive (more 1s), else 0 (more 0s or only 0)
    return {index: 1 if vote > 0 else 0 for index, vote in majority_votes.items()}

def print_tree(node, level=0, side="root"):
    """ Recursively print the structure of the decision tree. """
    if node is None:
        print("The tree is empty.")
        return

    indent = "   " * level  # Indentation for visual representation

    # Check if node is a leaf
    if node != []:
        if node.left is None and node.right is None:
            # Leaf node: print predicted class and number of instances
            print(f"{indent}- {side} [Leaf] Predicted class: {node.predicted_class}, Instances: {len(node.instances)}")
        else:
            # Internal node: print splitting feature and threshold
            print(f"{indent}- {side} [Node] Feature: {node.feature}, Threshold: {node.threshold}, Instances: {len(node.instances)}")

            # Recursively print the left and right subtrees
            if node.left is not None:
                print_tree(node.left, level + 1, "left")
            else:
                print(f"{indent}   - left [Empty]")  # Show if the left child is missing

            if node.right is not None:
                print_tree(node.right, level + 1, "right")
            else:
                print(f"{indent}   - right [Empty]")  # Show if the right child is missing

def predict_parallel_processing(x, tree):
    print(x)
    return Parallel(n_jobs=-1)(delayed(tree_pred)(x, tr) for tr in tree)

if __name__ == '__main__':
    #print(best_split(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data_with_headers['class'], 2))

    single_tree = tree_grow(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data_with_headers['class'], 2, 2, 5)
    # print(single_tree)

    ensamble_tree = tree_grow_b(credit_data_with_headers, 'class', 2, 2, 5, 10)
    # print(ensamble_tree)

    #test prediction
    print('\n\n--prediction single tree')
    print(tree_pred(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], single_tree))

    #test prediction_b
    print('\n\n--prediction all trees')
    predictions = tree_pred_b(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'].iloc[-2:], ensamble_tree)
    # print(predictions)

    # test indians confusion matrix
    indians_tree = tree_grow(indians.drop('i', axis=1), indians['i'], 20, 5, 8)
    indians_pred = tree_pred(indians.drop('i', axis=1), indians_tree)
    pred_true = {'00': 0, '10': 0, '01': 0, '11': 0}
    for i in range(len(indians_pred)):
        # check whether class of original dataset is equal to predicted class
        if indians['i'][i] == indians_pred[i]:
            if indians_pred[i] == 1:
                pred_true['11'] += 1
            else:
                pred_true['00'] += 1
        else:
            if indians_pred[i] == 1:
                pred_true['10'] += 1
            else:
                pred_true['01'] += 1
    print(pred_true)

    # training - single tree
    print('\n\n--prediction single tree dataset')
    train_tree = tree_grow(training_features, training_data['post'], 15, 5, 41)
    test_tree = tree_pred(test_features, train_tree)
    confusion_matrix = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
    for i in range(len(test_tree)):
        # check whether pred (tree) and true data are equal
        if test_tree[i] == 0:
            if test_data['post'][i] == 0:
                confusion_matrix['TN'] += 1
            if test_data['post'][i] > 0:
                confusion_matrix['FN'] += 1
        if test_tree[i] > 0:
            if test_data['post'][i] > 0:
                confusion_matrix['TP'] += 1
            if test_data['post'][i] == 0:
                confusion_matrix['FP'] += 1

    accuracy = (confusion_matrix['TN'] + confusion_matrix['TP']) / len(test_tree)
    precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
    recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
    print('single tree', accuracy, precision, recall)
    print(confusion_matrix)

    # training - bagging
    print('\n\n--prediction bagging dataset')
    train_bagging = tree_grow_b(training_data, 'post', 15, 5, 41, 100)
    test_bagging = tree_pred_b(test_data, train_bagging)
    confusion_matrix = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
    for i in range(len(test_bagging)):
        # check whether pred (tree) and true data are equal
        if test_bagging[i] == 0:
            if test_data['post'][i] == 0:
                confusion_matrix['TN'] += 1
            if test_data['post'][i] > 0:
                confusion_matrix['FN'] += 1
        if test_bagging[i] > 0:
            if test_data['post'][i] > 0:
                confusion_matrix['TP'] += 1
            if test_data['post'][i] == 0:
                confusion_matrix['FP'] += 1

    accuracy = (confusion_matrix['TN'] + confusion_matrix['TP']) / len(test_tree)
    precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
    recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
    print('bagging', accuracy, precision, recall)
    print(confusion_matrix)

    # training - random forest
    print('\n\n--prediction random forest dataset')
    train_random = tree_grow_b(training_data, 'post', 15, 5, 6, 100)
    test_random = tree_pred_b(test_data, train_random)
    confusion_matrix = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
    for i in range(len(test_random)):
        # check whether pred (tree) and true data are equal
        if test_random[i] == 0:
            if test_data['post'][i] == 0:
                confusion_matrix['TN'] += 1
            if test_data['post'][i] > 0:
                confusion_matrix['FN'] += 1
        if test_random[i] > 0:
            if test_data['post'][i] > 0:
                confusion_matrix['TP'] += 1
            if test_data['post'][i] == 0:
                confusion_matrix['FP'] += 1

    accuracy = (confusion_matrix['TN'] + confusion_matrix['TP']) / len(test_tree)
    precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
    recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
    print('random forest', accuracy, precision, recall)
    print(confusion_matrix)
