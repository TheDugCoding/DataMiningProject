from assignment1 import *
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

def print_tree_recursive(node, level=0, side="root", split_level=3):
    """ Recursively print the structure of the decision tree. """
    if node is None:
        print("The tree is empty.")
        return

    indent = "   " * level  # Indentation for visual representation

    # Check if node is a leaf
    if node != [] and split_level > 0:
        if node.left is None and node.right is None:
            # Leaf node: print predicted class and number of instances
            print(f"{indent}- {side} [Leaf] Predicted class: {node.predicted_class}, Instances: {len(node.instances)}, Class distribution={}")
        else:
            # Internal node: print splitting feature and threshold
            print(f"{indent}- {side} [Node] Feature: {node.feature}, Threshold: {node.threshold}, Instances: {len(node.instances)}, Class distribution={}, left")

            # Recursively print the left and right subtrees
            print_tree_recursive(node.left, level + 1, "left", split_level - 1)
            print_tree_recursive(node.right, level + 1, "right", split_level - 1)


def print_tree(single_credit=False, ensamble_credit=False, single_indians=False, single_eclipse=False, bagging=False, random_forest=False):
    trees_to_process = [
        (single_credit, "Single tree - Credit data"),
        (single_indians, "Single tree - Indians data"),
        (single_eclipse, "Single tree - Eclipse data"),
        (ensamble_credit, "Ensamble credit"),
        (bagging, "Bagging:"),
        (random_forest, "Random forest")
    ]

    for tree, message in trees_to_process:
        if tree:
            if isinstance(tree, list):
                print(message)
                for t in tree:
                    print_tree_recursive(t)
            else:
                print(message)
                print_tree_recursive(tree)

def mcnemar_test(y_true, y_pred1, y_pred2):
    # We need to compare the predictions to build the contingency table
    #correct_bagging = (test_bagging == test_data['post'])
    correct_rf = (test_random == test_data['post'])

    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    # Contigency table
    table = np.zeros((2, 2))
    table[0, 0] = np.sum(correct1 & correct2)   # Both correct
    table[0, 1] = np.sum(~correct1 & correct2)  # Model 2 correct, Model 1 wrong
    table[1, 0] = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    table[1, 1] = np.sum(~correct1 & ~correct2) # Both wrong
    
    result = mcnemar(table, exact=True)
    return result.pvalue



if __name__ == '__main__':

    # Basic test on credit data. Prediction should be perfect.

    credit_data = genfromtxt('data/credit.txt', delimiter=',', skip_header=True)
    credit_x = credit_data[:, 0:5]
    credit_y = credit_data[:, 5]
    credit_tree = tree_grow(credit_x, credit_y, 2, 1, 5)
    credit_pred = tree_pred(credit_x, credit_tree)
    print(pd.crosstab(np.array(credit_y), np.array(credit_pred)))
    print_tree(single_credit=credit_tree)
    

    # Trainign to perfrom the Statistical tests 

    # Read data
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


    # Statistical Tests (McNemar) for the difference in Accuracy 

	# Pairwise comparisons (without correction) 
    p_value_single_tree_bagging = mcnemar_test(test_data['post'], test_tree, test_bagging)
    p_value_bagging_rf = mcnemar_test(test_data['post'], test_bagging, test_random)
    p_value_rf_single_tree = mcnemar_test(test_data['post'], test_random, test_tree)

    # Bonferroni correction: 3 comparisons, so we divide the alpha level by 3
    alpha = 0.05
    bonferroni_alpha = 0.05 / 3
    significance_levels = [alpha, bonferroni_alpha]

    for alpha in significance_levels:
        print(f"Significance level:  alpha = {alpha:.4f}\n")

        print(f"Single Tree vs Bagging p-value: {p_value_single_tree_bagging:.4f}")
        if p_value_single_tree_bagging < alpha:
            print("Significant difference after Bonferroni correction.")
        else:
            print("No significant difference after Bonferroni correction.")

        print(f"Bagging vs Random Forest p-value: {p_value_bagging_rf:.4f}")
        if p_value_bagging_rf < alpha:
            print("Significant difference after Bonferroni correction.")
        else:
            print("No significant difference after Bonferroni correction.")

        print(f"Random Forest vs Single Tree p-value: {p_value_rf_single_tree:.4f}")
        if p_value_rf_single_tree < alpha:
            print("Significant difference after Bonferroni correction.")
        else:
            print("No significant difference after Bonferroni correction.")