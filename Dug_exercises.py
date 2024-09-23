from collections import Counter
import numpy as np
from itertools import combinations
import pandas as pd

credit_data = np.genfromtxt('data/credit.txt', delimiter=',', skip_header=True)
credit_data_with_headers = pd.read_csv('data/credit.txt', delimiter=',')
#assignment 1

"""gini index for two class

i(t) = p(0|t) * p(1|t) = p(0|t) * 1 - p(0|t)

impurity reduction = ∆i(s,t) = i(t) − {π(l)i(l) + π(r)i(r)}
"""

#this should work even with multiple classes (not true the formula for the multiclass Gini Index is deifferent)
def gini_index_calc(x):
    gini_index = 1
    for class_name, value in Counter(x).items():
        gini_index *= (value/len(x))
    #print(gini_index)
    return gini_index

def best_split(x,y):
    best_impurity_reduction = 1.1
    best_value = 0
    if len(x) == len(y):
        print(x, y)
        sorted_values = np.sort(np.unique(credit_data[:, 3]))
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
        return best_value

    else:
        raise ValueError("Arrays must have the same size")

def rSubset(arr, r):

    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))


labels = [1,0,1,1,1,0,0,1,1,0,1]


#test Gini index
gini_index_calc(labels)
print(best_split(credit_data[:,3],credit_data[:,5]))

#homework part 1
x = [2,2,3,4,4,5,6,7,8,9]
y = [0,0,0,1,1,1,0,2,2,2]

#test for the subset, remember to remove the duplicate from the feature list and select only the unique values
arr = [1, 2, 3, 4]
r = 2
for index in range(1, len(arr)+1):
    print (rSubset(arr, index))

def best_split_v2(x,y):
    best_impurity_reduction = float('inf')
    best_value = 0
    combinations_of_features = []
    # Initialize an empty list to store the separated column names
    split_columns = []
    # Loop through each element in the Index and split by commas
    for col in x.columns:
        split_columns.extend(col.split(','))
    if len(x) == len(y):
        #print(x, y)
        #sorted_values = np.sort(np.unique(list(x.columns.split(','))))

        #print(sorted_values)
        #calculating all the possible feature combinations fron nfeats
        for index in range(1, len(split_columns) + 1):
            combinations_of_features.append(rSubset(split_columns, index))
        for combination in combinations_of_features:
            for tuple in combination:
                print(tuple)
                #select all the indexes that belogs to one of the features belonging to one of the two nodes
                indexes_left_child = [i for i, value in enumerate(x) if value in tuple]
                indexes_right_child = list(set(range(len(x))) - set(indexes_left_child))
                #calculate gini index for the current split, for both children
                gini_index_left_child = gini_index_calc(y[indexes_left_child])
                gini_index_right_child = gini_index_calc(y[indexes_right_child])
                #calculate impurity reduction, lecture 2 slide 12
                impurity_reduction = gini_index_calc(y) - (len(y[indexes_left_child])/len(y) * gini_index_left_child + len(y[indexes_right_child])/len(y) * gini_index_right_child)
                if impurity_reduction < best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_value = tuple
        return best_value

    else:
        raise ValueError("Arrays must have the same size")

def best_split_v3(x,y):
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
            print(sorted_values)
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
                if impurity_reduction < best_impurity_reduction:
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

#best_split_v2(credit_data[:,3],credit_data[:,5])
#best_split_v2(credit_data_with_headers, credit_data[:,5])
print(best_split_v3(credit_data_with_headers.loc[:, credit_data_with_headers.columns != 'class'], credit_data[:,5]))

data_matrix = [[1,0,1,1],[1,0,0,1],[0,1,0,1]]
labels = [[0,1],[1,1]]