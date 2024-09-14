from collections import Counter
import numpy as np

credit_data = np.genfromtxt('data/credit.txt', delimiter=',', skip_header=True)

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
    print(gini_index)
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




labels = [1,0,1,1,1,0,0,1,1,0,1]


#test Gini index
gini_index_calc(labels)
print(best_split(credit_data[:,3],credit_data[:,5]))