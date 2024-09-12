import numpy as np

def best_split(x, y):
    sorted = np.sort(np.unique(x[:3]))
    print((sorted[0:7]+sorted[1:8])/2)

def impurity(x):
    sum = 0
    for i in x:
        sum += i
    prob_0 = sum/len(x)
    prob_1 = 1-prob_0
    return prob_0 * prob_1

def tree_grow(x, y, nmin, minleaf, nfeat):
    tree = 1
    nodelist = x
    # nfeat: number of columns considered per split

    split = best_split(x, y)

    for i in nodelist:
        len(nfeat) - 1 
        node = i
    return tree

def tree_pred():
    print('tree')

data_matrix = [1,0,1,1,1,0,0,1,1,0,1]
labels = [0]
# tree_grow(data_matrix, labels, 2, 2, 2)
best_split(data_matrix, labels)
