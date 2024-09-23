import numpy  as np

def impurity(x):
    """
    Computes the impurity of a vector (pf class labels) using the Gini index. 
    """
    array_len = len(x)
    if np.all(x==0) or np.all(x==1):
        return 0
    else:
        count_x = np.count_nonzero(x)
        count_y = array_len - count_x
        gini_index = (count_x / array_len) * (count_y / array_len) 
        return gini_index

def bestsplit(x, y):
    """
    Computes the best value on a numeric attribute vector x given the vector of labels y
    """
    best_split = 0
    best_reduction = 0
    sorted_x = np.sort(x)
    num_instances = len(x)
    parent_impurity = (np.count_nonzero(y == 0) / num_instances) 

    for i in range(0, num_instances, 2):
        c = int(np.mean(sorted_x[i:i+2]))

        # split instances
        pred_left = y[x <= c]
        pred_right = y[x > c]       

        # compute impurity after the split
        impurity_left = impurity(pred_left)
        impurity_right = impurity(pred_right)

        # compute impurity reduction
        impurity_reduction = parent_impurity - ((len(pred_left)/num_instances) * impurity_left + (len(pred_right)/num_instances) * impurity_right)

        # update best split
        if impurity_reduction > 0 and impurity_reduction > best_reduction: 
            best_split = c
            best_reduction = impurity_reduction
    
    return best_split 

class Node:
    def __init__(self, feature, c, prediction, left, right):
        self.feature = feature
        self.c = c
        self.prediction = prediction
        self.left = left
        self.right = right 


def tree_grow(x, y, nmin, minleaf, nfeat):
    nodelist = x
    while len(nodelist) > 0:
        # early stopping
        if x < nmin:
            pass
        for feature in x:
            # looking for the best value to split a given attribute
            c = best_split(x[feature], y)

            # split instances
            pred_left = y[x <= c]
            pred_right = y[x > c]   

            #if impurity(i) > 0:
            


if __name__ == "__main__":
    # Read the dataset in the txt file into a 2d-matrix 
    credit_data = np.genfromtxt('dm/credit.txt', delimiter=',', skip_header=True)

    # Compute the gini index 
    array = np.array([1,0,1,1,1,0,0,1,1,0,1])
    gini_index = impurity(array)

    # Compute the best split 
    best_split = bestsplit(credit_data[:,3], credit_data[:,5])

    #print(credit_data)
    print(gini_index)
    print(best_split)



