import numpy  as np
from collections import Counter

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
        instances_left = x <= c
        instances_right = x > c
        pred_left = y[instances_left]
        pred_right = y[instances_right]       

        # compute impurity after the split
        impurity_left = impurity(pred_left)
        impurity_right = impurity(pred_right)

        # compute impurity reduction
        impurity_reduction = parent_impurity - ((len(pred_left)/num_instances) * impurity_left + (len(pred_right)/num_instances) * impurity_right)

        # update best split
        if impurity_reduction > 0 and impurity_reduction > best_reduction: 
            best_split = c
            best_reduction = impurity_reduction
    
    return best_split, best_reduction, x[instances_left], x[instances_right]


def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    Grows a Classification Tree
    """
    # root node contains all instances
    Tree = []
    nodelist = [{'root':x}]

    while len(nodelist) > 0:
        S = set()
        current_node = nodelist[0]
        skip = False

        # retrieve node type and label count
        node_label = list(nodelist[0].keys())[0]
        label_count = Counter(y)
        X = nodelist[0][node_label]

        # compute the best split for each feature
        for feature in range(nfeat):
            if not nodelist:
                break
            
            # early stopping: pure node
            if label_count[0] < nmin or label_count[1] < nmin:
                print('nmin')
                break

            # determine the threshold
            c, reduction, instances_left, instances_right = bestsplit(X[:, feature], y)
            
            # Convert instances_left and instances_right to tuples for set storage
            instances_left_tuple = tuple(map(tuple, instances_left))
            instances_right_tuple = tuple(map(tuple, instances_right))
               
            # store split info
            S.add((c, reduction, feature, instances_left_tuple, instances_right_tuple))

            # remove node from the list
            current_node = nodelist.pop(0)

            # early stopping: min num of instances per leaf 
            if len(instances_left) < minleaf or len(instances_right) < minleaf:
                print('minleaf')
                skip = True
                break
        
        if impurity(current_node) > 0 and not skip:
            # compute best split over all other best splits
            s = max(S, key=lambda x: x[2])
            print(s)
            Tree.append(s)

            # store nodes
            nodelist.append({'left': s[3]})
            nodelist.append({'right': s[4]})

        #elif nodelist and not skip:
            #nodelist.pop(0)

    return Tree



if __name__ == "__main__":
    # Read the dataset in the txt file into a 2d-matrix 
    credit_data = np.genfromtxt('dm/credit.txt', delimiter=',', skip_header=True)
    
    # Compute the gini index 
    array = np.array([1,0,1,1,1,0,0,1,1,0,1])
    gini_index = impurity(array)

    # Compute the best split 
    X = credit_data[:,3]
    y = credit_data[:,5]
    best_split, _, _, _ = bestsplit(X, y)

    print(credit_data)
    print(gini_index)
    print(best_split)
    
    # build classification tree
    X = credit_data.astype(int)
    tree = tree_grow(X, y, 1, 1, len(X[0]))
    print(tree)



