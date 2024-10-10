This repository contains our code for the assignments for the course INFOMDM Data Mining.

-- Instructions --
Due to parallelization, to call the functions the code must be structured in the following way:

if __name__ == '__main__':
    #code

if the parallelization doesn't work, please change the code in the function tree_grow_b()

from this:

#trees.append(tree_grow(x.loc[random_indexes_with_replacement[i], x.columns != target_feature].reset_index(drop=True), x.loc[random_indexes_with_replacement[i], target_feature].reset_index(drop=True), nmin, minleaf, nfeat))
result = pool.apply_async(tree_grow, args=(x.loc[random_indexes_with_replacement[i], x.columns != target_feature].reset_index(drop=True), x.loc[random_indexes_with_replacement[i], target_feature].reset_index(drop=True), nmin, minleaf, nfeat), callback=collect_result)

to this:

trees.append(tree_grow(x.loc[random_indexes_with_replacement[i], x.columns != target_feature].reset_index(drop=True), x.loc[random_indexes_with_replacement[i], target_feature].reset_index(drop=True), nmin, minleaf, nfeat))
#result = pool.apply_async(tree_grow, args=(x.loc[random_indexes_with_replacement[i], x.columns != target_feature].reset_index(drop=True), x.loc[random_indexes_with_replacement[i], target_feature].reset_index(drop=True), nmin, minleaf, nfeat), callback=collect_result)