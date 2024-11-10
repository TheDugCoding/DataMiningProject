from assignment1 import *
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import mean_squared_error
import seaborn as sns

def print_tree_recursive(header, node, level=0, side="root", split_level=3):
    """ Recursively print the structure of the decision tree. """
    if node is None:
        print("The tree is empty.")
        return
    if isinstance(node, Tree):
       node = node.root
                
    indent = "   " * level  # Indentation for visual representation
    # Check if node is a leaf
    if node != [] and split_level > 0:
        if node.left is None and node.right is None:
            # Leaf node: print predicted class and number of instances
            print(f"{indent}- {side} [Leaf] Predicted class: {node.predicted_class}, Instances: {len(node.instances)}, Class distribution={node.class_distribution}")
        else:
            # Internal node: print splitting feature and threshold 
            print(f"{indent}- {side} [Node] Feature: {header[node.feature]}, Threshold: {node.threshold}, Instances: {len(node.instances)}, Class distribution={node.class_distribution}")

            # Recursively print the left and right subtrees
            print_tree_recursive(header, node.left, level + 1, "left", split_level - 1)
            print_tree_recursive(header, node.right, level + 1, "right", split_level - 1)


def print_tree(header, single_credit=False, ensamble_credit=False, single_indians=False, single_eclipse=False, bagging=False, random_forest=False):
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
                    print_tree_recursive(header=header, node=t)
            else:
                print(message)
                print_tree_recursive(header=header, node=tree)

def mcnemar_test(y_true, y_pred1, y_pred2):
    # We need to compare the predictions to build the contingency table
    #correct_bagging = (test_bagging == test_data['post'])
    #correct_rf = (test_random == y_true)

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

# Define the features we want to keep based on column names
FEATURES = ['pre', 'post', 'FOUT', 'MLOC', 'NBD', 'PAR', 'VG', 'NOF', 'NOM', 'NSF', 'NSM', 'ACD', 'NOI', 'NOT', 'TLOC', 'NOCU']

def load_and_filter_data(file_path, delimiter=';'):
    """
    Load data from a CSV file using NumPy, filter columns based on the FEATURES list,
    and return filtered data as a NumPy array.
    """
    # Load the header separately to determine column indices
    with open(file_path, 'r') as file:
        header = file.readline().strip().split(delimiter)
    
    # Find indices of columns to keep
    keep_indices = [i for i, col in enumerate(header) if any(col.startswith(f) for f in FEATURES)]
    
    # Load the entire dataset, skipping the header
    data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=1, usecols=keep_indices)
    
    # Get the filtered column names
    filtered_header = [header[i] for i in keep_indices]
    
    # Return the filtered data, index of the 'post' column in the filtered list, and the filtered header
    post_idx = filtered_header.index('post')  # Now it's the index within `filtered_header`
    return data, post_idx, filtered_header

def prepare_data(data, post_idx):
    """
    Binarize the 'post' column and separate features and labels for training/testing.
    """
    # Binarize the 'post' column: set to 1 if greater than 0, else 0
    data[:, post_idx] = np.where(data[:, post_idx] > 0, 1, 0)
    
    # Split into features (X) and labels (y)
    X = np.delete(data, post_idx, axis=1)  # All columns except 'post'
    y = data[:, post_idx].astype(int)      # The 'post' column as integer labels
    return X, y

def compute_confusion_matrix(predictions, true_values):
    """Compute the confusion matrix."""
    matrix = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
    for pred, true in zip(predictions, true_values):
        if pred == 0 and true == 0:
            matrix['TN'] += 1
        elif pred == 0 and true == 1:
            matrix['FN'] += 1
        elif pred == 1 and true == 1:
            matrix['TP'] += 1
        elif pred == 1 and true == 0:
            matrix['FP'] += 1
    return matrix

def compute_metrics(conf_matrix):
    """Calculate accuracy, precision, and recall from confusion matrix."""
    tn, fp, fn, tp = conf_matrix['TN'], conf_matrix['FP'], conf_matrix['FN'], conf_matrix['TP']
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return accuracy, precision, recall

# Function to plot model performance
def plot_model_performance(models_metrics, model_names):
    """Plot accuracy, precision, and recall for each model."""
    # Convert the list of metrics dictionaries to a NumPy array for easier plotting
    metrics_array = np.array(models_metrics)
    
    # Extract individual metrics for plotting
    accuracies = metrics_array[:, 0]
    precisions = metrics_array[:, 1]
    recalls = metrics_array[:, 2]

    # Plotting accuracy, precision, and recall for each model
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].bar(model_names, accuracies, color='skyblue')
    ax[0].set_title("Model Accuracy")
    ax[0].set_ylim([0, 1])
    ax[0].set_xlabel("Models")
    ax[0].set_ylabel("Accuracy")
    
    ax[1].bar(model_names, precisions, color='salmon')
    ax[1].set_title("Model Precision")
    ax[1].set_ylim([0, 1])
    ax[1].set_xlabel("Models")
    ax[1].set_ylabel("Precision")
    
    ax[2].bar(model_names, recalls, color='lightgreen')
    ax[2].set_title("Model Recall")
    ax[2].set_ylim([0, 1])
    ax[2].set_xlabel("Models")
    ax[2].set_ylabel("Recall")
    
    plt.tight_layout()
    plt.savefig("all_model_performance.png", format="png")
    plt.show()

def calculate_mse(model_type, X_train, y_train, X_test, y_test, n_trials=100, n_trees=100):
    """Run multiple trials and compute mean squared error (MSE) for a specified model type."""
    mse_values = []

    for _ in tqdm(range(n_trials), desc=f"Running {model_type} trials", unit="trial"):
        if model_type == "single_tree":
            model = tree_grow(X_train, y_train, 15, 5, 41)
            predictions = tree_pred(X_test, model)
        
        elif model_type == "bagging":
            model = tree_grow_b(X_train, y_train, 15, 5, 41, n_trees)
            predictions = tree_pred_b(X_test, model)
        
        elif model_type == "random_forest":
            model = tree_grow_b(X_train, y_train, 15, 5, 6, n_trees)
            predictions = tree_pred_b(X_test, model)
        
        # Calculate MSE for this trial and store it
        mse = mean_squared_error(y_test, predictions)
        mse_values.append(mse)

    return mse_values

def plot_mse_variances(X_train, y_train, X_test, y_test, n_trials=100, n_trees=100):
    """Plot mean squared error (MSE) and variances for single tree, bagging, and random forest models."""
    # Calculate MSE values across multiple trials for each model type
    single_tree_mse = calculate_mse("single_tree", X_train, y_train, X_test, y_test, n_trials)
    bagging_mse = calculate_mse("bagging", X_train, y_train, X_test, y_test, n_trials, n_trees)
    random_forest_mse = calculate_mse("random_forest", X_train, y_train, X_test, y_test, n_trials, n_trees)

    # Calculate mean and variance for each model's MSE
    mean_mse = {
        "Single Tree": np.mean(single_tree_mse),
        "Bagging": np.mean(bagging_mse),
        "Random Forest": np.mean(random_forest_mse)
    }
    mse_variances = {
        "Single Tree": np.var(single_tree_mse),
        "Bagging": np.var(bagging_mse),
        "Random Forest": np.var(random_forest_mse)
    }

    # Plotting mean MSE
    plt.figure(figsize=(12, 6))
    plt.bar(mean_mse.keys(), mean_mse.values(), color=['blue', 'green', 'orange'])
    plt.title("Mean MSE Comparison")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.savefig("mean_mse_comparison.png", format="png")
    plt.show()

    # Plotting MSE variances
    plt.figure(figsize=(12, 6))
    plt.bar(mse_variances.keys(), mse_variances.values(), color=['blue', 'green', 'orange'])
    plt.title("MSE Variance Comparison")
    plt.ylabel("MSE Variance")
    plt.savefig("mse_variance_comparison.png", format="png")
    plt.show()

    return mean_mse, mse_variances

def calculate_tree_predictions_rf(X_train, y_train, X_test, n_trees=100):
    """Get predictions from individual trees in a random forest model."""
    predictions = []
    for _ in range(n_trees):
        rf_model = tree_grow_b(X_train, y_train, 15, 5, 6, 100)  
        single_tree_pred = tree_pred(X_test, rf_model[0])
        predictions.append(single_tree_pred)
    return np.array(predictions)

def plot_rf_correlation_matrix(predictions_rf):
    """Plot the correlation matrix for predictions of individual trees in a random forest."""
    n_trees = predictions_rf.shape[0]
    correlation_matrix = np.zeros((n_trees, n_trees))
    
    # Calculate pairwise correlations between tree predictions
    for i in range(n_trees):
        for j in range(n_trees):
            correlation_matrix[i, j] = np.corrcoef(predictions_rf[i], predictions_rf[j])[0, 1]

    # Plotting the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Random Forest Tree Prediction Correlation Matrix")
    plt.xlabel("Tree Index")
    plt.ylabel("Tree Index")
    plt.savefig("random_forest_tree_correlation_matrix.png", format="png")
    plt.show()

if __name__ == '__main__':
    """ Uncomment to test data structures has been handled consistently""" 
    # Load and prepare Eclipse 2.0 and Eclipse 3.0 datasets
    eclipse_2_data, post_index_2, header_train = load_and_filter_data('data/eclipse-metrics-packages-2.0.csv')
    eclipse_3_data, post_index_3, header_test = load_and_filter_data('data/eclipse-metrics-packages-3.0.csv')

    # Prepare training and test datasets
    X_train, y_train = prepare_data(eclipse_2_data, post_index_2)
    X_test, y_test = prepare_data(eclipse_3_data, post_index_3)

    # Output the results to verify
    print(f"Eclipse 2 Data set size: {X_train.shape}")
    print(f"Eclipse 3 Data set size: {X_test.shape}")
    print(f"Class distribution for Eclipse 2: {np.bincount(y_train)}")
    print(f"Class distribution for Eclipse 3: {np.bincount(y_test)}")
    # After train-test split 
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels distribution: {np.bincount(y_train)}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Test labels distribution: {np.bincount(y_test)}")
        
    # Model training and evaluation metrics collection
    model_metrics = []
    model_names = ["Single Tree", "Bagging", "Random Forest"]

    # Training - single tree
    train_tree = tree_grow(X_train, y_train, 15, 5, 41)
    test_tree = tree_pred(X_test, train_tree)
    print_tree(header_train, single_eclipse=train_tree)
    single_tree_matrix = compute_confusion_matrix(test_tree, y_test)
    single_tree_metrics = compute_metrics(single_tree_matrix)
    print('Single Tree', single_tree_metrics)
    print(single_tree_matrix)
    model_metrics.append(single_tree_metrics)

    # Training - bagging
    train_bagging = tree_grow_b(X_train, y_train, 15, 5, 41, 100)
    test_bagging = tree_pred_b(X_test, train_bagging)
    bagging_matrix = compute_confusion_matrix(test_bagging, y_test)
    bagging_metrics = compute_metrics(bagging_matrix)
    print('Bagging', bagging_metrics)
    print(bagging_matrix)
    model_metrics.append(bagging_metrics)

    # Training - random forest
    train_random = tree_grow_b(X_train, y_train, 15, 5, 6, 100)
    test_random = tree_pred_b(X_test, train_random)
    random_matrix = compute_confusion_matrix(test_random, y_test)
    random_metrics = compute_metrics(random_matrix)
    print('Random Forest', random_metrics)
    print(random_matrix)
    model_metrics.append(random_metrics)


    # Statistical Tests (McNemar) for Accuracy Differences
    comparisons = [
        ("Single Tree vs Bagging", test_tree, test_bagging),
        ("Bagging vs Random Forest", test_bagging, test_random),
        ("Random Forest vs Single Tree", test_random, test_tree)
    ]

    # Set significance levels with Bonferroni correction
    alpha, bonferroni_alpha = 0.05, 0.05 / 3
    for comparison_name, model1_preds, model2_preds in comparisons:
        p_value = mcnemar_test(y_test, model1_preds, model2_preds)
        print(f"{comparison_name} p-value: {p_value:.4f}")
        
        if p_value < alpha:
            print(f"{comparison_name}: Significant difference at alpha = {alpha:.4f}")
        else:
            print(f"{comparison_name}: No significant difference at alpha = {alpha:.4f}")
        
        if p_value < bonferroni_alpha:
            print(f"{comparison_name}: Significant difference after Bonferroni correction at alpha = {bonferroni_alpha:.4f}")
        else:
            print(f"{comparison_name}: No significant difference after Bonferroni correction at alpha = {bonferroni_alpha:.4f}")
                        
    # Plot model performance comparison
    plot_model_performance(model_metrics, model_names)
        
    # Plot MSE and variance for each model type
    mean_mse, mse_variances = plot_mse_variances(X_train, y_train, X_test, y_test)
    print("Mean MSE:", mean_mse)
    print("MSE Variances:", mse_variances)

    # Decorrelated predictions analysis with Random Forest
    rf_predictions = calculate_tree_predictions_rf(X_train, y_train, X_test, n_trees=10)
    plot_rf_correlation_matrix(rf_predictions)
