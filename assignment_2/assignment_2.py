import os

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


data_folder = os.path.join(os.getcwd(), 'data\\')
training_folders = ['fold1', 'fold2', 'fold3', 'fold4']
testing_folders = ['fold5']
removing_words = [8, 23]

# Function to load reviews and labels from a folder
def load_data_from_folder(main_folder,  folders, label):
    reviews = []
    labels = []

    for fold in folders:
        fold_path = os.path.join(main_folder, fold)
        for file_name in os.listdir(fold_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(fold_path, file_name)

                # Read the content of the txt file
                with open(file_path, 'r', encoding='utf-8') as file:
                    review = file.read().strip()
                    reviews.append(review)
                    labels.append(label)

    return pd.DataFrame({'review': reviews, 'label': labels})


# load the data for negative review and put a 0
deceptive_reviews_training = load_data_from_folder(f'{data_folder}deceptive_from_MTurk', training_folders, label=0)
deceptive_reviews_testing = load_data_from_folder(f'{data_folder}deceptive_from_MTurk', testing_folders, label=0)

#  load the data for negative review and put a 1
truthful_reviews_training = load_data_from_folder(f'{data_folder}truthful_from_Web', training_folders, label=1)
truthful_reviews_testing =  load_data_from_folder(f'{data_folder}truthful_from_Web', testing_folders, label=1)

# Combine the two datasets into one DataFrame
combined_reviews_training = pd.concat([deceptive_reviews_training, truthful_reviews_training], ignore_index=True)
combined_reviews_testing = pd.concat([deceptive_reviews_testing, truthful_reviews_testing], ignore_index=True)

# Initialize lists to store metrics
metrics_tree = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
metrics_rf = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for words_to_remove in removing_words:
    print(f" words removed: {words_to_remove}")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=words_to_remove, stop_words='english')
    X_train = vectorizer.fit_transform(combined_reviews_training['review'])
    X_test = vectorizer.transform(combined_reviews_testing['review'])

    y_train = combined_reviews_training['label']
    y_test = combined_reviews_testing['label']

    # Step 2: Train a classification tree using cross-validation for hyperparameter tuning
    # We'll tune the 'ccp_alpha' parameter (cost-complexity pruning)

    # Define the decision tree model
    decision_tree = DecisionTreeClassifier(random_state=42)

    # Hyperparameter tuning using GridSearchCV for finding the best 'ccp_alpha', it also automatically apply cross validation
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'ccp_alpha': [0.0, 0.01, 0.02, 0.03],  # Cost-complexity pruning parameter
        'max_depth': [None, 10, 20, 30],  # Add max depth to control the depth of the tree

    }  # Example values for alpha
    grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best model after hyperparameter tuning
    best_tree = grid_search.best_estimator_

    # Step 3: Evaluate the model on the test set
    y_pred = best_tree.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(conf_matrix)

    # Step 4: Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append metrics for Decision Tree
    metrics_tree['accuracy'].append(accuracy)
    metrics_tree['precision'].append(precision)
    metrics_tree['recall'].append(recall)
    metrics_tree['f1'].append(f1)

    # Print out the results
    # Print the selected hyperparameters of the best tree
    print("Best Decision Tree Model:", best_tree)
    print(f"Criterion: {best_tree.criterion}")
    print(f"Min Samples Split: {best_tree.min_samples_split}")
    print(f"CCP Alpha: {best_tree.ccp_alpha}")
    print(f"Min Samples Leaf: {best_tree.min_samples_leaf}")
    print(f"Max Depth: {best_tree.max_depth}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # Output Decision Tree metrics
    print("Best Decision Tree Model:", best_tree)
    """
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Negative Class): {precision[0]:.4f}")  # Accessing negative class precision
    print(f"Recall (Negative Class): {recall[0]:.4f}")  # Accessing negative class recall
    print(f"F1 Score (Negative Class): {f1[0]:.4f}")  # Accessing negative class F1 score
    print(f"Precision (Positive Class): {precision[1]:.4f}")  # Accessing positive class precision
    print(f"Recall (Positive Class): {recall[1]:.4f}")  # Accessing positive class recall
    print(f"F1 Score (Positive Class): {f1[1]:.4f}")
    """

    # Get feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Get the feature importances from the decision tree
    importances_tree = best_tree.feature_importances_

    # Sort the features by importance
    indices_tree = np.argsort(importances_tree)[::-1]

    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances_tree})
    feature_importance_pos = feature_importance.sort_values('Importance', ascending=False)[:5]
    feature_importance_neg = feature_importance.sort_values('Importance', ascending=True)[:5]

    # Print the top 5 most important features
    print("Top 5 important features for Decision Tree:")
    print('real', feature_importance_pos['Feature'].values,'fake', feature_importance_neg['Feature'].values)



    #random forest



    random_forest = RandomForestClassifier(random_state=42, oob_score=True)

    # Define the hyperparameter grid
    param_grid_rf = {
        'n_estimators': [50, 100, 200, 300],  # Number of trees
        'max_features': ['sqrt', 'log2', None],  # Number of features to consider for splits
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    }

    # GridSearchCV to find the best parameters
    grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    # Best model after hyperparameter tuning
    best_rf = grid_search_rf.best_estimator_

    # Step 3: Evaluate the Random Forest model on the test set
    y_pred_rf = best_rf.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred_rf)

    print("Confusion Matrix:")
    print(conf_matrix)

    # Get the OOB score
    oob_accuracy = best_rf.oob_score_

    # Step 4: Calculate performance metrics
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    # Append metrics for Random Forest
    metrics_rf['accuracy'].append(accuracy_rf)
    metrics_rf['precision'].append(precision_rf)
    metrics_rf['recall'].append(recall_rf)
    metrics_rf['f1'].append(f1_rf)

    # Print out the results
    # Print the selected hyperparameters of the best Random Forest model
    print("Best Random Forest Model:", best_rf)
    print(f"Number of Estimators: {best_rf.n_estimators}")
    print(f"Max Features: {best_rf.max_features}")
    print(f"Max Depth: {best_rf.max_depth}")
    print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
    print(f"Random Forest Precision: {precision_rf:.4f}")
    print(f"Random Forest Recall: {recall_rf:.4f}")
    print(f"Random Forest F1 Score: {f1_rf:.4f}")
    print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
    """
    print(f"Precision (Negative Class): {precision_rf[0]:.4f}")
    print(f"Recall (Negative Class): {recall_rf[0]:.4f}")
    print(f"F1 Score (Negative Class): {f1_rf[0]:.4f}")
    print(f"Precision (Positive Class): {precision_rf[1]:.4f}")
    print(f"Recall (Positive Class): {recall_rf[1]:.4f}")
    print(f"F1 Score (Positive Class): {f1_rf[1]:.4f}")
    """
    # Get the maximum depth of each tree in the random forest
    max_depths = [tree.tree_.max_depth for tree in best_rf.estimators_]
    # Display the maximum depth of the entire forest (deepest tree)
    print(f"Maximum depth among all trees: {max(max_depths)}")
    print(f"OOB Accuracy: {oob_accuracy:.4f}")

    # Get the feature importances from the decision tree
    importances_tree = best_rf.feature_importances_


    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances_tree})
    feature_importance_pos = feature_importance.sort_values('Importance', ascending=False)[:5]
    feature_importance_neg = feature_importance.sort_values('Importance', ascending=True)[:5]

    # Print the top 5 most important features
    print("Top 5 important features for Decision Tree:")
    print('real', feature_importance_pos['Feature'].values, 'fake', feature_importance_neg['Feature'].values)

# Plotting the metrics for Decision Tree
plt.figure(figsize=(12, 8))
plt.plot(removing_words, metrics_tree['accuracy'], marker='o', label='Accuracy')
plt.plot(removing_words, metrics_tree['precision'], marker='o', label='Precision')
plt.plot(removing_words, metrics_tree['recall'], marker='o', label='Recall')
plt.plot(removing_words, metrics_tree['f1'], marker='o', label='F1 Score')

plt.title('Decision Tree Classifier Metrics over Iterations')
plt.xlabel('Number of Words Removed')
plt.ylabel('Score')
plt.xticks(removing_words)
plt.grid()
plt.legend()
plt.show()

# Plotting the metrics for Random Forest
plt.figure(figsize=(12, 8))
plt.plot(removing_words, metrics_rf['accuracy'], marker='o', label='Accuracy')
plt.plot(removing_words, metrics_rf['precision'], marker='o', label='Precision')
plt.plot(removing_words, metrics_rf['recall'], marker='o', label='Recall')
plt.plot(removing_words, metrics_rf['f1'], marker='o', label='F1 Score')

plt.title('Random Forest Classifier Metrics over Iterations')
plt.xlabel('Number of Words Removed')
plt.ylabel('Score')
plt.xticks(removing_words)
plt.grid()
plt.legend()
plt.show()