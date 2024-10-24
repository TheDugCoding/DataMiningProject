import os

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier


data_folder = os.path.join(os.getcwd(), 'data\\')
training_folders = ['fold1', 'fold2', 'fold3', 'fold4']
testing_folders = ['fold5']


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


vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # You can use ngrams to add bigram features
X_train = vectorizer.fit_transform(combined_reviews_training['review'])
X_test = vectorizer.transform(combined_reviews_testing['review'])

y_train = combined_reviews_training['label']
y_test = combined_reviews_testing['label']

# Step 2: Train a classification tree using cross-validation for hyperparameter tuning
# We'll tune the 'ccp_alpha' parameter (cost-complexity pruning)

# Define the decision tree model
decision_tree = DecisionTreeClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV for finding the best 'ccp_alpha', it also automatically apply cross validation
param_grid = {'ccp_alpha': [0.0, 0.001, 0.01, 0.1]}  # Example values for alpha
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_tree = grid_search.best_estimator_

# Step 3: Evaluate the model on the test set
y_pred = best_tree.predict(X_test)

# Step 4: Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print out the results
print("Best Decision Tree Model:", best_tree)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


#random forest



random_forest = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid
param_grid_rf = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider for splits
    'max_depth': [None, 10, 20, 30]  # Maximum depth of the trees
}

# GridSearchCV to find the best parameters
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_rf = grid_search_rf.best_estimator_

# Step 3: Evaluate the Random Forest model on the test set
y_pred_rf = best_rf.predict(X_test)

# Step 4: Calculate performance metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Print out the results
print("Best Random Forest Model:", best_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest Precision: {precision_rf:.4f}")
print(f"Random Forest Recall: {recall_rf:.4f}")
print(f"Random Forest F1 Score: {f1_rf:.4f}")