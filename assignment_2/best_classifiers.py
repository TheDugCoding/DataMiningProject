import os
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define data loading and preprocessing functions
data_folder = os.path.join(os.getcwd(), 'data/')
training_folders = ['fold1', 'fold2', 'fold3', 'fold4']
testing_folders = ['fold5']

def load_data_from_folder(main_folder, folders, label):
    reviews = []
    labels = []
    for fold in folders:
        fold_path = os.path.join(main_folder, fold)
        for file_name in os.listdir(fold_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(fold_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    reviews.append(file.read().strip())
                    labels.append(label)
    return pd.DataFrame({'review': reviews, 'label': labels})

# Load datasets
deceptive_reviews_training = load_data_from_folder(f'{data_folder}deceptive_from_MTurk', training_folders, label=0)
deceptive_reviews_testing = load_data_from_folder(f'{data_folder}deceptive_from_MTurk', testing_folders, label=0)
truthful_reviews_training = load_data_from_folder(f'{data_folder}truthful_from_Web', training_folders, label=1)
truthful_reviews_testing = load_data_from_folder(f'{data_folder}truthful_from_Web', testing_folders, label=1)

combined_reviews_training = pd.concat([deceptive_reviews_training, truthful_reviews_training], ignore_index=True)
combined_reviews_testing = pd.concat([deceptive_reviews_testing, truthful_reviews_testing], ignore_index=True)

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = ''.join([i for i in text if not i.isdigit()])
    word_tokens = word_tokenize(text)
    return ' '.join([w for w in word_tokens if w.lower() not in stop_words])

X_train = combined_reviews_training['review'].apply(preprocess_text)
y_train = combined_reviews_training['label']
X_test = combined_reviews_testing['review'].apply(preprocess_text)
y_test = combined_reviews_testing['label']




######################### Naive Bayes #########################

# Count Vectorization
print("Performing Count Vectorization...")
count_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# Chi-square feature selection
n_features = 800  # Adjust the number of features as needed
feature_selector = SelectKBest(score_func=chi2, k=n_features)
X_train_selected = feature_selector.fit_transform(X_train_count, y_train)
X_test_selected = feature_selector.transform(X_test_count)


# Get the selected feature names
feature_names = count_vectorizer.get_feature_names_out()
selected_features_mask = feature_selector.get_support()
selected_features = feature_names[selected_features_mask]

# Train Naive Bayes with selected features
print("Training Naive Bayes model on selected features...")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_selected, y_train)

# Make predictions for both train and test sets
print("Making predictions...")
y_train_pred = nb_classifier.predict(X_train_selected)  # Predictions on training set
y_test_pred = nb_classifier.predict(X_test_selected)    # Predictions on test set

# Function to calculate and return metrics
def calculate_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1-score': report['1']['f1-score']
    }

# Calculate metrics for train and test sets
train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

# Prepare data for plotting
metrics = {
    'Train Accuracy': train_metrics['accuracy'],
    'Train Precision': train_metrics['precision'],
    'Train Recall': train_metrics['recall'],
    'Train F1 Score': train_metrics['f1-score'],
    'Test Accuracy': test_metrics['accuracy'],
    'Test Precision': test_metrics['precision'],
    'Test Recall': test_metrics['recall'],
    'Test F1 Score': test_metrics['f1-score'],
}

metric_names = list(metrics.keys())
metric_values = list(metrics.values())



######################### Logistic Regression #########################

# pre-processing
list_models = []
for n_gram in [(1,1), (1,2)]:
    for words_to_remove in [5, 6, 7, 8, 9, 10, 15, 18, 19, 20, 21, 22, 23, 24]:
        for iter in (100, 250, 500, 750, 1000, 5000):
            def transform(train, test):
                vectorizer =  TfidfVectorizer(ngram_range=n_gram, max_df=0.9, min_df=words_to_remove, stop_words='english')
                X_train = vectorizer.fit_transform(train)
                X_test = vectorizer.transform(test)
                feature_names = vectorizer.get_feature_names_out()
                return X_train, X_test, feature_names


            X_train, X_test, feature_names = transform(combined_reviews_training['review'], combined_reviews_testing['review'])
            y_train = combined_reviews_training['label']
            y_test = combined_reviews_testing['label']

            # use CV for built-in cross validation
            # Lasso: l1 penalty with liblinear solver
            regress = LogisticRegressionCV(penalty='l1', solver='liblinear', max_iter=iter)
            regress.fit(X_train, y_train)
            y_pred = regress.predict(X_test)

            coefficients = regress.coef_[0]
            feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})
            feature_importance_pos = feature_importance.sort_values('Importance', ascending=False)[:5]
            feature_importance_neg = feature_importance.sort_values('Importance', ascending=True)[:5]
            scores = classification_report(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            pre = round(precision_score(y_test, y_pred), 3)
            rec = round(recall_score(y_test, y_pred), 3)
            f1 = round(f1_score(y_test, y_pred), 3)
            c_value = regress.C_
            list_models.append([n_gram, words_to_remove, iter, c_value, acc, pre, rec, f1])
        if n_gram == (1,2) and words_to_remove in (22,23):
            print('real', feature_importance_pos['Feature'].values,'fake', feature_importance_neg['Feature'].values)
print(*list_models, sep='\n')




######################### Decision Tree #########################

for n_gram in n_grams_range:
    for words_to_remove in removing_words:
        print(f"\n\n\n---number of ngrams: {n_gram}, words removed: {words_to_remove}")
        vectorizer =  TfidfVectorizer(ngram_range=n_gram, max_df=0.9, min_df=words_to_remove, stop_words='english')
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
            'ccp_alpha': [0.0, 0.001, 0.01, 0.1],  # Cost-complexity pruning parameter
            'max_depth': [None, 10, 20, 30]  # Add max depth to control the depth of the tree
        }  # Example values for alpha
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
        print(f"max depth: {best_tree.max_depth}")

        # Get feature names from the vectorizer
        feature_names = vectorizer.get_feature_names_out()

        # Get the feature importances from the decision tree
        importances_tree = best_tree.feature_importances_

        # Sort the features by importance
        indices_tree = np.argsort(importances_tree)[::-1]

        # Print the top 5 most important features
        print("Top 5 important features for Decision Tree:")
        for i in range(5):
            print(f"{i + 1}. {feature_names[indices_tree[i]]} ({importances_tree[indices_tree[i]]:.8f})")





######################### Random Forest  #########################
        random_forest = RandomForestClassifier(random_state=42, oob_score=True)

        # Define the hyperparameter grid
        param_grid_rf = {
            'n_estimators': [50, 100, 200, 300],  # Number of trees
            'max_features': ['sqrt', 'log2', None, 5, 10],  # Number of features to consider for splits
            'max_depth': [None, 10, 20, 30]  # Maximum depth of the trees
        }

        # GridSearchCV to find the best parameters
        grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)

        # Best model after hyperparameter tuning
        best_rf = grid_search_rf.best_estimator_

        # Step 3: Evaluate the Random Forest model on the test set
        y_pred_rf = best_rf.predict(X_test)

        # Get the OOB score
        oob_accuracy = best_rf.oob_score_

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
        # Get the maximum depth of each tree in the random forest
        max_depths = [tree.tree_.max_depth for tree in best_rf.estimators_]
        # Display the maximum depth of the entire forest (deepest tree)
        print(f"Maximum depth among all trees: {max(max_depths)}")
        print(f"OOB Accuracy: {oob_accuracy:.4f}")

        # Get the feature importances from the decision tree
        importances_rf = best_rf.feature_importances_

        # Sort the features by importance
        indices_tree = np.argsort(importances_rf)[::-1]

        # Print the top 5 most important features
        print("Top 5 important features for Random Forest:")
        for i in range(5):
            print(f"{i + 1}. {feature_names[indices_tree[i]]} ({importances_rf[indices_tree[i]]:.9f})")