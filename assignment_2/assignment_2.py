import os
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
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

# Plot performance metrics for both train and test sets
plt.figure(figsize=(14, 6))
plt.bar(metric_names, metric_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan'])
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.title('Model Performance Metrics for Train and Test Sets')
for i, v in enumerate(metric_values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.savefig('chi_800.png', bbox_inches='tight')  # Save plot
plt.show()

# Retrieve Chi-square scores for each feature and corresponding names
chi2_scores = feature_selector.scores_[selected_features_mask]
selected_features_scores = list(zip(selected_features, chi2_scores))

# Separate features based on class
# Sort features by score in descending order to find the top features
top_features_0 = sorted(selected_features_scores, key=lambda x: x[1], reverse=True)[:5]
top_features_1 = sorted(selected_features_scores, key=lambda x: x[1], reverse=True)[:5]

# Display the top 5 features for each class
print("\nTop 5 Most Important Features for Class 0 (Deceptive):")
for feature, score in top_features_0:
    print(f"{feature}: {score:.4f}")

print("\nTop 5 Most Important Features for Class 1 (Truthful):")
for feature, score in top_features_1:
    print(f"{feature}: {score:.4f}")

# Prepare data for top features for each class
features_0, scores_0 = zip(*top_features_0)
features_1, scores_1 = zip(*top_features_1)

# Plot for Class 0 (Deceptive)
plt.figure(figsize=(8, 5))
plt.barh(features_0, scores_0, color='skyblue')
plt.xlabel('Chi-square Score')
plt.title('Top 5 Features for Class 0 (Deceptive)')
plt.xlim(0, max(scores_0) + 0.1)  # Add some padding to the x-axis
plt.grid(axis='x')
plt.savefig('chi_800_top_features_class_0.png', bbox_inches='tight')  # Save plot
plt.show()

# Plot for Class 1 (Truthful)
plt.figure(figsize=(8, 5))
plt.barh(features_1, scores_1, color='lightgreen')
plt.xlabel('Chi-square Score')
plt.title('Top 5 Features for Class 1 (Truthful)')
plt.xlim(0, max(scores_1) + 0.1)  # Add some padding to the x-axis
plt.grid(axis='x')
plt.savefig('chi_800_top_features_class_1.png', bbox_inches='tight')  # Save plot
plt.show()