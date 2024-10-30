import os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

nltk.download('punkt_tab')


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

# pre-processing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    # Check if text is a string, if not, convert or skip
    if not isinstance(text, str):
        text = str(text)
    # Make string lowercase and remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # Tokenize
    word_tokens = word_tokenize(text)
    # Remove stopwords
    text = [w for w in word_tokens if w.lower() not in stop_words]
    return ' '.join(text)

X_train = combined_reviews_training['review'].apply(preprocess_text)
y_train = combined_reviews_training['label']
X_test = combined_reviews_testing['review']
y_test = combined_reviews_testing['label']

# TF-IDF Vectorization
print("Performing TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# feature selection
n_features = 10
feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = feature_selector.transform(X_test_tfidf)

# Train Naive Bayes
print("Training Naive Bayes model")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_selected, y_train)

# Make predictions
print("Making predictions")
y_pred = nb_classifier.predict(X_test_selected)

# Print performance metrics
print("\nClassification Report:")
classification_report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report)

# Get feature importance information
feature_names = tfidf_vectorizer.get_feature_names_out()
selected_features_mask = feature_selector.get_support()
selected_features = feature_names[selected_features_mask]
feature_scores = feature_selector.scores_[selected_features_mask]

# Print top features
print("\nTop 10 Most Important Features:")
top_features = sorted(zip(selected_features, feature_scores), key=lambda x: x[1], reverse=True)[:10]
for feature, score in top_features:
    print(f"{feature}: {score:.4f}")

# Print model parameters
print("\nModel Parameters:")
print("Number of features selected:", n_features)
print("Class prior probabilities:", nb_classifier.class_prior)
print("Number of samples used for training:", X_train_selected.shape[0])

# Print feature sparsity information
print("\nFeature Sparsity Information:")
sparsity = 1.0 - (X_train_selected.nnz / float(X_train_selected.shape[0] * X_train_selected.shape[1]))
print(f"Sparsity of feature matrix: {sparsity:.2%}")

# Plot metrics
accuracy = classification_report['accuracy']
precision = classification_report['1']['precision']
recall = classification_report['1']['recall']
f1 = classification_report['1']['f1-score']
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(8, 5))
plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
for i, v in enumerate(metric_values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()
