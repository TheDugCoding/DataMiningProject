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
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
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
count_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# Chi-square feature selection
n_features = 800  # Adjust the number of features as needed
feature_selector = SelectKBest(score_func=chi2, k=n_features)
X_train_selected = feature_selector.fit_transform(X_train_count, y_train)
X_test_selected = feature_selector.transform(X_test_count)

# Train Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_selected, y_train)
y_pred_nb = nb_classifier.predict(X_test_selected)


######################### Logistic Regression #########################

# Tfidf Vectorization for Logistic Regression
vectorizer_lr = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=22, stop_words='english')
X_train_lr = vectorizer_lr.fit_transform(X_train)
X_test_lr = vectorizer_lr.transform(X_test)

# Train Logistic Regression
regress = LogisticRegressionCV(penalty='l1', solver='liblinear', max_iter=100)
regress.fit(X_train_lr, y_train)
y_pred_lr = regress.predict(X_test_lr)


######################### Decision Tree #########################

# Tfidf Vectorization for Decision Tree
vectorizer_dt = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5, stop_words='english')
X_train_dt = vectorizer_dt.fit_transform(X_train)
X_test_dt = vectorizer_dt.transform(X_test)

# Train Decision Tree
decision_tree = DecisionTreeClassifier(ccp_alpha=0.03, criterion='entropy', random_state=42)
decision_tree.fit(X_train_dt, y_train)
y_pred_dt = decision_tree.predict(X_test_dt)


######################### Random Forest #########################

# Tfidf Vectorization for Random Forest
vectorizer_rf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=8, stop_words='english')
X_train_rf = vectorizer_rf.fit_transform(X_train)
X_test_rf = vectorizer_rf.transform(X_test)

# Train Random Forest
best_rf = RandomForestClassifier(max_features='log2', n_estimators=300, oob_score=True, random_state=42)
best_rf.fit(X_train_rf, y_train)
y_pred_rf = best_rf.predict(X_test_rf)


######################### Collecting Test Metrics #########################

# Function to collect metrics
def collect_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Store metrics in a DataFrame
metrics = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])

# Collect metrics for each model
models = ["Naive Bayes", "Logistic Regression", "Decision Tree", "Random Forest"]
predictions = [y_pred_nb, y_pred_lr, y_pred_dt, y_pred_rf]

for model_name, y_pred in zip(models, predictions):
    accuracy, precision, recall, f1 = collect_metrics(y_test, y_pred)
    metrics = metrics._append({"Model": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}, ignore_index=True)

print(metrics)
