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


######################### Mc Nemar's Tests with Bonferroni Correction #########################

# Define a function for McNemar's test
def mcnemar_test(y_true, y_pred1, y_pred2):
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    # Construct the 2x2 contingency table
    table = np.zeros((2, 2))
    table[0, 0] = np.sum(correct1 & correct2)    # Both correct
    table[0, 1] = np.sum(~correct1 & correct2)   # Model 2 correct, Model 1 wrong
    table[1, 0] = np.sum(correct1 & ~correct2)   # Model 1 correct, Model 2 wrong
    table[1, 1] = np.sum(~correct1 & ~correct2)  # Both wrong

    result = mcnemar(table, exact=True)
    return result.pvalue

# Pairwise McNemar tests with Bonferroni correction
# Alpha level with Bonferroni correction for 6 comparisons (4 models, C(4, 2) = 6)
alpha = 0.05
bonferroni_alpha = alpha / 6

# Conduct pairwise tests
print(f"Using Bonferroni-corrected alpha level: {bonferroni_alpha:.4f}\n")

# Model pairs for comparison
model_names = ["Naive Bayes", "Logistic Regression", "Decision Tree", "Random Forest"]
model_preds = [y_pred_nb, y_pred_lr, y_pred_dt, y_pred_rf]

for i in range(len(model_preds)):
    for j in range(i + 1, len(model_preds)):
        p_value = mcnemar_test(y_test, model_preds[i], model_preds[j])
        print(f"{model_names[i]} vs {model_names[j]} p-value: {p_value:.4f}")
        if p_value < bonferroni_alpha:
            print("Significant difference after Bonferroni correction.\n")
        else:
            print("No significant difference after Bonferroni correction.\n")
