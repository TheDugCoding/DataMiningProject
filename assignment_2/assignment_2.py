import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

data_folder = os.path.join(os.getcwd(), 'assignment_2/data/')
training_folders = ['fold1', 'fold2', 'fold3', 'fold4']
testing_folders = ['fold5']
n_grams_range = [(1,1), (1,2), (2,2)]
removing_words = [2, 4, 6, 8, 10]

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
for n_gram in n_grams_range:
    for words_to_remove in removing_words:
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
        regress = LogisticRegressionCV(penalty='l1', solver='liblinear')
        regress.fit(X_train, y_train)
        y_pred = regress.predict(X_test)

        coefficients = regress.coef_[0]
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefficients)})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        scores = classification_report(y_test, y_pred)
        print(n_gram, words_to_remove, scores, feature_importance[:5])