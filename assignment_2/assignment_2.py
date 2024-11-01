import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

data_folder = os.path.join(os.getcwd(), 'assignment_2/data/')
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
list_models = []
for n_gram in [(1,2)]:
    for words_to_remove in (range(2, 30)):
        def transform(train, test):
            # strip punctuation
            train = train.str.replace('[^\w\s]','')
            test = test.str.replace('[^\w\s]','')
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
        regress = LogisticRegressionCV(penalty='l1', solver='liblinear')
        regress.fit(X_train, y_train)
        y_pred = regress.predict(X_test)

        coefficients = regress.coef_[0]
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})
        feature_importance_pos = feature_importance.sort_values('Importance', ascending=False)[:5]
        feature_importance_neg = feature_importance.sort_values('Importance', ascending=True)[:5]

        scores = classification_report(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        pre = round(precision_score(y_test, y_pred), 4)
        rec = round(recall_score(y_test, y_pred), 4)
        f1 = round(f1_score(y_test, y_pred), 4)
        c_value = regress.C_
        list_models.append([n_gram, words_to_remove, c_value, acc, pre, rec, f1])
        if n_gram == (1,2) and words_to_remove == 22:
            cm = confusion_matrix(y_test, y_pred)
            plot = sns.heatmap(cm, annot=True)
            fig = plot.get_figure()
            fig.savefig('confusion')
            fig.clf()

            feature_plot = sns.barplot(feature_importance_pos, x='Feature', y='Importance')
            feature_plot.set_title('Top 5 features for class 1')
            fig = feature_plot.get_figure()
            fig.savefig('feature_logistic_pos')
            fig.clf()

            feature_plot = sns.barplot(feature_importance_neg, x='Feature', y='Importance')
            feature_plot.set_title('Top 5 features for class 0')
            fig = feature_plot.get_figure()
            fig.savefig('feature_logistic_neg')
# print(*list_models, sep='\n')