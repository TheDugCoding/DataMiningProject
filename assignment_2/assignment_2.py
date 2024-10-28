import os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

# downloads and saves stopwords to remove
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

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

def clean_data(dataset):
    clean_list = []
    for i in range(len(dataset)):
        # remove punctuation
        review = dataset['review'][i].translate(str.maketrans('', '', string.punctuation)).lower()
        # remove numbers
        review = ''.join([i for i in review if not i.isdigit()])
        # remove stopwords
        word_tokens = word_tokenize(review)
        review = [w for w in word_tokens if not w.lower() in stop_words]
        clean_list.append(' '.join(review))
    return clean_list

def transform(list):
    vec = CountVectorizer()
    matrix = vec.fit_transform(list)
    df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out())
    return df

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
tdm_training = clean_data(combined_reviews_training)
tdm_testing = clean_data(combined_reviews_testing)

X_train = transform(tdm_training)
# remove sparse terms by using terms found at least 50 times
# could be done by appearance in % of docs instead
X_train = X_train[X_train.columns[X_train.sum() >= 50]]
y_train = combined_reviews_training['label']
X_test = transform(tdm_testing)
y_test = combined_reviews_testing['label']

# use CV for built-in cross validation, for now use C=1
regress = LogisticRegression(C=1, penalty='l1', solver='liblinear')
regress.fit(X_train, y_train)
score = regress.score(X_test, y_test)
print(score)