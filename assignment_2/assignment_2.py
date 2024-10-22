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
reviews_training = []
for i in range(len(combined_reviews_training)):
    # make string lowercase and strip punctuation
    review = combined_reviews_training['review'][i].translate(str.maketrans('', '', string.punctuation)).lower()
    # remove numbers
    review = ''.join([i for i in review if not i.isdigit()])
    # remove stopwords
    word_tokens = word_tokenize(review)
    review = [w for w in word_tokens if not w.lower() in stop_words]
    reviews_training.append(review)

X_train = pd.DataFrame(reviews_training)
y_train = combined_reviews_training['label']
X_test = combined_reviews_testing['review']
y_test = combined_reviews_testing['label']

print(X_train)

# use CV for built-in cross validation, for now use C=1
regress = LogisticRegression(C=1, penalty='l1', solver='liblinear')
# regress.fit(X_train, y_train)
# score = regress.score(X_test, y_test)
# print(score)