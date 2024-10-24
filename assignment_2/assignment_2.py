import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
reviews_training = []
for i in range(len(combined_reviews_training)):
    # make string lowercase and strip punctuation
    review = combined_reviews_training['review'][i].translate(str.maketrans('', '', string.punctuation)).lower()
    # remove numbers
    review = ''.join([i for i in review if not i.isdigit()])
    # remove stopwords
    word_tokens = word_tokenize(review)
    review = [w for w in word_tokens if not w.lower() in stop_words]
    reviews_training.append(' '.join(review))

X_train = pd.DataFrame(reviews_training, columns=['review'])
y_train = combined_reviews_training['label']
X_test = combined_reviews_testing['review']
y_test = combined_reviews_testing['label']

# feature selection
n_features = 10
feature_selector = SelectKBest(score=mutual_info_classif, k=n_features)
X_train = feature_selector.fit_transform(X_train, y_train)
X_test = feature_selector.transform(X_test, y_test)

# Print mutual information scores
if hasattr(X, 'columns'):  # For pandas DataFrame
    feature_mask = feature_selector.get_support()
    selected_features = X.columns[feature_mask].tolist()
    mi_scores = feature_selector.scores_
    
    print("\nMutual Information scores for selected features:")
    selected_features_scores = [(name, score) for name, score, selected 
                                in zip(X.columns, mi_scores, feature_mask) 
                                if selected]
    for name, score in sorted(selected_features_scores, 
                            key=lambda x: x[1], reverse=True):
        print(f"{name}: {score:.4f}")

# Estimate Multinomial Naive Bayes
