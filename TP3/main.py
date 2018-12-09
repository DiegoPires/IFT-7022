import os
import pandas as pd
import numpy as np
from operator import itemgetter

from utility import get_complet_path, bcolors, clean_results
from sklearn_classifiers import SkLearnClassifier, ClassifierTestSet

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from keras_classifier import remove_saved_keras_models, get_simple_keras_classifier, get_denser_keras_classifier, get_denser_keras_classifier_with_tokenizer
from keras_classes import DataDTO, CountVectorizerDTO, KerasTokenizerDTO

# This could help to extract features from text: https://www.kaggle.com/kmader/toxic-emojis
# And this too: https://nlpforhackers.io/sentiment-analysis-intro/

# Loads the data for training and evaluation
def get_train_data():
    # Read our train data from file
    df = pd.read_table(get_complet_path('data/train.txt'), sep='\t', header=0, usecols=[1,2,3,4])
    df.fillna('', inplace=True)
    
    # Threat all the columns as one
    df['text'] = df[['turn1', 'turn2', 'turn3']].apply(lambda x: "<p>" + "</p><p>".join(x) + "</p>" , axis=1)
    df.drop('turn1', inplace=True, axis=1)
    df.drop('turn2', inplace=True, axis=1)
    df.drop('turn3', inplace=True, axis=1)

    texts = df["text"].values
    labels = df["label"].values

    target_names = df.label.unique()
    
    # Create our data object with sentiments
    data_train, data_test, target_train, target_test = train_test_split(
            texts,
            labels,
            test_size=0.20,
            train_size=0.80,
            random_state=1000)

    return data_train, data_test, target_train, target_test, target_names

# Loads the data do be predicted
def get_predict_data():
    # Read our train data from file
    df = pd.read_table(get_complet_path('data/devwithoutlabels.txt'), sep='\t', header=0, usecols=[1,2,3])
    df.fillna('', inplace=True)
    
    # Threat all the columns as one
    df['text'] = df[['turn1', 'turn2', 'turn3']].apply(lambda x: "<p>" + "</p><p>".join(x) + "</p>", axis=1)
    df.drop('turn1', inplace=True, axis=1)
    df.drop('turn2', inplace=True, axis=1)
    df.drop('turn3', inplace=True, axis=1)

    values = df["text"].values
    return np.ndenumerate(values)

# Train multiple SkLearn Classifiers differents, get the best result and predict the texts without label
def test_with_sklearn_classifiers(data_train, data_test, target_train, target_test, target_names, verbose=False):

    classifiers = [
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.8, min_df=0.11, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.8, min_df=0.11, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
    ]
    
    if (verbose):
        headerClassifier = ClassifierTestSet('Header', None)
        print(headerClassifier.str_keys())

    results = []
    for classifier in classifiers: 
        skLearnClassifier = SkLearnClassifier(data_train, data_test, target_train, target_test, target_names)
        skLearnClassifier.train_classifier(classifier, False)
        
        write_classifier_result_to_file('sklearn_classifiers.txt', skLearnClassifier)
        results.append(skLearnClassifier)

    return predict_with_best(results, 'sklearn_prediction.txt')

# Train multiple keras classifiers differents, takes the best one and predict the texts without label
def test_with_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose=False, remove_models=False):
    
    remove_saved_keras_models(remove_models)
    data_dto = DataDTO(data_train, data_test, target_train, target_test, target_names) # 15000

    results = []
    results.append(get_simple_keras_classifier('simple', data_dto, verbose=verbose))
    results.append(get_denser_keras_classifier('denser', data_dto, verbose=verbose))
    results.append(get_denser_keras_classifier_with_tokenizer('denser_and_tokenizer', data_dto, verbose=verbose))

    for keras_c in results:
        write_classifier_result_to_file('keras_classifiers.txt', keras_c)

    return predict_with_best(results, 'keras_prediction.txt')

# Finds the best classifier and use it to predict the texts
def predict_with_best(results, file_results_name):
    results.sort(key=lambda x: x.accuracy, reverse=True)
    best_classifier = results[0]

    # Just show top 10
    print ("\n\n{}## The top 10 of classifiers: {}{}".format(bcolors.HEADER, type(best_classifier), bcolors.ENDC))
    print ("\nClassifier|Accuracy")
    for classifier in results[:10]:
        print("{}|{}{}{}".format(
            classifier, 
            bcolors.WARNING, 
            classifier.accuracy, 
            bcolors.ENDC))

    print ("\n\n{}## 10 first predictions for: {}{}".format(bcolors.HEADER, type(best_classifier), bcolors.ENDC))
    print ("\nPrediction|Sentence")
    
    predictions = []
    # Predicting all talks with our best classifier
    for index, text in get_predict_data():
        prediction = best_classifier.predict(text)[0] # 0 to remove from numpy array
        predictions.append(prediction)
    
        write_results_to_file(file_results_name, prediction, text)
        if (index[0] <= 10):
            print ("{}{}{}|{}".format(
                bcolors.WARNING,
                prediction,
                bcolors.ENDC,
                text))

    return np.array(predictions)

# Writes the result for classifier on file
def write_classifier_result_to_file(file, classifier):
    path = get_complet_path('results/' + file)
    if not os.path.exists(path):
        highscore = open(path, 'w')
        highscore.write("classifier|accuracy\n")
        highscore.close()    

    highscore = open(path, 'a')
    highscore.write(str(classifier) + '|' + str(classifier.accuracy) + '\n')
    highscore.close()

# writes the results for prediction on file
def write_results_to_file(file, prediction, text):
    path = get_complet_path('results/' + file)
    if not os.path.exists(path):
        highscore = open(path, 'w')
        highscore.write("prediction|text\n")
        highscore.close()  

    highscore = open(path, 'a')
    highscore.write(prediction + '|' + text + '\n')
    highscore.close()

# Prepare data and call classifiers
def main(verbose=False, remove_saved_keras_models=False):
    data_train, data_test, target_train, target_test, target_names = get_train_data()
    
    sk_predictions = test_with_sklearn_classifiers(data_train, data_test, target_train, target_test, target_names, verbose)
    ke_predictions = test_with_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose, remove_saved_keras_models)

    mean_between_results = np.mean(sk_predictions == ke_predictions)

    print("\n\n### Difference between best classifiers is: {}{:.4f}{}".format(
            bcolors.OKBLUE,
            mean_between_results,
            bcolors.ENDC
            ))

if __name__ == '__main__':
    clean_results()
    main(verbose=False, remove_saved_keras_models=False) # TODO: Change this to receive args from command prompt