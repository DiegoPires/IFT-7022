import os
import pandas as pd
import numpy as np
from operator import itemgetter

from utility import get_complet_path, bcolors
from sklearn_classifiers import SkLearnClassifier, ClassifierTestSet
from keras_classifier import SimpleKerasClassifier

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def get_train_data():
    # Read our train data from file
    df = pd.read_table(get_complet_path('data/train.txt'), sep='\t', header=0, usecols=[1,2,3,4])
    df.fillna('', inplace=True)
    
    # Threat all the columns as one
    df['text'] = df[['turn1', 'turn2', 'turn3']].apply(lambda x: '; '.join(x), axis=1)
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

def get_predict_data():
    # Read our train data from file
    df = pd.read_table(get_complet_path('data/devwithoutlabels.txt'), sep='\t', header=0, usecols=[1,2,3])
    df.fillna('', inplace=True)
    
    # Threat all the columns as one
    df['text'] = df[['turn1', 'turn2', 'turn3']].apply(lambda x: '; '.join(x), axis=1)
    df.drop('turn1', inplace=True, axis=1)
    df.drop('turn2', inplace=True, axis=1)
    df.drop('turn3', inplace=True, axis=1)

    values = df["text"].values
    return np.ndenumerate(values)

# Prepare data and call classifiers
def main(verbose=False):
    data_train, data_test, target_train, target_test, target_names = get_train_data()
    
    test_with_sklearn_classifiers(data_train, data_test, target_train, target_test, target_names, verbose)
    #test_with_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose)

# Train a neural network with Keras and predict the conversations without label
def test_with_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose=False):
    
    trainedClassifier = SimpleKerasClassifier(data_train, data_test, target_train, target_test, target_names, verbose)

# Get best sklearn classifier using the test set and use it to predict the conversations without label
def test_with_sklearn_classifiers(data_train, data_test, target_train, target_test, target_names, verbose=False):

    classifiers = [
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),

        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),

        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),

        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.8, min_df=0.11, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
    ]
    
    if (verbose):
        headerClassifier = ClassifierTestSet('Header', None)
        print(headerClassifier.str_keys())

    results = []
    for classifier in classifiers: 
        skLearnClassifier = SkLearnClassifier(data_train, data_test, target_train, target_test, target_names)
        mean = skLearnClassifier.mean_from_classifier(classifier)

        results.append((skLearnClassifier, mean))

        if (verbose):
            print("{} | {}{}{}".format( 
                classifier,
                bcolors.BOLD,
                mean,
                bcolors.ENDC))
            #skLearnClassifier.show_most_informative_features(n=5)
            #skLearnClassifier.show_analyses()

    results.sort(key=itemgetter(1), reverse=True)
    best_classifier = results[0][0]

    print("{} ## The best classifier is: {} - {}{}".format(
        bcolors.HEADER,
        best_classifier.classifier,
        results[0][1],
        bcolors.ENDC))

    # Predicting all talks with our best classifier
    for _, text in get_predict_data():
        prediction = best_classifier.predict(text)
        print("# {}{}{} - {}".format(
            bcolors.WARNING,
            prediction,
            bcolors.ENDC,
            text))

if __name__ == '__main__':  
   main(verbose=False)