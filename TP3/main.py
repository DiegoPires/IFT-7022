import os
import pandas as pd
import numpy as np
from operator import itemgetter

from utility import get_complet_path, bcolors
from sklearn_classifiers import SkLearnClassifier, ClassifierTestSet

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from keras_classifier import remove_saved_keras_models, get_simple_keras_classifier, get_denser_keras_classifier, get_denser_keras_classifier_with_tokenizer

# Loads the data for training and evaluation
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

# Loads the data do be predicted
def get_predict_data():
    # Read our train data from file
    df = pd.read_table(get_complet_path('data/devwithoutlabels.txt'), sep='\t', header=0, usecols=[1,2,3])
    df.fillna('', inplace=True)
    
    # Threat all the columns as one
    df['text'] = df[['turn1', 'turn2', 'turn3']].apply(lambda x: '; '.join(x), axis=1)
    df.drop('turn1', inplace=True, axis=1)
    df.drop('turn2', inplace=True, axis=1)
    df.drop('turn3', inplace=True, axis=1)

    values = df["text"].values[:50] # TODO: Just returning 10 for testing
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
    for classifier in classifiers[:2]: 
        skLearnClassifier = SkLearnClassifier(data_train, data_test, target_train, target_test, target_names)
        skLearnClassifier.train_classifier(classifier, verbose)
        
        results.append(skLearnClassifier)

    return predict_with_best(results)

# Train multiple keras classifiers differents, takes the best one and predict the texts without label
def test_with_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose=False, remove_models=False):
    
    remove_saved_keras_models(remove_models)

    results = []
    results.append(get_simple_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose))
    results.append(get_denser_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose))
    results.append(get_denser_keras_classifier_with_tokenizer(data_train, data_test, target_train, target_test, target_names, verbose))

    return predict_with_best(results)

# Finds the best classifier and use it to predict the texts
def predict_with_best(results):
    results.sort(key=lambda x: x.accuracy, reverse=True)
    best_classifier = results[0]

    print("\n\n{}## The best {} is: {} - Accuracy on training: {} {}".format(
        bcolors.HEADER,
        type(best_classifier),
        best_classifier,
        best_classifier.accuracy,
        bcolors.ENDC))

    predictions = []
    # Predicting all talks with our best classifier
    for _, text in get_predict_data():
        prediction = best_classifier.predict(text)[0] # 0 to remove from numpy array
        predictions.append(prediction)
        
        print("# {}{}{} - {}".format(
            bcolors.WARNING,
            prediction,
            bcolors.ENDC,
            text))

    return np.array(predictions)

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
   main(verbose=False, remove_saved_keras_models=False) # TODO: Change this to receive args from command prompt