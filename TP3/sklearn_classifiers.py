## This code is a simplier and modified version from TP2 code
import time
import string
import pprint
from collections import Counter
from utility import get_directory_content, bcolors #stuff not important are here

import nltk
from nltk.corpus import stopwords, wordnet

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from operator import itemgetter

import numpy as np

# Silent numpy warnings for better reporting 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Class used to test all the scenarios of test for SkLearn
# Some reminder about max_df and min_df = https://stackoverflow.com/questions/27697766/understanding-min-df-and-max-df-in-scikit-countvectorizer
class ClassifierTestSet:
    def __init__(self, name, classifier, stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False):
        self.name = name
        self.classifier = classifier
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.use_Tfid = use_Tfid
        self.binary = binary

    # To return all properties concatenated to the report header
    def str_keys(self):
        sb = []
        for key in self.__dict__:
            # TODO: this is not pythonic i guess...
            if (key != 'classifier'):
                sb.append("{key}".format(key=key, value=self.__dict__[key]))
 
        return ' | '.join(sb)

    # To return all the properties values concatenated to the report
    def __str__(self):
        sb = []
        for key in self.__dict__:
            # TODO: this is not pythonic i guess...
            if (key != 'classifier'):
                sb.append("{value}".format(key=key, value=self.__dict__[key]))
 
        return ' | '.join(sb)

# Most resources taken from http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# This class train and tests SkLearn classifiers
class SkLearnClassifier():
    def __init__(self, data_train, data_test, target_train, target_test, target_names):
        self.data_train = data_train
        self.data_test = data_test
        self.target_train = target_train
        self.target_test = target_test
        self.target_names = target_names

    def __create_pipeline(self, classifierTest):
        count_vectorizer = CountVectorizer(
            strip_accents = 'unicode',
            stop_words = classifierTest.stop_words, 
            lowercase = True, 
            max_df = classifierTest.max_df, 
            min_df = classifierTest.min_df,
            binary = classifierTest.binary
        )

        self.pipeline = Pipeline([
            ('vect', count_vectorizer),
            ('clf', classifierTest.classifier),
        ])

        self.classifier = classifierTest.classifier

        if (classifierTest.use_Tfid):
            self.pipeline.steps.insert(1,['tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)])
        else:
            pass
            #self.pipeline.steps.insert(1,['tfidf', TfidfTransformer()])

        self.pipeline.fit(self.data_train, self.target_train)  

    def __predict(self):
        self.predicted = self.pipeline.predict(self.data_test)

    def __mean(self):
        return np.mean(self.predicted == self.target_test)

    def show_analyses(self):
        print(metrics.classification_report(self.target_test, self.predicted, target_names=['angry','sad', 'happy', 'others']))

    def mean_from_classifier(self, classifierTest):
        self.__create_pipeline(classifierTest)
        self.__predict()
        return self.__mean()

    def predict(self, text):
        test_text = np.array([text])
        return self.pipeline.predict(test_text)

    # This method was extract from https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html 
    # I dont clain ownership of it, its just for evaluation purposes, to see how the classifier is trained
    def show_most_informative_features(self, text=None, n=7):
        # Extract the vectorizer and the classifier from the pipeline
        vectorizer = self.pipeline.named_steps['vect']
        classifier = self.pipeline.named_steps['clf']

        # Check to make sure that we can perform this computation
        if not hasattr(classifier, 'coef_'):
            raise TypeError(
                "Cannot compute most informative features on {}.".format(
                    classifier.__class__.__name__
                )
            )

        if text is not None:
            # Compute the coefficients for the text
            tvec = self.pipeline.transform([text]).toarray()
        else:
            # Otherwise simply use the coefficients
            tvec = classifier.coef_

        # Zip the feature names with the coefs and sort
        coefs = sorted(
            zip(tvec[0], vectorizer.get_feature_names()), key=itemgetter(0), reverse=True)

        # Get the top n and bottom n coef, name pairs
        topn  = zip(coefs[:n], coefs[:-(n+1):-1])

        # Create the output string to return
        output = []

        # If text, add the predicted value to the output.
        if text is not None:
            output.append("\"{}\"".format(text))
            output.append(
                "Classified as: {}".format(self.pipeline.predict([text]))
            )
            output.append("")

        # Create two columns with most negative and most positive features.
        for (cp, fnp), (cn, fnn) in topn:
            output.append(
                "{:0.4f}{: >15}    {:0.4f}{: >15}".format(
                    cp, fnp, cn, fnn
                )
            )

        print("\n".join(output)) 
