## This code is a simplier and modified version from TP2 code
import time
import string
import pprint
from collections import Counter
from utility import get_directory_content, bcolors #stuff not important are here

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TweetTokenizer

from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag

from operator import itemgetter

import numpy as np

from classifier import Classifier

# Silent numpy warnings for better reporting 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Some good stuff to be tested from here (But still not used): https://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost

# Example from here: https://scikit-learn.org/0.19/auto_examples/hetero_feature_union.html
class ExtraCountFeature(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features=features

    def fit(self, X, y=None):
        return self

    def transform(self, phrases):
        feature_list = []
        for text in phrases:
            data = {}
            for feature in self.features:
                data.update({ feature : text.count(feature) })
            feature_list.append(data)
        
        return feature_list

# https://nlpforhackers.io/sentiment-analysis-intro/
class ExtraSentimentFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tt = TweetTokenizer()
        self.lemm = WordNetLemmatizer()
    
    def penn_to_wn(self, tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None
    
    def clean_text(self, text):
        text = text.replace("<p>", "").replace("</p>", " ")
        text = text.decode("utf-8")
    
        return text
        
    def fit(self, X, y=None):
        return self

    def get_sentiment(self, phrase):
        tagged_sentence = pos_tag(self.tt.tokenize(phrase))
    
        negative = 0
        positive = 0
        for word, tag in tagged_sentence:
            wn_tag = self.penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = self.lemm.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment = swn_synset.pos_score() - swn_synset.neg_score()

            if sentiment >= 0:
                positive = positive + 1
            else:
                negative = negative + 1

        return { 'positive' : positive, 'negative' : negative }

    def transform(self, phrase):
        resultat = [ self.get_sentiment(x) for x in phrase ]
        return resultat

# Class used to test all the scenarios of test for SkLearn
# Some reminder about max_df and min_df = https://stackoverflow.com/questions/27697766/understanding-min-df-and-max-df-in-scikit-countvectorizer
class ClassifierTestSet:
    def __init__(self, name, classifier, stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1, 1), apply_count_features=False, apply_sentiment_features=False):
        self.name = name
        self.classifier = classifier
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.use_Tfid = use_Tfid
        self.binary = binary
        self.ngram_range=ngram_range
        self.apply_count_features = apply_count_features
        self.apply_sentiment_features = apply_sentiment_features

    # To return all properties concatenated to the report header
    def str_keys(self):
        sb = []
        for key in self.__dict__:
            # TODO: this is not pythonic i guess...
            if (key != 'classifier'):
                sb.append("{key}".format(key=key, value=self.__dict__[key]))
 
        return '|'.join(sb)

    # To return all the properties values concatenated to the report
    def __str__(self):
        sb = []
        for key in self.__dict__:
            if (key != 'classifier'):
                sb.append("{value}".format(key=key, value=self.__dict__[key]))
 
        return '|'.join(sb)

# Most resources taken from http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# This class train and tests SkLearn classifiers
class SkLearnClassifier(Classifier):
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
            binary = classifierTest.binary,
            ngram_range = classifierTest.ngram_range
        )

        index_classifier = 1
        transformer_list = []

        text_pipeline = Pipeline([
            ('vect', count_vectorizer)
        ])

        if (classifierTest.use_Tfid):
            text_pipeline.steps.insert(index_classifier,['tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)])
            index_classifier = index_classifier + 1

        transformer_list.append(('text', text_pipeline))

        if (classifierTest.apply_count_features):
            transformer_list.append(('emojis', Pipeline([
                ('count', ExtraCountFeature(['😍','😁','😭','😂','😠','💔','😞', ':)', ':(', '!', '?'])),
                ('vect', DictVectorizer())
            ])))
        
        if (classifierTest.apply_sentiment_features):
            transformer_list.append(('sentiment', Pipeline([
                ('count', ExtraSentimentFeature()),
                ('vect', DictVectorizer())
            ])))

        # Just use the FeatureUnion if we have at least one of the extra features, otherwise crashs
        if (len(transformer_list) == 1):
            text_pipeline.steps.insert(index_classifier,['clf', classifierTest.classifier])
            self.pipeline = text_pipeline
        else:
            self.pipeline = Pipeline([
                ('features', FeatureUnion(
                    transformer_list=transformer_list
                )),
                ('clf', classifierTest.classifier)
            ])

        self.classifier = classifierTest.classifier
        self.classifier_test = classifierTest
        
        self.pipeline.fit(self.data_train, self.target_train)  

    def __predict(self):
        self.predicted = self.pipeline.predict(self.data_test)

    def __mean(self):
        self.accuracy = np.mean(self.predicted == self.target_test)

    def show_analyses(self):
        print(metrics.classification_report(self.target_test, self.predicted, target_names=['angry','sad', 'happy', 'others']))

    def train_classifier(self, classifierTest, verbose):
        self.__create_pipeline(classifierTest)
        self.__predict()
        self.__mean()
        
        if (verbose):
            print("{} | {}{}{}".format( 
                classifierTest,
                bcolors.BOLD,
                self.accuracy,
                bcolors.ENDC))

            #skLearnClassifier.show_most_informative_features(n=5)
            #skLearnClassifier.show_analyses()

    def predict(self, text):
        test_text = np.array([text])
        return self.pipeline.predict(test_text)

    def __str__(self):
        return str(self.classifier_test)

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
