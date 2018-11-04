import nltk
import time
import string
import pprint
from collections import Counter
from utility import get_directory_content, bcolors #stuff not important are here

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

import numpy as np

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# DataSet to be used for SkLearn classifiers
class SentimentDataSet:
    def __init__(self):
        self.data = []
        self.target = []

# Class used to test all the scenarios of test for SkLearn
class ClassifierTestSet:
    def __init__(self, name, classifier, stop_words=None, max_df=1, min_df=0, use_Tfid=False, apply_stem=False, use_open_words=False):
        self.name = name
        self.classifier = classifier
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.use_Tfid = use_Tfid
        self.apply_stem=apply_stem
        self.use_open_words=use_open_words
        
    def __str__(self):
        sb = []
        for key in self.__dict__:
            # TODO: this is not pythonic i guess...
            if (key != 'classifier'):
                sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
 
        return ' | '.join(sb)

# Lot's of influence from these for the Naive Bayes
# https://www.twilio.com/blog/2017/09/sentiment-analysis-python-messy-data-nltk.html
# https://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
# https://stackoverflow.com/questions/20827741/nltk-naivebayesclassifier-training-for-sentiment-analysis

# This class is used to train NaiveBayesClassifier from NLTK package,
# everything is done manually: steamming, lemming, stop_words, minimal, frequency of words, etc
# BIGGEST DOWNPOINT: The training of process is VERY VERY slow (Up to 30 seconds)
class SentimentClassifier:

    def __init__(self, stemming, lemming, remove_stopwords, with_tagging):
        # Stuff that just need to be initialized one time...
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.global_stopwords = set(nltk.corpus.stopwords.words('english'))
        self.translator = str.maketrans('', '', string.punctuation)
        self.min_freq = 5

        self.bayes_classifier = None
        
        #initialize all other stuff
        self.stem = stemming 
        self.lemma = lemming
        self.remove_stopwords = remove_stopwords
        self.with_tagging = with_tagging

        self.sentiments = []
        self.vocabulary = []
        
    # Returns the tag of an word accordling to treebank
    def __get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV

    # Apply Lemmatizer or Stemmer for an array of tokens
    def __apply_lemma_or_stem(self, token):
        if (self.stem):
            token = self.stemmer.stem(token[0].lower())
        elif (self.lemma):
            if token[1] is None:
                token = self.lemmatizer.lemmatize(token[0].lower())
            else:
                token = self.lemmatizer.lemmatize(token[0].lower(), self.__get_wordnet_pos(token[1]))
        return token
   
    # Tokenize a phrase, apply Lemmatizer, or Stemmer, remove stopwords and filter allowed word tagging
    def __tokenize_and_normalize(self, phrase, stopwords = [], allowed_tags = []):
        phrase_without_ponctuation = phrase.translate(self.translator)
        tokens = word_tokenize(phrase_without_ponctuation)
        tokens_tagged = nltk.pos_tag(tokens)
        tokens = [self.__apply_lemma_or_stem(token) for token in tokens_tagged if token[0].lower() not in stopwords and (self.__get_wordnet_pos(token[1]) in allowed_tags or len(allowed_tags) == 0 )]
        return tokens

    # Give and array of phrases, returns an array of tokens
    def __apply_features(self, phrases, sentiment, verbose=False):

        start_time = time.time()

        vocabulary = []
        list_tokenized = []
        stopwords = self.global_stopwords if self.remove_stopwords else []
        allowed_tags = [wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV] if self.with_tagging else []

        for phrase in phrases:
            tokens = self.__tokenize_and_normalize(phrase, stopwords, allowed_tags)
            list_tokenized.append((tokens, sentiment))
            vocabulary.extend(tokens)
        
        counter = Counter(vocabulary)
        accepted_tokens = [key for key in counter.keys() if counter.get(key) >= self.min_freq]
        #filtered_vocabulary = [token for token in vocabulary if token in accepted_tokens]
        self.vocabulary.extend(accepted_tokens)

        filtered_list_tokenized =  []
        for item_tokenized in list_tokenized:
            filtered = [token for token in item_tokenized[0] if token in accepted_tokens]
            filtered_list_tokenized.append((filtered, item_tokenized[1]))

        if (verbose):
            print("{}# {:.2f} seconds to apply features <{}> to the Sentiment Classifier {}".format(bcolors.WARNING, (time.time() - start_time), sentiment, bcolors.ENDC))

        return filtered_list_tokenized

    # Format the array of tokens to be used in the NLTK Naive Bayes Classifier
    def __format_for_bayes(self, tokens):
        return ({ token: True for token in tokens })
    
    # Add a list of phrases corresponding to an review
    def add_sentiment(self, sentiment_list, sentiment, verbose=False):
        feature_set = self.__apply_features(sentiment_list, sentiment, verbose)
        self.sentiments.append(feature_set)

    # To create an object for the Naive Bayes classifier
    def create_bayes_classifier(self, verbose=False):
        start_time = time.time()

        training_set = []
        for sentiment_set in self.sentiments:
            for sentiment in sentiment_set:
                set = self.__format_for_bayes(sentiment[0])
                training_set.append(tuple((set, sentiment[1])))

        self.bayes_classifier = NaiveBayesClassifier.train(training_set)
        
        if (verbose):
            self.bayes_classifier.show_most_informative_features(30)
            print("{}# {:.2f} seconds to create NaiveBayes classifier{}".format(bcolors.WARNING, (time.time() - start_time), bcolors.ENDC))

    # After an Naive Bayes classifier was created, this is the use of it
    def classify_with_bayes(self, phrase, verbose=False):
        start_time = time.time()
        if self.bayes_classifier != None:
            tokens = self.__tokenize_and_normalize(phrase)
            classification = self.bayes_classifier.classify(self.__format_for_bayes(tokens))
        else:
            classification = 'NO CLASSIFIER TRAINED'
        
        if (verbose):
            print("{}# {:.2f} seconds to classify a phrase ({}...) with Naive Bayes classifier{}".format(bcolors.WARNING, (time.time() - start_time), phrase[:30], bcolors.ENDC))

        return classification

def validate_classification(classification, classification_should_be, verbose=False):
    if (classification == classification_should_be):
        msg = "{}## GOT IT RIGHT - {} {}".format(bcolors.OKBLUE, classification, bcolors.ENDC)
        validation = 1
    else:
        msg = "{}## GOT IT WRONG - {} - Should have been {}{}".format(bcolors.FAIL, classification, classification_should_be, bcolors.ENDC)
        validation = 0
    
    if (verbose):
        print(msg)
    return validation

def classify_nltk_naive_bayes(negative_reviews, positive_reviews):

    # 80% of first entries as the training set
    training_set_positive = positive_reviews[:int((.8)*len(positive_reviews))] 
    training_set_negative = negative_reviews[:int((.8)*len(negative_reviews))] 

    # The reste as test set
    test_set_positive = positive_reviews[int((.8)*len(positive_reviews)):]
    test_set_negative = negative_reviews[int((.8)*len(negative_reviews)):]

    # Initialize everything
    sentiClassifier = SentimentClassifier(
        stemming=True, 
        lemming=False, 
        remove_stopwords=True,
        with_tagging=True)

    sentiClassifier.add_sentiment(training_set_positive, "positive")
    sentiClassifier.add_sentiment(training_set_negative, "negative")
    
    sentiClassifier.create_bayes_classifier(verbose=False)
    
    good_bayes_guess_positive = 0
    for review in test_set_positive:
        classification = sentiClassifier.classify_with_bayes(review)
        good_bayes_guess_positive += validate_classification(classification, "positive", verbose=False)

    good_bayes_guess_negative = 0
    for review in test_set_negative:
        classification = sentiClassifier.classify_with_bayes(review)
        good_bayes_guess_negative += validate_classification(classification, "negative", verbose=False)
        
    print("{}### Naive Bayes - Results:{} \n\n{}/{} from positive reviews \n{}/{} from negative reviews\n".format(
        bcolors.HEADER, 
        bcolors.ENDC,
        good_bayes_guess_positive, len(test_set_positive), 
        good_bayes_guess_negative, len(test_set_negative)))
    
    # TODO

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Custom Vectorizer to be able to use Stemmer and filter open words       
class CustomCountVectorizer(CountVectorizer):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64, apply_stem=False, use_open_words=False):

        self.apply_stem = apply_stem
        self.use_open_words = use_open_words

        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents,
            lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words,
            token_pattern=token_pattern, ngram_range=ngram_range, analyzer=analyzer,
            max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

    def build_analyzer(self):
        analyzer = super(CustomCountVectorizer, self).build_analyzer()
        if (self.apply_stem):
            analyser_result = lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
        else:
            analyser_result = lambda doc: (w for w in analyzer(doc))

        return analyser_result
# Most resources taken from http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# This class train and tests SkLearn classifiers
class SkLearnClassifier():
    def __init__(self, data_train, data_test, target_train, target_test):
        # split the data between the training and testing
        self.data_train = data_train
        self.data_test = data_test
        self.target_train = target_train
        self.target_test = target_test

    def __create_pipeline(self, classifierTest):
        count_vectorizer = CustomCountVectorizer(
            strip_accents = 'unicode', 
            stop_words = classifierTest.stop_words, 
            lowercase = True, 
            max_df = classifierTest.max_df, 
            min_df = classifierTest.min_df
        )

        self.pipeline = Pipeline([
            ('vect', count_vectorizer),
            ('clf', classifierTest.classifier),
        ])

        if (classifierTest.use_Tfid):
            self.pipeline.steps.insert(1,['tfidf', TfidfTransformer()])

        self.pipeline.fit(self.data_train, self.target_train)  

    def __predict(self):
        self.predicted = self.pipeline.predict(self.data_test)

    def __mean(self):
        return np.mean(self.predicted == self.target_test)

    def show_analyses(self):
        print(metrics.classification_report(self.target_test, self.predicted, target_names=['negative','positive']))

    def mean_from_classifier(self, classifierTest):
        self.__create_pipeline(classifierTest)
        self.__predict()
        return self.__mean()

def classify_sklearn(negative_reviews, positive_reviews):
    # Create our data object with sentiments
    dataset = SentimentDataSet()
    dataset.data.extend(negative_reviews + positive_reviews)
    dataset.target.extend(['negative']*len(negative_reviews))
    dataset.target.extend(['positive']*len(positive_reviews))

    data_train, data_test, target_train, target_test  = train_test_split(
            dataset.data, 
            dataset.target,
            test_size=0.20,
            train_size=0.80)

    classifiers = [
            # This one was not required, just doing, to see the results...
            ClassifierTestSet('SGD ', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None), None, min_df=0.3, max_df=0.5, apply_stem=False),
            ClassifierTestSet('SGD ', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None), None, min_df=0.3, max_df=0.5, apply_stem=True),
            ClassifierTestSet('Naive Bayes', MultinomialNB()),
            ClassifierTestSet('Naive Bayes', MultinomialNB(), 'english'), 
            ClassifierTestSet('Naive Bayes', MultinomialNB(), 'english', min_df=0.1, max_df=0.9), 
            ClassifierTestSet('Naive Bayes', MultinomialNB(), 'english', min_df=0.3, max_df=0.9),
            ClassifierTestSet('Naive Bayes', MultinomialNB(), 'english', min_df=0.3, max_df=0.5), 
            ClassifierTestSet('Naive Bayes', MultinomialNB(), None, min_df=0.3, max_df=0.5), 
            #ClassifierTestSet('Naive Bayes', MultinomialNB(), 'english', min_df=0.3, max_df=0.5, use_Tfid=True), 
            ClassifierTestSet('Logistic Regression', LogisticRegression()),
            ClassifierTestSet('Logistic Regression', LogisticRegression(), 'english'),
            ClassifierTestSet('Logistic Regression', LogisticRegression(), 'english', min_df=0.1, max_df=0.9),
            ClassifierTestSet('Logistic Regression', LogisticRegression(), 'english', min_df=0.3, max_df=0.9),
            ClassifierTestSet('Logistic Regression', LogisticRegression(), 'english', min_df=0.3, max_df=0.5),
            ClassifierTestSet('Logistic Regression', LogisticRegression(), None, min_df=0.3, max_df=0.5)
            #ClassifierTestSet('Logistic Regression', LogisticRegression(), 'english', min_df=0.3, max_df=0.5, use_Tfid=True)
        ]
    
    for classifier in classifiers[:2]:
        skLearnClassifier = SkLearnClassifier(data_train, data_test, target_train, target_test)
        mean = skLearnClassifier.mean_from_classifier(classifier)

        print("# {} -  Mean: {}{}{}".format( 
            classifier,
            bcolors.BOLD,
            mean,
            bcolors.ENDC))

        #skLearnClassifier.show_analyses()

def main():

    negative_reviews = get_directory_content("Book/neg_Bk/*.text")
    positive_reviews = get_directory_content("Book/pos_Bk/*.text")

    # Results using all the criterias possibles
    # 83/200 from positive reviews  -- 41.5% succes for positive reviews
    # 185/200 from negative reviews -- 92.5% succes for negative reviews
    # classify_nltk_naive_bayes(negative_reviews, positive_reviews)
    
    classify_sklearn(negative_reviews, positive_reviews)

if __name__ == '__main__':  
   main()