import nltk
import time
import string
from collections import Counter
from utility import get_directory_content, bcolors
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

class SentimentClassifier:

    def __init__(self, stemming, lemming, remove_stopwords, with_tagging):
        # Stuff that just need to be initialized one time...
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.global_stopwords = set(nltk.corpus.stopwords.words('english'))
        self.translator = str.maketrans('', '', string.punctuation)
        self.min_freq = 5
        
        #initialize all other stuff
        self.stem = stemming 
        self.lemma = lemming
        self.remove_stopwords = remove_stopwords
        self.with_tagging = with_tagging

        self.sentiments = []

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

    # After 
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

def main():
    negative_reviews = get_directory_content("Book/neg_Bk/*.text")
    positive_reviews = get_directory_content("Book/pos_Bk/*.text")

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

    sentiClassifier.add_sentiment(training_set_negative, "negative")
    sentiClassifier.add_sentiment(training_set_positive, "positive")
    
    #### NAIVE BAYES
    sentiClassifier.create_bayes_classifier(verbose=False)

    good_guess_positive = 0
    for review in test_set_positive:
        classification = sentiClassifier.classify_with_bayes(review)
        good_guess_positive += validate_classification(classification, "positive", verbose=False)

    good_guess_negative = 0
    for review in test_set_negative:
        classification = sentiClassifier.classify_with_bayes(review)
        good_guess_negative += validate_classification(classification, "negative", verbose=False)

    print("{}### Naive Bayes - Results:{} \n\n{}/{} from positive reviews \n{}/{} from negative reviews\n".format(
        bcolors.HEADER, 
        bcolors.ENDC,
        good_guess_positive, len(test_set_positive), 
        good_guess_negative, len(test_set_negative)))

    ### LINEAR REGRESSION

    # TODO

if __name__ == '__main__':  
   main()