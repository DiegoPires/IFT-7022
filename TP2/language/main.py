import nltk
from utility import read_file, get_directory_content, bcolors #stuff not important are here

import nltk.data

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import numpy as np

# Inspiration came from: http://michelleful.github.io/code-blog/2015/06/18/classifying-roads/

class ClassifierTest:
    def __init__(self, name, classifier, ngram):
        self.name = name
        self.classifier = classifier
        self.ngram = ngram
    def __str__(self):
        sb = []
        for key in self.__dict__:
            # TODO: this is not pythonic i guess...
            if (key != 'classifier'):
                sb.append("{key}={value}".format(key=key, value=self.__dict__[key]))
 
        return ' | '.join(sb)

class Classifier:
    def __init__(self, classifier_test, language, texts, verbose=False):

        np_texts = np.array(texts)
        np_language = np.array(language)

        self.vectorizer = CountVectorizer(
            lowercase=True,
            analyzer='char',
            ngram_range=(1, classifier_test.ngram))
    
        np_texts = self.vectorizer.fit_transform(np_texts)

        if (verbose):
            print(self.vectorizer.get_feature_names())

        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', classifier_test.classifier) ])
        
        self.pipeline.fit(texts, np_language)

    def predict(self, text):
        test_text = np.array([text])
        return self.pipeline.predict(test_text)

# Prepare data to be used
def prepare_data():
    corpus_languages = ['english', 'spanish', 'french', 'portuguese']
    
    language = []
    texts = []
    
    for corpus_language in corpus_languages:
        text = read_file('identification_langue/corpus_entrainement/{}-training.txt'.format(corpus_language))
        tokenizer = nltk.data.load('tokenizers/punkt/{}.pickle'.format(corpus_language))

        sentences = tokenizer.tokenize(text.strip())
        texts.extend(sentences)
        language.extend([corpus_language]*len(sentences))

    return texts, language
    
def main():
    
    # Data to train/test
    sentences, language = prepare_data()
    tests = get_directory_content("identification_langue/corpus_test1/*.txt")
    
    # Use cases for test
    test_cases = [
        ClassifierTest('Naive Bayes', MultinomialNB(), 1),
        ClassifierTest('Naive Bayes', MultinomialNB(), 2),
        ClassifierTest('Naive Bayes', MultinomialNB(), 3),
        ClassifierTest('Linear Regression', LogisticRegression(), 1),
        ClassifierTest('Linear Regression', LogisticRegression(), 2),
        ClassifierTest('Linear Regression', LogisticRegression(), 3),
        #ClassifierTest('KNeighborsClassifier', KNeighborsClassifier(3), 1), # Strange results
        #ClassifierTest('SVC', SVC(kernel="linear", C=0.025), 2), # Always predict right, not matter n-gram, maybe it ignores it?
        #ClassifierTest('SVC', SVC(gamma=2, C=1), 3), # Never works
        #ClassifierTest('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=5), 1), # Strange results
        #ClassifierTest('GaussianNB', GaussianNB(), 1), # Doenst work...,
        #ClassifierTest('SGD ', SGDClassifier(loss='hinge', penalty='l2',
        #                                    alpha=1e-3, random_state=42,
        #                                    max_iter=5, tol=None), 1) # Strange results
    ]

    for test_case in test_cases: #[:1]:
        classifier = Classifier(test_case, language, sentences, verbose=False)

        print("{}{}{}".format(bcolors.HEADER, test_case, bcolors.ENDC))

        for test in tests:
            prediction = classifier.predict(test)
            print ('# Prediction: {} | Text: {}'.format(prediction, test[:70].replace("\n", "")))
    
if __name__ == '__main__':  
   main()