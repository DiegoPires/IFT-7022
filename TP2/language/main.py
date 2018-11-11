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
    def str_keys(self):
        sb = []
        for key in self.__dict__:
            # TODO: this is not pythonic i guess...
            if (key != 'classifier'):
                sb.append("{key}".format(key=key, value=self.__dict__[key]))
 
        return ' | '.join(sb)
    def __str__(self):
        sb = []
        for key in self.__dict__:
            # TODO: this is not pythonic i guess...
            if (key != 'classifier'):
                sb.append("{value}".format(key=key, value=self.__dict__[key]))
 
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
    
def main(verbose):
    
    # Data to train/test
    sentences, language = prepare_data()
    tests_language, tests_text = get_directory_content("identification_langue/corpus_test1/*.txt")
    
    # Use cases for test
    test_cases = [
        
        ClassifierTest('MultinomialNB', MultinomialNB(), 1),
        ClassifierTest('MultinomialNB', MultinomialNB(), 2),
        ClassifierTest('MultinomialNB', MultinomialNB(), 3),
        ClassifierTest('LogisticRegression', LogisticRegression(), 1),
        ClassifierTest('LogisticRegression', LogisticRegression(), 2),
        ClassifierTest('LogisticRegression', LogisticRegression(), 3),
        
        ClassifierTest('KNeighborsClassifier 3 neighbors', KNeighborsClassifier(3), 1), # Strange predictions
        ClassifierTest('KNeighborsClassifier 3 neighbors', KNeighborsClassifier(3), 2), # Strange predictions
        ClassifierTest('KNeighborsClassifier 3 neighbors', KNeighborsClassifier(3), 3), # Strange predictions
        ClassifierTest('KNeighborsClassifier 5 neighbors', KNeighborsClassifier(5), 1), # Strange predictions
        ClassifierTest('KNeighborsClassifier 5 neighbors', KNeighborsClassifier(5), 2), # Strange predictions
        ClassifierTest('KNeighborsClassifier 5 neighbors', KNeighborsClassifier(5), 3), # Strange predictions
        
        ClassifierTest('Linear SVC', LinearSVC(random_state=0, tol=1e-5), 1), # -
        ClassifierTest('Linear SVC', LinearSVC(random_state=0, tol=1e-5), 2), # GOOD
        ClassifierTest('Linear SVC', LinearSVC(random_state=0, tol=1e-5), 3), # GOOD
        
        ClassifierTest('SVC gamma auto', SVC(gamma='auto'), 1), # strange
        ClassifierTest('SVC gamma auto', SVC(gamma='auto'), 2), # strange
        ClassifierTest('SVC gamma auto', SVC(gamma='auto'), 3), # strange
        
        ClassifierTest('SVC linear', SVC(kernel="linear", C=0.025), 1), # This linear with 1 gram its better than the other class
        ClassifierTest('SVC linear', SVC(kernel="linear", C=0.025), 2), # GOOD
        ClassifierTest('SVC linear', SVC(kernel="linear", C=0.025), 3), # GOOD

        ClassifierTest('SVC gamma 2', SVC(gamma=2, C=1), 1), # always english...
        ClassifierTest('SVC gamma 2', SVC(gamma=2, C=1), 2), # always english...
        ClassifierTest('SVC gamma 2', SVC(gamma=2, C=1), 3), # always english...

        ClassifierTest('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=5), 1), # very bad
        ClassifierTest('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=5), 2), # Strange results
        ClassifierTest('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=5), 3), # Strange results
        
        ClassifierTest('SGDClassifier ', SGDClassifier(max_iter=1000), 1), # 
        ClassifierTest('SGDClassifier ', SGDClassifier(max_iter=1000), 2), # GOOD
        ClassifierTest('SGDClassifier ', SGDClassifier(max_iter=1000), 3), # GOOD      

        ### ClassifierTest('GaussianNB', GaussianNB(), 1), # Doenst work... too dense error,
                                                                    
    ]

    # Just to show header
    headerClassifier = ClassifierTest('Header', None, 0)
    print(headerClassifier.str_keys())

    # tests our cases
    for test_case in test_cases: #[:1]
        classifier = Classifier(test_case, language, sentences, verbose=False)

        predictions = []
        for test in tests_text: 
            prediction = classifier.predict(test)
            predictions.append(prediction[0])
            if (verbose):
                print ('# Prediction: {} | Text: {}'.format(prediction, test[:70].replace("\n", "")))
            
        mean = np.mean(np.array(predictions) == tests_language)
        
        print("{}{}{} | {}".format(bcolors.HEADER, test_case, bcolors.ENDC, mean))
    
if __name__ == '__main__':  
   main(verbose=False)