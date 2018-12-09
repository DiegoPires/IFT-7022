import abc
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

from classifier import Classifier
from utility import bcolors

class CountVectorizerDTO():
    def __init__(self, strip_accents=None, stop_words=None, lowercase=True, max_df=1.0, min_df=1, binary=False, ngram_range=(1,1)):
        self.strip_accents = strip_accents
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.max_df = max_df
        self.min_df = min_df
        self.binary = binary
        self.ngram_range = ngram_range

class KerasTokenizerDTO():
    def __init__(self, num_words=None, lower=True, split=' ', char_level=False, mode='binary'):
        self.num_words=num_words
        self.lower=lower
        self.split=split
        self.char_level=char_level
        self.mode=mode # "binary", "count", "tfidf", "freq"
        
class DataDTO():
    def __init__(self, data_train, data_test, target_train, target_test, target_names, vocab_size=15000):
        self.data_train = data_train 
        self.data_test = data_test 
        self.target_train = target_train
        self.target_test = target_test
        self.target_names = target_names
        self.vocab_size = vocab_size

class KerasClassifier(Classifier):
    def __init__(self, name, model, vectorize_data, verbose):
        self.name = name
        self.model = model
        self.vectorizer = vectorize_data.vectorizer
        self.labels = vectorize_data.data_dto.target_names
        self.labels.sort()
        self.loss, self.accuracy = self.get_loss_and_accuracy(model, vectorize_data, verbose)

        if (verbose):
            print("{} | {}{}{}".format( 
                self.name,
                bcolors.BOLD,
                self.accuracy,
                bcolors.ENDC))
        
    def get_loss_and_accuracy(self, model, vectorize_data, verbose):
        loss, accuracy = model.evaluate(vectorize_data.X_train, vectorize_data.y_train, verbose=verbose)
        if (verbose):
            print("Training Accuracy: {:.4f} ; Loss: {:.4f}".format(accuracy, loss))
        
        loss, accuracy = model.evaluate(vectorize_data.X_test, vectorize_data.y_test, verbose=verbose)
        if (verbose):
            print("Testing Accuracy:  {:.4f} ; Loss: {:.4f}".format(accuracy, loss))

        return loss, accuracy

# Class used to encapsulate each Keras test done with Vectorizer
class KerasClassifierWithVectorizer(KerasClassifier):
    def predict(self, text):
        test_text = np.array([text])
        vectorized = self.vectorizer.transform(test_text)
        y_classes =  self.model.predict_classes(vectorized)
        return self.labels[y_classes]

# Class used to encapsulate Keras test done with Tokenizer
class KerasClassifierWithTokenizer(KerasClassifier):
    def predict(self, text):
        test_text = np.array([text])
        vectorized = self.vectorizer.texts_to_matrix(test_text, mode=self.vectorizer.mode)
        y_classes =  self.model.predict_classes(vectorized)
        return self.labels[y_classes]

# Small DTO to facilitate passing parameters to methods. It vectorize our data to be able to use with Keras
class Vectorized():
    def __init__(self, data_dto):
        self.data_dto = data_dto

    def initialize_with_count_vectorizer(self, count_vectorizer_dto=CountVectorizerDTO()):
        self.vectorizer = CountVectorizer(
            strip_accents = count_vectorizer_dto.strip_accents,
            stop_words = count_vectorizer_dto.stop_words, 
            lowercase = count_vectorizer_dto.lowercase, 
            max_df = count_vectorizer_dto.max_df, 
            min_df = count_vectorizer_dto.min_df,
            binary = count_vectorizer_dto.binary,
            ngram_range = count_vectorizer_dto.ngram_range
        )
        self.vectorizer.fit(self.data_dto.data_train)

        self.X_train = self.vectorizer.transform(self.data_dto.data_train)
        self.X_test  = self.vectorizer.transform(self.data_dto.data_test)
        
        # Need to transforms the texts in number to be able to use with Keras
        labelencoder_y_1 = LabelEncoder()
        self.y_train = to_categorical(labelencoder_y_1.fit_transform(self.data_dto.target_train))
        self.y_test =  to_categorical(labelencoder_y_1.fit_transform(self.data_dto.target_test))

        self.input_dim = self.X_train.shape[1]  # Number of features

    def initialize_with_keras_tokenizer(self, keras_tokenizer_dto=KerasTokenizerDTO()):
        # define Tokenizer with Vocab Size
        self.vectorizer = Tokenizer(num_words=self.data_dto.vocab_size)
        self.vectorizer.fit_on_texts(self.data_dto.data_train)

        self.X_train = self.vectorizer.texts_to_matrix(self.data_dto.data_train, mode=keras_tokenizer_dto.mode)
        self.X_test = self.vectorizer.texts_to_matrix(self.data_dto.data_test, mode=keras_tokenizer_dto.mode)

        encoder = LabelBinarizer()
        encoder.fit(self.data_dto.target_train)
        self.y_train = encoder.transform(self.data_dto.target_train)
        self.y_test = encoder.transform(self.data_dto.target_test)

        self.vectorizer.mode = keras_tokenizer_dto.mode

