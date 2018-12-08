import os
import glob

from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical
from keras.models import load_model

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

from utility import get_complet_path
from pathlib import Path

from classifier import Classifier

import numpy as np

# Class used to encapsulate each Keras test, so it can be testable easily later
class KerasClassifier(Classifier):
    loss = 0
    accurary = 0
    model = None
    name = ''
    vectorizer = None
    labels = []
    def __init__(self, name, model, vectorizer, labels):
        self.name = name
        self.model = model
        self.vectorizer = vectorizer
        self.labels = labels
        self.labels.sort()

    def predict(self, text):
        test_text = np.array([text])
        vectorized = self.vectorizer.transform(test_text)
        y_classes =  self.model.predict_classes(vectorized)

        return self.labels[y_classes]

    def __str__(self):
        return self.name

# Small DTO to facilitate passing parameters to methods. It vectorize our data to be able to use with Keras
class Vectorized():
    def __init__(self, data_train, data_test, target_train, target_test, target_names):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(data_train)

        self.X_train = self.vectorizer.transform(data_train)
        self.X_test  = self.vectorizer.transform(data_test)
        
        # Need to transforms the texts in number to be able to use with Keras
        labelencoder_y_1 = LabelEncoder()
        self.y_train = to_categorical(labelencoder_y_1.fit_transform(target_train))
        self.y_test =  to_categorical(labelencoder_y_1.fit_transform(target_test))

        self.input_dim = self.X_train.shape[1]  # Number of features

        self.target_names = target_names
        
# Remove all saved models from the directory
def remove_saved_keras_models(remove_models):
    if (remove_models):
        files = glob.glob(get_complet_path("keras_models") + '/*')
        for f in files:
            os.remove(f)

# Get saved file from model if exists
def get_saved_model(name):
    model = None
    path = get_complet_path('keras_models') + "/" + name + ".h5"

    # We save/load model to improve performance
    if (Path(path).is_file()):
        model = load_model(path)

    return model, path

# return a KerasClassifier from a trained model
def get_keras_classifier(name, model, vectorize_data, verbose):

    classifier = KerasClassifier(name, model, vectorize_data.vectorizer, vectorize_data.target_names)
    
    loss, accuracy = model.evaluate(vectorize_data.X_train, vectorize_data.y_train, verbose=verbose)
    if (verbose):
        print("Training Accuracy: {:.4f} ; Loss: {:.4f}".format(accuracy, loss))
    
    classifier.loss, classifier.accuracy = model.evaluate(vectorize_data.X_test, vectorize_data.y_test, verbose=verbose)
    if (verbose):
        print("Testing Accuracy:  {:.4f} ; Loss: {:.4f}".format(classifier.accuracy, classifier.loss))

    return classifier

# Create a simple keras classifier with one layer
# This example does not have a hidden layer, just input and output
def get_simple_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose=False):
    name = 'simple'

    vectorize_data = Vectorized(data_train, data_test, target_train, target_test, target_names)

    model, path = get_saved_model(name)
    if (model == None):
        model = Sequential()
        model.add(layers.Dense(10, input_dim=vectorize_data.input_dim, activation='relu'))
        model.add(layers.Dense(len(vectorize_data.target_names), activation='softmax', kernel_initializer='uniform')) 

        # categorial is the way to go for multiple possible categorys as results, instead of binary
        model.compile(loss='categorical_crossentropy',  
                    optimizer='adam', 
                    metrics=['accuracy'])
        
        if (verbose):
            model.summary()

        history = model.fit(vectorize_data.X_train, vectorize_data.y_train,
                        epochs=100,
                        verbose=verbose,
                        validation_data=(vectorize_data.X_test, vectorize_data.y_test),
                        batch_size=100)
    
        model.save(path)

        if (verbose):
            print (history)

    return get_keras_classifier(name, model, vectorize_data, verbose)
