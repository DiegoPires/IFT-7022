import os
import glob

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

from utility import get_complet_path
from pathlib import Path

from keras_classes import KerasClassifierWithVectorizer, KerasClassifierWithTokenizer, Vectorized

import numpy as np

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

# Create a simple keras classifier with one layer
# This example does not have a hidden layer, just input and output
def get_simple_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose=False):
    name = 'simple'

    vectorize_data = Vectorized(data_train, data_test, target_train, target_test, target_names)
    vectorize_data.initialize_with_count_vectorizer()

    model, path = get_saved_model(name)
    if (model == None):
        model = Sequential()
        model.add(Dense(10, input_dim=vectorize_data.input_dim, activation='relu'))
        model.add(Dense(len(vectorize_data.target_names), activation='softmax', kernel_initializer='uniform')) 

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

    return KerasClassifierWithVectorizer(name, model, vectorize_data, verbose)

def get_denser_keras_classifier(data_train, data_test, target_train, target_test, target_names, verbose=False):
    name = 'denser'

    vectorize_data = Vectorized(data_train, data_test, target_train, target_test, target_names)
    vectorize_data.initialize_with_count_vectorizer()

    model, path = get_saved_model(name)
    if (model == None):
        model = Sequential()
        model.add(Dense(512, input_dim=vectorize_data.input_dim, activation='relu'))
        model.add(Dropout(0.5)) # To avoid overfitting
        model.add(Dense(256, activation='sigmoid'))
        model.add(Dropout(0.5)) # To avoid overfitting
        model.add(Dense(len(vectorize_data.target_names), activation='softmax', kernel_initializer='uniform')) 

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

    return KerasClassifierWithVectorizer(name, model, vectorize_data, verbose)

def get_denser_keras_classifier_with_tokenizer(data_train, data_test, target_train, target_test, target_names, verbose=False):
    name = 'denser_and_tokenizer'

    vectorize_data = Vectorized(data_train, data_test, target_train, target_test, target_names, 15000)
    vectorize_data.initialize_with_keras_tokenizer()

    model, path = get_saved_model(name)
    if (model == None):
        model = Sequential()
        
        model.add(Dense(512, input_shape=(vectorize_data.vocab_size,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(vectorize_data.target_names)))
        model.add(Activation('softmax'))

        # categorial is the way to go for multiple possible categorys as results, instead of binary
        model.compile(loss='categorical_crossentropy',  
                    optimizer='adam', 
                    metrics=['accuracy'])
        
        if (verbose):
            model.summary()

        history = model.fit(vectorize_data.X_train, vectorize_data.y_train,
                        epochs=30,
                        batch_size=100,
                        verbose=verbose,
                        validation_split=0.1)
    
        model.save(path)

        if (verbose):
            print (history)

    return KerasClassifierWithTokenizer(name, model, vectorize_data, verbose)