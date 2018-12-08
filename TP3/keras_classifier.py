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

class KerasClassifier(Classifier):
    loss = 0
    accurary = 0
    model = None
    name = ''
    vectorizer = None
    labels = []
    def __init__(self, name, vectorizer, labels):
        self.name = name
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

def RemoveSavedKerasModels(remove_models):
    if (remove_models):
        files = glob.glob(get_complet_path("keras_models") + '/*')
        for f in files:
            os.remove(f)

def GetSimpleKerasClassifier(data_train, data_test, target_train, target_test, target_names, verbose=False):
    vectorizer = CountVectorizer()
    vectorizer.fit(data_train)
    
    classifier = KerasClassifier('simple', vectorizer, target_names)
    path = get_complet_path('keras_models') + "/" + classifier.name + ".h5"
    
    X_train = vectorizer.transform(data_train)
    X_test  = vectorizer.transform(data_test)
    
    # The only way to transform the output from text into number, the only way to train our model
    labelencoder_y_1 = LabelEncoder()
    y_train = to_categorical(labelencoder_y_1.fit_transform(target_train))
    y_test =  to_categorical(labelencoder_y_1.fit_transform(target_test))

    input_dim = X_train.shape[1]  # Number of features

    # We save/load model to improve performance
    if (not Path(path).is_file()):
        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(len(target_names), activation='softmax', kernel_initializer='uniform')) 

        # categorial is the way to go for multiple possible categorys as results, instead of binary
        model.compile(loss='categorical_crossentropy',  
                    optimizer='adam', 
                    metrics=['accuracy'])
        
        if (verbose):
            model.summary()

        history = model.fit(X_train, y_train,
                        epochs=100,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=100)
    
        model.save(path)

        if (verbose):
            print (history)

    else:
        model = load_model(path)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    
    if (verbose):
        print("Training Accuracy: {:.4f} ; Loss: {:.4f}".format(accuracy, loss))
    
    classifier.loss, classifier.accuracy = model.evaluate(X_test, y_test, verbose=False)
     
    if (verbose):
        print("Testing Accuracy:  {:.4f} ; Loss: {:.4f}".format(classifier.accuracy, classifier.loss))

    classifier.model = model

    return classifier