from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical
from keras.models import model_from_json

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

from utility import get_complet_path
from pathlib import Path

def SimpleKerasClassifier(data_train, data_test, target_train, target_test, target_names, verbose=False):
    classifier_name = 'simple'

    vectorizer = CountVectorizer()
    vectorizer.fit(data_train)
    X_train = vectorizer.transform(data_train)
    X_test  = vectorizer.transform(data_test)
    
    # The only way to transform the output from text into number, the only way to train our model
    labelencoder_y_1 = LabelEncoder()
    y_train = to_categorical(labelencoder_y_1.fit_transform(target_train))
    y_test =  to_categorical(labelencoder_y_1.fit_transform(target_test))

    input_dim = X_train.shape[1]  # Number of features

    path_to_model = Path(get_complet_path('keras_models') + "/" + classifier_name +".json")
    if (not path_to_model.is_file()):
        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(len(target_names), activation='softmax', kernel_initializer='uniform')) 

        # categorial is the way to go for multiple possible categorys as results, instead of binary
        model.compile(loss='categorical_crossentropy',  
                    optimizer='adam', 
                    metrics=['accuracy'])
        
        save_model(classifier_name, model, verbose)
    else:
        model = load_model(classifier_name, verbose)

    if (verbose):
        model.summary()

    history = model.fit(X_train, y_train,
                     epochs=100,
                     verbose=False,
                     #validation_data=(X_test, y_test),
                     batch_size=100)

    if (verbose):
        print ("history" + history)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    
    if (verbose):
        print("Training Accuracy: {:.4f}".format(accuracy))
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    
    if (verbose):
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    return (model, accuracy)

def save_model(name, model, verbose):
    path = get_complet_path('keras_models')

    # serialize model to JSON
    model_json = model.to_json()
    with open(path +"/" + name + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(path + "/" + name + ".h5")

    if (verbose):
        print("Saved model '" + name + "' to disk")

def load_model(name, verbose):
    path = get_complet_path('keras_models') 

    json_file = open(path + "/" + name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + "/" + name + ".h5")
    
    if (verbose):
        print("Loaded model '" + name + "' from disk")

    return loaded_model
