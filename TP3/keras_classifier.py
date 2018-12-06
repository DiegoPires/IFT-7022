from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

def SimpleKerasClassifier(data_train, data_test, target_train, target_test, target_names, verbose=False):
    
    vectorizer = CountVectorizer()
    vectorizer.fit(data_train)
    X_train = vectorizer.transform(data_train)
    X_test  = vectorizer.transform(data_test)
    
    labelencoder_y_1 = LabelEncoder()
    y = labelencoder_y_1.fit_transform(target_train)

    input_dim = X_train.shape[1]  # Number of features

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(len(target_names), activation='softmax', init='uniform')) 
    model.compile(loss='categorical_crossentropy',  
                optimizer='adam', 
                metrics=['accuracy'])
    
    if (verbose):
        model.summary()

    history = model.fit(X_train, to_categorical(y),
                     epochs=100,
                     verbose=False,
                     # validation_data=(X_test, target_test),
                     batch_size=10)

    loss, accuracy = model.evaluate(X_train, target_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, target_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    return model