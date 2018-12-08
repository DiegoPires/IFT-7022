import abc

# Little class used to help evaluation of SkLearn and Keras classifiers
class Classifier(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(self, text):
        raise NotImplementedError('users must define __str__ to use this base class')