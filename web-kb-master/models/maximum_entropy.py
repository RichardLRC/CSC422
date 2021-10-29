import os
import nltk

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

class MaximumEntropy:
    def __init__(self):
        self.classifier = None

    def list_to_dict(self, words_list):
        return dict([(word, True) for word in words_list])

    def fit(self, X_train, Y_train):
        training_set = []

        for index, row in X_train.iterrows():
            training_set.append((row[row != 0].index.tolist(), Y_train[index]))

        training_set_formatted = [(self.list_to_dict(element[0]), element[1]) for element in training_set]

        numIterations = 20
        algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
        self.classifier = nltk.MaxentClassifier.train(training_set_formatted, algorithm, max_iter=numIterations)
        self.classifier.show_most_informative_features(100)

    def predict(self, X_test):
        test_set = []
        output = []

        for index, row in X_test.iterrows():
            test_set.append(row[row != 0].index.tolist())
        
        test_set_formatted = [(self.list_to_dict(element)) for element in test_set]

        for review in test_set_formatted:
            determined_label = self.classifier.classify(review)
            output.append(determined_label)

        return output

    def name(self):
        return 'Maximum Entropy'
