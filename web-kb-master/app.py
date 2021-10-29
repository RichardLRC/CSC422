import os
import pandas as pd
import matplotlib.pyplot as plt

from models.pca_knn import PCA_KNN
from models.naive_bayes import NaiveBayes, MultinomialNaiveBayes
from models.maximum_entropy import MaximumEntropy


rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
classes = ['course', 'department', 'faculty', 'project', 'staff', 'student']

def run(model, train, test):
    Y_train = train['d_type']
    Y_test = test['d_type']
    X_train = train.iloc[:, 1:]
    X_test = test.iloc[:, 1:]

    print('Fitting...')
    model.fit(X_train, Y_train)
    print('Predicting...')
    predicted = model.predict(X_test)
    comparison = list(zip(Y_test, predicted))
    accuracy = sum(list(map(lambda x: 1 if x[0] == x[1] else 0, comparison))) / len(comparison)
    print(f"{model.name()} accuracy: {accuracy}")

    for c in classes:
        class_comparison = list(filter(lambda x: x[0] == c, comparison))
        accuracy = sum(list(map(lambda x: 1 if x[0] == x[1] else 0, class_comparison))) / len(class_comparison)
        print(f"{model.name()} {c} accuracy: {accuracy}")

    return predicted


def cross_validation(dataset):
    for train, test in dataset:
        print(f'Reading {train} ...')
        train_data = pd.read_csv(train)
        print(f'Reading {test} ...')
        test_data = pd.read_csv(test)

        print(f'Running PCA_KNN: {os.path.basename(test)} ...')
        run(PCA_KNN(), train_data, test_data)
        print(f'Running NaiveBayes: {os.path.basename(test)} ... ')
        run(NaiveBayes(), train_data, test_data)
        print(f'Running MultinomialNaiveBayes: {os.path.basename(test)} ...')
        run(MultinomialNaiveBayes(), train_data, test_data)
        print(f'Running MaximumEntropy: {os.path.basename(test)} ...')
        run(MaximumEntropy(), train_data, test_data)

        print('Destroying dataframe...')
        del train_data
        del test_data

def most_common(lst):
    return max(set(lst), key=lst.count)

def make_graph():
    plt.figure()
    models = ['Naive Bayes', 'Multinomial NB', 'Maximum Entropy', 'PCA-KNN', 'Ensemble']
    accuracies = [0.3633, 0.8729, 0.7810, 0.7009, 0.8156]
    plt.barh(models, accuracies)
    plt.show()


if __name__ == '__main__':

    # cross validation

    dataset = [['no_cornell.csv', 'cornell.csv'],
               ['no_texas.csv', 'texas.csv'],
               ['no_washington.csv', 'washington.csv'],
               ['no_wisconsin.csv', 'wisconsin.csv']]

    dataset = map(lambda x: [os.path.join(rootdir, 'data', 'schools_no_other', 'training', x[0]),
                             os.path.join(rootdir, 'data', 'schools_no_other', x[1])],
                  dataset)

    cross_validation(dataset)

    # run with best CV model

    test = os.path.join(rootdir, "data", "tokens_no_other.csv")
    print(f'Reading {test} ...')
    test_data = pd.read_csv(test)
    test_data = test_data.drop(columns=['d_school'])

    no_texas_train = os.path.join(rootdir, "data", "schools_no_other", "training", "no_texas.csv")
    print(f'Reading {no_texas_train} ...')
    no_texas_train_data = pd.read_csv(no_texas_train)

    no_washington_train = os.path.join(rootdir, "data", "schools_no_other", "training", "no_washington.csv")
    print(f'Reading {no_washington_train} ...')
    no_washington_train_data = pd.read_csv(no_washington_train)

    dataset = [[no_texas_train_data, NaiveBayes()],
               [no_texas_train_data, MultinomialNaiveBayes()],
               [no_texas_train_data, MaximumEntropy()],
               [no_washington_train_data, PCA_KNN()]]

    predicted = []

    for train_data, model in dataset:
        predicted.append(run(model, train_data, test_data))

    ensemble = list(zip(predicted[0], predicted[1], predicted[2], predicted[3]))
    ensemble_predicted = list(map(lambda x: most_common(x), ensemble))
    comparison = list(zip(test_data['d_type'], ensemble_predicted))
    accuracy = sum(list(map(lambda x: 1 if x[0] == x[1] else 0, comparison))) / len(comparison)
    print(f"ensemble accuracy: {accuracy}")

    make_graph()





