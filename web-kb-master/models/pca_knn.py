import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

class PCA_KNN:
    def __init__(self):
        self.matrix_w = []
        self.knn = None
        self.scaler = StandardScaler()

    def fit(self, X_train, Y_train):
        train_standardized = self.scaler.fit(X_train).transform(X_train)
        train_cov = np.cov(train_standardized.T)

        eigenvalues, eigenvectors = np.linalg.eigh(train_cov)
        sorted_eigenvalues = sorted(eigenvalues, reverse=True)
        fig = plt.figure()
        dimensions = tuple(range(1, len(X_train.columns)+1, 1))
        print(len(dimensions))
        print(len(sorted_eigenvalues))
        fig.add_subplot()
        plt.bar(dimensions, sorted_eigenvalues)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        # plt.show()

        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]

        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        self.matrix_w = np.matrix(list(map(lambda x: x[1], eig_pairs))).T
        self.train_pc = np.dot(train_standardized, self.matrix_w)

        self.knn = KNeighborsClassifier(n_neighbors=50, p=2)
        self.knn.fit(self.train_pc[:, :500], Y_train)

    def predict(self, X_test):
        test_standardized = self.scaler.transform(X_test)
        test_pc = np.dot(test_standardized, self.matrix_w)

        return self.knn.predict(test_pc[:, :500])

    def name(self):
        return 'PCA+KNN'

