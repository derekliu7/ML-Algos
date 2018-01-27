from collections import Counter, defaultdict
import numpy as np


class NaiveBayes(object):

    def __init__(self, alpha=1):
        '''
        INPUT:
        - alpha: float, laplace smoothing constant
        '''

        self.class_totals = None
        self.class_feature_totals = None
        self.class_counts = None
        self.alpha = alpha

    def _compute_likelihood(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        Compute the totals for each class and the totals for each feature
        and class.
        '''

        class0 = []
        class1 = []
        list0 = []
        list1 = []
        for i in range(len(y)):
            if y[i] == 1:
                list1.append(X[i])
            if y[i] == 0:
                list0.append(X[i])
        self.class_totals = {0: np.sum(list0), 1: np.sum(list1)}

        array0 = np.asarray(list0)
        array1 = np.asarray(list1)

        for k in range(len(array0[1])):
            class0.append(sum(array0[:, k]))
            class1.append(sum(array1[:, k]))

        self.class_feature_totals = {0: class0, 1: class1}

    def fit(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT: None
        '''

        # This section is given to you.

        # compute priors
        self.class_counts = Counter(y)

        # compute likelihoods
        self._compute_likelihood(X, y)

    def predict(self, X):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix

        OUTPUT:
        - predictions: numpy array
        '''
        ham = self.class_counts[0]
        spam = self.class_counts[1]
        prob_feature0 = []
        prob_feature1 = []
        for i in range(X.shape[1]):
            prob_feature0.append(
                np.log((self.class_feature_totals[0][i] + 1)) - np.log(self.class_totals[0] + X.shape[1]))
            prob_feature1.append(
                np.log((self.class_feature_totals[1][i] + 1)) - np.log(self.class_totals[1] + X.shape[1]))
        A0 = X.dot(np.asarray(prob_feature0))
        for i in range(len(A0)):
            A0[i] = A0[i] + np.log(ham / (ham + spam))
        A1 = X.dot(np.asarray(prob_feature1))
        for j in range(len(A1)):
            A1[j] = A1[j] + np.log(spam / (ham + spam))
        final = []
        for i in range(len(A0)):
            if A0[i] > A1[i]:
                final.append(0)
            elif A0[i] < A1[i]:
                final.append(1)
            else:
                final.append(1)
        return np.asarray(final)

    def score(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - accuracy: float between 0 and 1

        Calculate the accuracy, the percent of documents predicted correctly.
        '''

        return sum(self.predict(X) == y) / float(len(y))
