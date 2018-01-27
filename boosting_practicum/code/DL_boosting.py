import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50

    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        Build the estimators for the AdaBoost estimator.
        '''

        # Initialize weights to 1 / n_samples
        self.estimator_weight_ = np.full(y.shape, 1 / len(y))
        # For each of n_estimators, boost
        for i in range(self.n_estimator):
            self.estimators_.append(self._boost(x, y, self.estimator_weight_))
            # Append estimator, sample_weights and error to lists

    def _boost(self, x, y, sample_weight):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array

        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)

        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        '''

        estimator = clone(self.base_estimator)

        # Fit according to sample weights, emphasizing certain data points
        estimator.fit(x, y, sample_weight)
        y_hat = estimator.predict(x)
        incorrect = y != y_hat
        total_error = (incorrect.dot(sample_weight.T)) / np.sum(sample_weight)
        alpha = np.log((1 - total_error) / total_error)
        index_ = np.where(incorrect == 1)
        for i in index_:
            sample_weight[i] = sample_weight[i] * np.exp(alpha)
        self.estimator_weight_ = sample_weight
        return estimator, alpha

    def predict(self, x):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix

        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''

        # get predictions from tree family
        listofresults = []
        for i in self.estimators_:
            result = i[0].predict(x)
            index_ = np.where(result == 0)
            for j in index_:
                result[j] = -1
            listofresults.append(result * i[1])
        hat = np.sum(listofresults, axis=0)
        return hat > 0

        # set negative predictions to -1 instead of 0 (so we have -1 vs. 1)

    def score(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''
        y_ = self.predict(x)

        return sum(y_ == y) / len(y)
