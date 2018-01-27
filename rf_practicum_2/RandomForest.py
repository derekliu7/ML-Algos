from DL_ImprovedDecisionTree import ImprovedDecisionTree
import numpy as np
from collections import Counter


class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0],
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):

        # * Return a list of num_trees DecisionTrees.
        list_samples = []
        whole = np.column_stack((X, y))
        for i in range(num_trees):
            list_samples.append(whole[np.random.choice(
                whole.shape[0], num_samples, replace=True)])
        list_trees = []
        for sample in list_samples:
            tree = ImprovedDecisionTree()
            tree.fit(sample[:, :-1], sample[:, -1])
            list_trees.append(tree)
        return list_trees

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''

        # * Each one of the trees is allowed to predict on the same row of input data. The majority vote
        # is the output of the whole forest. This becomes a single prediction.

        list_row_pred = []
        for row in len(X):
            list_pred = []
            for tree in self.forest:
                list_pred.append(tree.predict(X[row, :]))
            lab = Counter(list_pred)
            list_row_pred.append(lab.most_common(1)[0][0])
        return np.asarray(list_row_pred)

    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data.
        '''
        y_ = self.predict(X)

        return sum(y_ == y) / len(y)
