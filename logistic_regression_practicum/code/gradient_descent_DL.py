import numpy as np

__author__ = "you"


class Grad(object):
    def __init__(self, fit_intercept=True, normalize=False, gradient=None, mu=None, sigma=None):
        '''
        INPUT: GradientDescent, boolean
        OUTPUT: None
        Initialize class variables. Gradient is the function used to compute the gradient.
        '''
        self.coeffs = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.mu = mu
        self.sigma = sigma
        self.alpha = None
        self.gradient = gradient
        self.transformed_X = None

    def run(self, X, y, coeffs=None, alpha=0.01, num_iterations=10000):
        '''
        INPUT: X: 2 dimensional numpy array of features data, y: n x 1 dimensional
        array of labels
            OUTPUT: None, updates the coefficients
        Updates coeffs num_iterations numbers of times, using gradient to calculate the gradient at each step.

        '''

        self.calculate_normalization_factors(X)
        X = self.maybe_modify_matrix(X)
        self.transformed_X = X

        self.alpha = alpha
        self.coeffs = coeffs

        if self.coeffs is None:
            self.coeffs = np.zeros(X.shape[1])

        for i in range(num_iterations):
            self.coeffs -= self.alpha * \
                np.array(self.gradient(X, y, self.coeffs))
        return self.coeffs

    def calculate_normalization_factors(self, X):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array
        OUTPUT: None
        Initialize mu and sigma instance variables to be the numpy arrays
        containing the mean and standard deviation for each column of X.
        '''
        self.mu = np.average(X, 0)
        self.sigma = np.std(X, 0)
        # Don't normalize intercept column
        self.mu[self.sigma == 0] = 0
        self.sigma[self.sigma == 0] = 1

    def add_intercept(self, X):
        '''
        INPUT: 2 dimensional numpy array
        OUTPUT: 2 dimensional numpy array
        Return a new 2d array with a column of ones added as the first
        column of X.
        '''
        return np.column_stack((np.ones(len(X)), X))

    def maybe_modify_matrix(self, X):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array
        OUTPUT: 2 dimensional numpy array
        Depending on the settings, normalizes X and adds a feature for the
        intercept.
        '''
        if self.normalize:
            X = (X - self.mu) / self.sigma
        if self.fit_intercept:
            return self.add_intercept(X)
        return X
