import numpy as np
from gradient_descent_DL import Grad

__author__ = "Jared Thompson"


class LogisticRegression(object):

    def __init__(self, fit_intercept=True, scale=True, norm=None):
        '''
        INPUT: GradientDescent, function, function, function
        OUTPUT: None

        Initialize class variables. Takes three functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        predict: function to calculate the predicted values (0 or 1) for
        the given data
        '''

        gradient_choices = {None: self.cost_gradient,
                            "L1": self.cost_gradient_lasso,
                            "L2": self.cost_gradient_ridge}

        self.alpha = None
        self.gamma = None
        self.coeffs = None
        self.num_iterations = 0
        self.fit_intercept = fit_intercept
        self.scale = scale
        self.normalize = False
        if norm:
            self.norm = norm
            self.normalize = True

        self.gradient = gradient_choices[norm]

    def fit(self,  X, y, alpha=0.01, num_iterations=10000, gamma=0.):
        '''
        INPUT: 2 dimensional numpy array, numpy array, float, int, float
        OUTPUT: numpy array

        Main routine to train the model coefficients to the data
        the given coefficients.
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.num_iterations = num_iterations

        grad = Grad(gradient=self.gradient, normalize=True)
        new_coeffs = grad.run(
            X, y, coeffs=self.coeffs, alpha=self.alpha, num_iterations=self.num_iterations)

        self.coeffs = new_coeffs
        return grad.transformed_X

    def predict(self, X):
        '''
        INPUT: 2 dimensional numpy array, numpy array
        OUTPUT: numpy array

        Calculate the predicted values (0 or 1) for the given data with
        the given coefficients.
        '''

        return np.around(self.hypothesis(X, self.coeffs))

    def hypothesis(self, X, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array
        OUTPUT: numpy array of floats

        Calculate the predicted percentages (floats between 0 and 1)
        for the given data with the given coefficients.
        '''

        return 1. / (1. + np.exp(-1. * (X.dot(coeffs))))

    def cost_function(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float (a scalar)

        Calculate the value of the cost function for the data with the
        given coefficients.
        '''

        hype = self.hypothesis(X, coeffs)
        m = y.shape[0]
        return (1. / m) * ((y.T).dot(hype)) + ((np.ones(shape=(y.shape)) - y).T).dot(np.ones(shape=(hype.shape)) - hype)

    def cost_lasso(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float

        Calculate the value of the cost function with lasso regularization
        for the data with the given coefficients.
        '''

        m = y.shape[0]
        return self.cost_function(X, y, coeffs) + self.gamma * (np.sum(np.absolute(coeffs))) / (2 * m)

    def cost_ridge(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float

        Calculate the value of the cost function with ridge regularization
        for the data with the given coefficients.
        '''

        m = y.shape[0]
        return self.cost_function(X, y, coeffs) + self.gamma * (np.sum(coeffs**2)) / (2 * m)

    def cost_gradient(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function at the given value
        for the coeffs.

        Return an array of the same size as the coeffs array.
        '''
        hype = self.hypothesis(X, coeffs)
        return X.T.dot(hype - y) / len(y)

    def cost_gradient_lasso(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function with regularization
        at the given value for the coeffs.

        Return an array of the same size as the coeffs array.
        '''

        m = y.shape[0]
        return self.cost_gradient(X, y, coeffs) + self.gamma * (np.sum(np.absolute(coeffs))) / (2. * m)

    def cost_gradient_ridge(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function with regularization
        at the given value for the coeffs.

        Return an array of the same size as the coeffs array.
        '''

        m = y.shape[0]
        return self.cost_gradient(X, y, coeffs) + self.gamma * (np.sum(coeffs**2.)) / (2. * m)
