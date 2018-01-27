import numpy as np
from gradient_descent_solution import GradientDescent

__author__ = "Jared Thompson"

class LogisticRegression(object):

    def __init__(self, fit_intercept = True, scale = True, norm = "L2"):
        '''
        INPUT: GradientDescent, function, function, function
        OUTPUT: None

        Initialize class variables. Takes three functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        predict: function to calculate the predicted values (0 or 1) for
        the given data
        '''
        gradient_choices = {None: self.cost_gradient, "L1": self.cost_gradient_lasso, "L2": self.cost_gradient_ridge}

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


    def predict(self, X):
        '''
        INPUT: 2 dimensional numpy array, numpy array
        OUTPUT: numpy array

        Calculate the predicted values (0 or 1) for the given data with
        the given coefficients.
        '''


    def hypothesis(self, X, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array
        OUTPUT: numpy array

        Calculate the predicted percentages (floats between 0 and 1)
        for the given data with the given coefficients.
        '''

    def cost_function(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float

        Calculate the value of the cost function for the data with the
        given coefficients.
        '''


    def cost_lasso(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float

        Calculate the value of the cost function with lasso regularization
        for the data with the given coefficients.
        '''


    def cost_ridge(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float

        Calculate the value of the cost function with ridge regularization
        for the data with the given coefficients.
        '''


    def cost_gradient(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function at the given value
        for the coeffs.

        Return an array of the same size as the coeffs array.
        '''

    def cost_gradient_lasso(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function with regularization
        at the given value for the coeffs.

        Return an array of the same size as the coeffs array.
        '''

    def cost_gradient_ridge(self, X, y, coeffs):
        '''
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function with regularization
        at the given value for the coeffs.

        Return an array of the same size as the coeffs array.
        '''