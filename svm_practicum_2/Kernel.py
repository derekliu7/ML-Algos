import numpy.linalg as la

class Kernel(object):
    """Implements a list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    
    Each kernel method should return a function of two arguments.
    This is known as a "closure".
    """
    
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        pass

    @staticmethod
    def _polykernel(dimension, offset):
        ''' Should be used in inhomogenous_polynomial
            and homogenous_polynomial.
            
            offset = 0 for homogenous case
                   = 1 for inhomogenous case  
        '''
        pass

    @staticmethod
    def inhomogenous_polynomial(dimension):
        pass

    @staticmethod
    def homogenous_polynomial(dimension):
        pass

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        pass
