import numpy as np
from sklearn import metrics
from cvxopt import matrix, solvers

class SVMClassifierError(Exception):
    """Base calss for exception in SVMClassifier module
    
    Attributes:
        message: the error message
    """
    def __init__(self, message):
        self.message = message

class DataNonSeparableError(SVMClassifierError):
    """Exception for non separable training data, raised only in a hard SVM classifier train() method"""
    pass

class ClassifierUndefinedError(SVMClassifierError):
    """Exception for predicting labels when the classifier parameters are not defined,
    raised only in a hard SVM classifier predict() method
    """
    pass
    
class SVMClassifier:
    """A hard SVM classifier using user defined kernel functions"""
    def __init__(self, kernel):
        """__init__ for SVMClassifier class
        
        Attributes:
            kernel(x1, x2): the kernel function that take 2 x observations
        """
        # if the hard constrained SVM is feasible
        self.infeasible = True
        # primal variables
        self.lambdas = None
        self.lambda_zero = None
        # dual variables
        self.multipliers = None
        # kernel function
        self.kernel = kernel

    def train(self, X, y):
        """Method to train the SVM classifier
        Takes features (X) and labels(y), uses a quadratic solver (cvxopt) to solve the dual primal problem,
        uses the optimal dual variables to compute primal variables.
        If the data is not separable, raise an exception, set self.infeasible to False, and leave optimal variables as None
        
        Attributes:
            X: numpy.ndarray, shape = [n, p]
            y: numpy.ndarray, shape = [n, 1]
            
        Exceptions:
            DataNonSeparableError: raised when the training data is not separable
        """
        assert X.shape[0] == y.shape[0]
        # number of observations, number of features
        n, p = X.shape

        # define parameters used by the quadratic solver
        # reference: http://cvxopt.org/userguide/coneprog.html#cvxopt.solvers.qp
        # reference: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
        gram = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                gram[i, j] = self.kernel(X[i, :], X[j, :])
        
        P = matrix(np.outer(y, y) * gram)
        q = matrix(np.ones(n) * -1)
        G = matrix(np.diag(np.ones(n) * -1))
        h = matrix(np.zeros(n))
        A = matrix(y, (1, n))
        b = matrix(0.00)
        solvers.options['show_progress'] = False # make the solver work silently
        sol=solvers.qp(P, q, G, h, A, b)
        
        # the quadratic solver cannot find optimal solution to the problem
        # so the training data is not separable
        if sol['status'] != 'optimal':
            raise DataNonSeparableError('The training data is not separable. The HARD SVM classifier has failed!')
        
        self.infeasible = False
        self.multipliers = np.array(sol['x'])
        self.multipliers = self.multipliers.flatten() # turn it into 1D array

        self.lambdas = np.zeros([1, p])
        # indices of support vectors, with active constraints
        sv_index = self.multipliers > 1e-5
        active_multipliers = self.multipliers[sv_index]
        X_sv = X[sv_index]
        y_sv = y[sv_index]
        # compute primal variable lambdas
        for multiplier, Xi, yi in zip(active_multipliers, X_sv, y_sv):
            self.lambdas += multiplier * yi * Xi
        # compute primal variable lambda_zero
        self.lambda_zero = 1 - np.dot(self.lambdas, X_sv[0])
        
    def predict(self, X):
        """Method to predict labels based on classifier parameters
        Takes features (X), uses classifier primal variables to work out f(x), 
        and uses sign(f(x)) to predict labels
        
        Attributes:
            X: numpy.ndarray, shape = [n, p]
        
        Return values:
            yhat: numpy.ndarray, shape = [n, 1], each element can take 2 possible values {0, 1}
        
        Exceptions:
            ClassifierUndefinedError: raised when the classifier status is infeasible
        """
        # the training data is not separable so the classifier parameters are not defined
        # cannot do the prediction
        if self.infeasible:
            raise ClassifierUndefinedError('The SVM classifier is not defined. Please train it with separable data!')
        
        n, p = X.shape
        y_hat = np.zeros(n)
        y_hat = self.lambda_zero + np.dot(X, self.lambdas.T)
        y_hat[y_hat <= 0] = 0
        y_hat[y_hat > 0] = 1
        return y_hat.flatten()
