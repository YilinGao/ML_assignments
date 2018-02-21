import numpy as np
from cvxopt import matrix, solvers

"""
a hard SVM classifier
"""
class SVMClassifier:
    def __init__(self, kernel):
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
        assert X.shape[0] == y.shape[0]
        # number of observations, number of features
        n, p = X.shape

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
        sol=solvers.qp(P, q, G, h, A, b)
#         if sol['status'] == 'optimal':
#             self.infeasible = False
        self.multipliers = np.array(sol['x'])
        self.multipliers = self.multipliers.flatten() # turn it into 1D array
#         else:
#             print('The data points are not separable!')
#             return

        self.lambdas = np.zeros([1, p])
        # indices of support vectors
        sv_index = self.multipliers > 1e-5
        active_multipliers = self.multipliers[sv_index]
        X_sv = X[sv_index]
        y_sv = y[sv_index]
        for multiplier, Xi, yi in zip(active_multipliers, X_sv, y_sv):
            self.lambdas += multiplier * yi * Xi
        self.lambda_zero = 1 - np.dot(self.lambdas, X_sv[0])

    def predict(self, X):
        n, p = X.shape
        # y_hat = []
        # for obs in X:
        #     yhat.append(self.lambda_zero + np.dot(self.lambdas, obs)[0])
        # y_hat = np.array(y_hat)
        y_hat = np.zeros(n)
        y_hat = self.lambda_zero + np.dot(X, self.lambdas.T)
        y_hat[y_hat <= 0] = 0
        y_hat[y_hat > 0] = 1
        return y_hat.flatten()
