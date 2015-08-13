# -*- coding: UTF-8 -*-
__author__ = 'Draonfly'
"""Support Vector Classifier"""
import numpy as np

class SVC:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 probability=False,tol=1e-3,max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.probability = probability
        self.tol = tol
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0

    def _kernel_trans(self, X, x_i):
        m, n = X.shape()
        K = np.mat(np.zeros((m, 1)))
        if self.kernel == 'linear':
            K = X * x_i.T
        elif self.kernel == 'rbf'
            for j in range(m):
                delta_row = X[j, :] - x_i
                K[j] = delta_row * delta_row.T
            K = np.exp(-self.gamma * K)
        else:
            raise NameError('%s is not recognized'% self.kernel)
        return K

    def _smo(self, X, y):
        self.K =
        iter = 0


    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_prob(self, X):
        pass