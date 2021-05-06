import numpy as np

from exceptions import SLAUException
from method.abstract import AbstractSLAU


class Sweep(AbstractSLAU):

    def __init__(self, matrix_a, matrix_b):
        self._check_correction(matrix_a, matrix_b)
        self.size = matrix_a.shape[0]

        self.a_matrix = np.zeros((self.size,))
        self.b_matrix = np.zeros((self.size,))
        self.c_matrix = np.zeros((self.size,))
        self.d_matrix = np.copy(matrix_b)

        self.alpha_matrix = np.zeros((self.size,))
        self.beta_matrix = np.zeros((self.size,))
        self.y_matrix = np.zeros((self.size,))

        self.X = None

        self._forwarded = False

        for i in range(self.size):
            if i > 0:
                self.a_matrix[i] = matrix_a[i, i - 1]
            if i < self.size - 1:
                self.c_matrix[i] = matrix_a[i, i + 1]
            self.b_matrix[i] = matrix_a[i, i]

    def solution(self):
        if self.X is None:
            self.forward()
            self.backward()
        return self.X

    def forward(self):
        for i in range(self.size):
            self.y_matrix[i] = self.b_matrix[i] + self.a_matrix[i] * self.alpha_matrix[i - 1]
            self.alpha_matrix[i] = - self.c_matrix[i] / self.y_matrix[i]
            self.beta_matrix[i] = (self.d_matrix[i] - self.a_matrix[i] * self.beta_matrix[i - 1]) / self.y_matrix[i]
        self._forwarded = True

    def backward(self):
        if not self._forwarded:
            raise SLAUException(SLAUException.PREMATURELY_BACKWARD)
        self.X = np.zeros((self.size,))
        self.X[self.size - 1] = self.beta_matrix[self.size - 1]
        for i in range(self.size - 2, -1, -1):
            self.X[i] = self.alpha_matrix[i] * self.X[i + 1] + self.beta_matrix[i]
