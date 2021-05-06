import numpy as np

from exceptions import SLAUException
from method.abstract import AbstractSLAU


class Gauss(AbstractSLAU):

    def __init__(self, matrix_a: np.ndarray, matrix_b: np.ndarray):
        self._check_correction(matrix_a, matrix_b)

        self._A = np.copy(matrix_a)
        self._B = np.copy(matrix_b)
        self.size = self._A.shape[0]
        self._X = None
        self._forwarded = False

    def solution(self):
        if self._X is None:
            self.forward()
            self.backward()
        return self._X

    def forward(self):

        for column in range(self.size):
            normalized = self._normalize_column(column)
            if not normalized:
                continue
            for row in range(column + 1, self.size):
                multiplier = self._A[row, column] / self._A[column, column]
                self._combine_row(row, column, multiplier)
                self._check_system_by_row(row)
        self._forwarded = True

        return self._A, self._B

    def backward(self) -> np.ndarray:
        if not self._forwarded:
            raise SLAUException(SLAUException.PREMATURELY_BACKWARD)
        self._X = np.empty((self.size,), dtype='float')
        for i in range(self.size - 1, -1, -1):
            self._calculate_x(i)
        return self._X

    def _normalize_column(self, column):
        if self._A[column, column] != 0:
            return 1

        for row in range(column + 1, self.size):
            if self._A[row, column] != 0:
                self._swap_rows(row, column)
                return 1
        return 0

    def _swap_rows(self, row1, row2):
        self._swap(self._A, row1, row2)
        self._swap(self._B, row1, row2)

    @classmethod
    def _swap(cls, matrix, row1, row2):
        matrix[[row1, row2]] = matrix[[row2, row1]]

    def _row_transformation(self, a):
        pass

    def _combine_row(self, row, original_row, multiplier):
        self._A[row] = self._A[row] - self._A[original_row] * multiplier
        self._B[row] = self._B[row] - self._B[original_row] * multiplier

    def _check_system_by_row(self, row):
        if np.max(np.abs(self._A[row])) == 0:
            if self._B[row] == 0:
                raise SLAUException(SLAUException.INDEFINITE)
            else:
                raise SLAUException(SLAUException.INCONSISTENT)

    def _calculate_x(self, row):
        coef = 0
        for i in range(self.size - 1, row, -1):
            coef += self._X[i] * self._A[row, i]
        coef = self._B[row] - coef

        if self._A[row, row] != 0:
            x = coef / self._A[row, row]
            self._X[row] = x
            return x

        else:
            raise SLAUException(SLAUException.INCORRECT)