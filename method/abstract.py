from abc import ABC, abstractmethod

import numpy as np


class AbstractSLAU(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def solution(self):
        pass

    def _check_correction(self, a: np.ndarray, b: np.ndarray):
        shape_a = a.shape
        shape_b = b.shape

        assert a.ndim == 2
        assert b.ndim == 1

        assert shape_a[0] == shape_b[0]
        assert shape_b[0] == shape_a[0]