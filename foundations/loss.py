import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss = 0
        for i in range(0, len(y_true)):
            loss += y_true[i]*np.log(y_pred[i] + 1e-7) + (1-y_true[i]) * np.log(1-y_pred[i])
        return round(-loss/len(y_true), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss = 0
        for i in range(0, len(y_true)):
            for c in range(0, y_true.shape[1]):
                loss += (1 if  y_true[i][c] == 1 else 0) * np.log(y_pred[i][c] + 1e-7)
        return round(-loss/len(y_true), 4)
