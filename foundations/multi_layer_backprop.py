import numpy as np
from typing import List
from numpy.typing import NDArray


class Solution:
    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, z)

    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:
        # Convert inputs to numpy arrays
        x = np.array(x, dtype=np.float64)
        W1 = np.array(W1, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64)
        W2 = np.array(W2, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)

        # Forward pass
        z1 = W1 @ x + b1
        a1 = self.relu(z1)

        preds = W2 @ a1 + b2

        loss = np.mean((preds - y_true) ** 2)

        # Backward pass
        dL_dz2 = 2 * (preds - y_true) / len(y_true)

        dW2 = np.outer(dL_dz2, a1)
        db2 = dL_dz2

        dL_da1 = W2.T @ dL_dz2
        dL_dz1 = dL_da1 * (z1 > 0)

        dW1 = np.outer(dL_dz1, x)
        db1 = dL_dz1

        return {
            "loss": float(np.round(loss, 4)),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }