import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        print("z ", z.shape)
        exponentiated = np.exp(z - np.max(z, axis=1, keepdims=True))
        print("exp ", exponentiated.shape)
        denominator = np.sum(exponentiated, axis=1, keepdims=True)
        print("den ", denominator)
        return np.round(exponentiated/denominator, 4)[0]
