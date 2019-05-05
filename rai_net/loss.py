"""
A loss function measures how good the predictions are. 
Use this to adjust parameters of the network.
"""
import numpy as np
from rai_net.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    """
    Computing sum of squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

class MAE(Loss):
    """
    Computing sum of absolute error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum(np.abs(predicted - actual))

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        grad = np.ones_like(predicted)
        grad[actual > predicted] = -1
        return grad
