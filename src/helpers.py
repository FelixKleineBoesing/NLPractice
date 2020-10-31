import numpy as np


def softmax(X: np.ndarray):
    assert (len(X.shape) == 2), "x must be two dimensional"
    exp_x = np.exp(X - np.max(X))
    return exp_x / np.sum(exp_x, axis=1).reshape(X.shape[0], 1)


def softmax_grad(X: np.ndarray):
    return softmax(X) * (1 - softmax(X))


def _reshape_activ_gradient(grad: np.ndarray):
    if len(np.array(grad).shape) == 0:
        grad = np.array([[grad], ])
    if len(grad.shape) == 1:
        grad = grad.reshape(grad.shape[0], 1)
    return grad


def cross_entropy(y_hat, y):
    if len(y.shape) == 3:
        return - np.sum([y[:, :, i] * np.log(y_hat) for i in range(y.shape[2])])
    else:
        return - np.sum(y * np.log(y_hat))


def cross_entropy_grad(y_hat, y):
    if len(y.shape) == 3:
        return np.sum([y_hat[:, :, 0] - y[:, :, i] for i in range(y.shape[2])], axis=0)
    else:
        return y_hat - y


class Layer:

    def __init__(self, weights: np.ndarray):
        self.weights = weights
        self.activation = None

    def set_activation(self, activation: np.ndarray):
        self.activation = activation