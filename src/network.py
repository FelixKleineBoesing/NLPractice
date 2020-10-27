import numpy as np


class Word2Vec:

    def __init__(self, vocab_size: int, n_hidden_neurons: int = 20, epochs: int = 20, learning_rate: float = 0.1,
                 clipping_grad_value: float = 50):
        self.epochs = 20
        self.n_hidden_neurons = n_hidden_neurons
        self.vocab_size = vocab_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.clipping_grad_value = clipping_grad_value
        self._input_layer = None
        self._hidden_layer = None
        self._output_layer = None
        self._init_network()

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        stopped = False
        epoch = 1

        while not stopped:
            y_hat = self._forward_pass(X)
            self._backward_pass(y_hat, y, X)
            epoch += 1
            if epoch >= self.epochs:
                stopped = True

    def predict(self, X: np.ndarray):
        pass

    def _forward_pass(self, X: np.ndarray):
        x_1 = X @ self._input_layer
        x_2 = x_1 @ self._hidden_layer
        y_hat = softmax(x_2)
        return y_hat

    def _backward_pass(self, y_hat: np.ndarray, y: np.ndarray, X: np.ndarray):
        weight_updates = {}

        cost_gradient = cross_entropy_grad(y_hat=y_hat, y=y)
        hidden_gradient = cost_gradient @ self._hidden_layer.T
        input_gradient = hidden_gradient @ self._input_layer.T

    def _clip_gradients(self, arr):
        arr[arr > self.clipping_grad_value] = self.clipping_grad_value
        arr[arr < -self.clipping_grad_value] = -self.clipping_grad_value
        return arr

    def _init_network(self):
        self._input_layer = np.random.normal(size=(self.vocab_size, self.n_hidden_neurons))
        self._hidden_layer = np.random.normal(size=(self.vocab_size, self.n_hidden_neurons))
        self._output_layer = None


def softmax(X: np.ndarray):
    assert (len(X.shape) == 2), "x must be two dimensional"
    exp_x = np.exp(X)
    return exp_x / np.sum(exp_x)


def softmax_grad(X: np.ndarray):
    return softmax(X) * (1 - softmax(X))


def _reshape_activ_gradient(grad: np.ndarray):
    if len(np.array(grad).shape) == 0:
        grad = np.array([[grad], ])
    if len(grad.shape) == 1:
        grad = grad.reshape(grad.shape[0], 1)
    return grad


def cross_entropy(y_hat, y):
    return - np.sum(y * np.log(y_hat))


def cross_entropy_grad(y_hat, y):
    return y_hat - y

