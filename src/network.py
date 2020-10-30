import numpy as np
import pandas as pd
from typing import Union
from sortedcontainers import SortedList

from src.data_constructors import build_word_indices, build_cbow_dataset


class Word2Vec:

    def __init__(self, n_hidden_neurons: int = 20, epochs: int = 20, learning_rate: float = 0.1,
                 clipping_grad_value: float = 50, batch_size: int = 1):
        self.epochs = 20
        self.n_hidden_neurons = n_hidden_neurons
        self.vocab_size = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.clipping_grad_value = clipping_grad_value
        self.batch_size = batch_size
        self.word_index = None
        self.index_word = None
        self._hidden_layer: Layer = None
        self._output_layer: Layer = None
        self._tracking = []

    def train(self, corpus: Union[list, str], window_size: int = 1):
        stopped = False
        epoch = 1
        vocab_size, word_index, index_word = build_word_indices(corpus)
        self.vocab_size = vocab_size
        self.word_index = word_index
        self.index_word = index_word
        self._init_network()
        X, y = build_cbow_dataset(corpus=corpus, word_index=word_index, windows_size=window_size)

        while not stopped:
            print("Epoch: {}".format(epoch))
            if self.batch_size > 1:
                splitted_indices = np.array_split(np.arange(X.shape[0]), self.batch_size)
            else:
                splitted_indices = np.arange(X.shape[0])
            for index in splitted_indices:
                y_hat = self._forward_pass(X[index, :, :])
                self._backward_pass(y_hat, y[index, :], X[index, :, :])
            y_hat = self._forward_pass(X)
            print("Loss: {}".format(np.sum(cross_entropy(y_hat, y))))
            epoch += 1
            if epoch >= self.epochs:
                stopped = True

    def predict(self, X: np.ndarray):
        pass

    def _forward_pass(self, X: np.ndarray):
        if len(X.shape) == 2:
            X = X.reshape((1, *X.shape))
        # multiply with each slice of 3rd axis and get mean
        X = np.sum(X, axis=2) / np.sum(X > 0, axis=(1, 2)).reshape(X.shape[0],1)
        x_1 = X @ self._hidden_layer.weights
        self._hidden_layer.set_activation(x_1)
        x_2 = x_1 @ self._output_layer.weights
        self._output_layer.set_activation(x_2)
        y_hat = softmax(x_2)

        return y_hat

    def _backward_pass(self, y_hat: np.ndarray, y: np.ndarray, X: np.ndarray):
        if len(X.shape) == 2:
            X = X.reshape((1, *X.shape))
        if len(y_hat.shape) == 1:
            y_hat = y_hat.reshape((1, *y_hat.shape))
        if len(y.shape) == 1:
            y = y.reshape((1, *y.shape))
        cost_gradient = cross_entropy_grad(y_hat=y_hat, y=y)
        X = np.sum(X, axis=2) / np.sum(X > 0, axis=(1, 2)).reshape(X.shape[0], 1)
        output_delta = np.empty((X.shape[0], self._output_layer.weights.shape[0], self._output_layer.weights.shape[1]))
        hidden_delta = np.empty((X.shape[0], self._hidden_layer.weights.shape[0], self._hidden_layer.weights.shape[1]))
        for i in range(X.shape[0]):
            output_delta[i, :, :] = np.outer(self._hidden_layer.activation[i, :], cost_gradient[i, :])
            hidden_delta[i, :, :] = np.outer(X[i, :], (self._output_layer.weights @ cost_gradient[i, :]))

        hidden_delta = np.mean(hidden_delta, axis=0)
        output_delta = np.mean(output_delta, axis=0)
        #output_delta = self._hidden_layer.activation.T @ cost_gradient
        #hidden_delta = np.mean(X, axis=2).T @ (cost_gradient @ self._output_layer.weights.T)

        self._output_layer.weights = self._output_layer.weights - self.learning_rate * output_delta
        self._hidden_layer.weights = self._hidden_layer.weights - self.learning_rate * hidden_delta

    def _clip_gradients(self, arr):
        arr[arr > self.clipping_grad_value] = self.clipping_grad_value
        arr[arr < -self.clipping_grad_value] = -self.clipping_grad_value
        return arr

    def _init_network(self):
        self._hidden_layer = Layer(np.random.rand(self.vocab_size, self.n_hidden_neurons))
        self._output_layer = Layer(np.random.rand(self.n_hidden_neurons, self.vocab_size))

    def get_loss_grad_informations(self):
        values = []
        for i in range(len(self._tracking)):
            tmp = {"epoch": i, "loss": self._tracking[i]["loss"]}

            for j in range(self._tracking[i]["weights_hidden"].shape[0]):
                tmp.update({"W1{}{}".format(j, k): r for k, r in enumerate(self._tracking[i]["weights_hidden"][j, :])})
                tmp.update({"dW1{}{}".format(j, k): r for k, r in enumerate(self._tracking[i]["gradient_hidden"][j, :])})

            for j in range(self._tracking[i]["weights_output"].shape[0]):
                tmp.update({"W2{}{}".format(j, k): r for k, r in enumerate(self._tracking[i]["weights_output"][j, :])})
                tmp.update({"dW2{}{}".format(j, k): r for k, r in enumerate(self._tracking[i]["gradient_output"][j, :])})
            values.append(tmp)
        return pd.DataFrame(values)

    def get_word_vector(self, word: str):
        index = self.word_index[word.lower()]
        return self._hidden_layer.weights[index, :]

    def get_similar_words(self, word: str, n_top: int = 5):
        n_top += 1
        word = word.lower()
        word_vector = self.get_word_vector(word)

        theta_sums = self._hidden_layer.weights @ word_vector.reshape(len(word_vector), 1)
        theta_den = (np.linalg.norm(word_vector) * np.linalg.norm(self._hidden_layer.weights, axis=1)).\
            reshape(theta_sums.shape[0], 1)
        theta = (theta_sums / theta_den).reshape(theta_sums.shape[0], )
        max_indices = np.argpartition(theta, -n_top)[-n_top:]
        return {self.index_word[i]: theta[i] for i in max_indices if self.index_word[i] != word}


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
    return - np.sum(y * np.log(y_hat))


def cross_entropy_grad(y_hat, y):
    return y_hat - y


class Layer:

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def set_activation(self, activation: np.ndarray):
        self.activation = activation
