import abc

import numpy as np


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def change_weights(self, weights: np.ndarray, gradients: np.ndarray, name: str = None):
        """

        :param name: id/name of weights and gradients which is used when current gradient are combined with
            past gradients
        :param weights:
        :param gradients:
        :return:
        """
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def change_weights(self, weights: np.ndarray, gradients: np.ndarray, name: str = None):
        return weights - self.learning_rate * gradients


class RMSProp(Optimizer):

    def __init__(self, inital_learning_rate: float = 0.3, exp_factor: float = 0.9):
        self.p = exp_factor
        self.e = 1e-10
        self.eta = inital_learning_rate
        self._past_vs = {}

    def change_weights(self, weights: np.ndarray, gradients: np.ndarray, name: str = None):
        assert name is not None, "name needs to be set for rmsprop"
        if name not in self._past_vs:
            v = np.power(gradients, 2)
        else:
            v = self.p * self._past_vs[name] + (1 - self.p) * np.power(gradients, 2)
        self._past_vs[name] = v
        weights_delta = - (self.eta / np.sqrt(v + self.e)) * gradients
        return weights + weights_delta


class Adam(Optimizer):

    def __init__(self, inital_learning_rate: float = 0.3, beta_one: float = 0.9, beta_two: float = 0.99):
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.e = 1e-10
        self.eta = inital_learning_rate
        self._past_vs = {}
        self._past_ss = {}

    def change_weights(self, weights: np.ndarray, gradients: np.ndarray, name: str = None):
        assert name is not None, "name needs to be set for adam"
        if name not in self._past_vs:
            v = gradients
            s = np.power(gradients, 2)
        else:
            v = self.beta_one * self._past_vs[name] + (1 - self.beta_one) * gradients
            s = self.beta_two * self._past_ss[name] + (1 - self.beta_two) * np.power(gradients, 2)
        self._past_vs[name] = v
        self._past_ss[name] = s
        weights_delta = - self.eta * (v / np.sqrt(s + self.e)) * gradients
        return weights + weights_delta
