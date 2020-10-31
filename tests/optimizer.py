import unittest
import numpy as np
from src.optimizer import SGD, RMSProp, Adam


class OptimizerTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.weights = np.array([1, 1.5, 2])
        cls.gradients_t_one = np.array([-0.5, 0.65, -0.35])
        cls.gradients_t_two = np.array([-0.15, 0.52, -0.25])

    def test_sgd(self):
        optimizer = SGD(0.5)
        new_weights = optimizer.change_weights(self.weights, self.gradients_t_one).tolist()
        self.assertListEqual([1.25, 1.175, 2.175], new_weights)
        new_weights = optimizer.change_weights(new_weights, self.gradients_t_two).tolist()
        self.assertListEqual([1.325, 0.915, 2.3], new_weights)

    def test_rmsprop(self):
        optimizer = RMSProp()
        new_weights = optimizer.change_weights(self.weights, self.gradients_t_one, "output").tolist()
        self.assertListEqual([1.29999999994, 1.200000000035503, 2.299999999877551], new_weights)
        new_weights = optimizer.change_weights(new_weights, self.gradients_t_two, "output").tolist()
        self.assertListEqual([1.3943975162483633, 0.9555597469474979, 2.5197345255241372], new_weights)

    def test_adam(self):
        optimizer = Adam()
        new_weights = optimizer.change_weights(self.weights, self.gradients_t_one, "output").tolist()
        self.assertListEqual([0.85000000003, 1.305000000023077, 1.895000000042857], new_weights)
        new_weights = optimizer.change_weights(new_weights, self.gradients_t_two, "output").tolist()
        self.assertListEqual([0.8079582730046314, 1.151844070808433, 1.8219637734317233], new_weights)

