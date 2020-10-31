import unittest
import numpy as np
from src.helpers import softmax, softmax_grad, cross_entropy, cross_entropy_grad


class HelpersTester(unittest.TestCase):

    p = np.array([[0.25, 0.89], [0.73, 0.13]])
    y = np.array([[0, 1], [0, 1]])

    def test_softmax(self):
        pred = softmax(self.p)
        self.assertListEqual([[0.3452465393936808, 0.6547534606063192], [0.6456563062257955, 0.35434369377420455]],
                             pred.tolist())

    def test_softmax_grad(self):
        pred = softmax_grad(self.p)
        print(pred)
        self.assertListEqual([[0.2260513664303684, 0.2260513664303684], [0.2287842404566573, 0.2287842404566573]],
                             pred.tolist())

    def test_cross_entropy(self):
        loss = cross_entropy(self.p, self.y)
        self.assertEqual(loss, 2.1567546447825063)

    def test_cross_entropy_grad(self):
        grad = cross_entropy_grad(self.p, self.y)
        self.assertListEqual(grad.tolist(), [[0.25, -0.10999999999999999], [0.73, -0.87]])