import unittest

from src.optimizer import Optimizer
from src.word2vec import Word2Vec


class Word2VecTester(unittest.TestCase):

    def test_construction(self):
        word2vec = Word2Vec()
        self.assertTrue(word2vec.learning_rate, 0.1)
        self.assertTrue(isinstance(word2vec.optimizer, Optimizer))
        self.assertTrue(word2vec.n_hidden_neurons, 20)
        self.assertTrue(word2vec.epochs, 20)
        self.assertTrue(word2vec.clipping_grad_value, 50)
        self.assertTrue(word2vec.batch_size, 1)
        self.assertTrue(word2vec.method, "cbow")

    def test_train(self):
        word2vec = Word2Vec(epochs=50)
        test = "This is a nice test"
        word2vec.train(corpus=test, window_size=2)
        self.assertTrue(word2vec.loss_per_epoch[-1] < 1.5)

