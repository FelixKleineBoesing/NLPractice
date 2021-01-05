from typing import List

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Embedding, GRU


class Encoder(Model):
    # Doc of subclass model:
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_units: List[int]):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, embedding_dim)
        self.lstm_layers = [
            GRU(
                unit,
                return_sequences=True,
                return_state=False if i < len(hidden_units) - 1 else True
            )
            for i, unit in enumerate(hidden_units)
        ]

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        for layer in self.lstm_layers:
            x = layer(x)
        return x