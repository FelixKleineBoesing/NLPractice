from typing import List

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Embedding, GRU, Dense

from src.models.attention_layers import AttentionDotProduct


class Decoder(Model):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_units: List[int]):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, embedding_dim)
        self.lstm_layers = [
            GRU(
                unit,
                return_sequences=False if i < len(hidden_units) - 1 else True,
                return_state=False if i < len(hidden_units) - 1 else True
            )
            for i, unit in enumerate(hidden_units)
        ]
        self.prob_layer = Dense(vocab_size, activation="softmax")
        self.attention_layer = AttentionDotProduct()

    def call(self, decoder_input, decoder_hidden, encoder_sequence_output):

        x = self.embedding_layer(decoder_input)
        weighted_sum_encoder, attention_scores = self.attention_layer(decoder_hidden, encoder_sequence_output)

        x = tf.concat([weighted_sum_encoder, x], axis=-1)
        x = tf.expand_dims(x, 1)
        for layer in self.lstm_layers:
            x = layer(x)

        decoder_output, decoder_state = x

        word_prob = self.prob_layer(decoder_output)

        return word_prob, decoder_state, attention_scores


