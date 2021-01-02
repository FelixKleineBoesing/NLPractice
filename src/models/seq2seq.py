from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, LSTM, Dense
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import Optimizer, Adam
from tensorflow.python.keras.losses import Loss, SparseCategoricalCrossentropy

from src.models.attention_layers import AttentionDotProduct


class Encoder(Model):
    # Doc of subclass model:
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_units: List[int]):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, embedding_dim)
        self.lstm_layers = [
            LSTM(
                unit,
                return_sequences=True,
                return_state=False if i < len(hidden_units) - 1 else True
            )
            for i, unit in enumerate(hidden_units)
        ]

    def call(self, inputs):
        x = self.embedding_layer(input)
        for layer in self.lstm_layers:
            x = layer(x)
        return x


class Decoder(Model):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_units: List[int]):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, embedding_dim)
        self.lstm_layers = [
            LSTM(
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

        word_prob = self.prob_layer()

        return word_prob, decoder_state, attention_scores


class Seq2Seq:

    def __init__(self, encoder: Encoder, decoder: Decoder, optimizer: Optimizer = None, loss_function: Loss = None):
        if optimizer is None:
            optimizer = Adam(0.01)
        if loss_function is None:
            loss_function = SparseCategoricalCrossentropy(reduction="none")
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.encoder = encoder
        self.decoder = decoder

    def calculate_loss(self, actuals, predictions):
        mask = tf.math.logical_not(tf.math.equal(actuals, 0))
        loss = self.loss_function(actuals, predictions)
        mask = tf.cast(mask, dtype=loss.dtype)
        return tf.reduce_mean(loss * mask)

    def _train_step(self, input, output):
        loss = 0
        with tf.GradientTape() as g:
            enc_seq_output, enc_hidden = self.encoder(input)
            dec_hidden = enc_hidden
            dec_input = input[:, 0]
            for i in range(1, output.shape[1]):
                word_prob, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_seq_output)
                actuals = output[:, i]

                if np.count_nonzero(actuals) == 0:
                    break

                loss += self.loss_function(actuals, word_prob)

                dec_input = actuals

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = g.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss.numpy()

    def train(self, input_language, output_language, number_epochs: int):

        # create dataset here from input and output language

        for epoch in range(number_epochs):
            pass
