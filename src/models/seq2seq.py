from typing import List, Union
import time
import numpy as np
import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
from tensorflow.keras import Model
from tensorflow.python.data import Dataset
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.helpers import tokenize_sentences, arraylike, create_train_test_splits, clean_sentence
from src.models.attention_layers import AttentionDotProduct

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


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


class Seq2Seq:

    def __init__(self, encoder: Encoder, decoder: Decoder, optimizer: Optimizer = None, loss_function: Loss = None,
                 num_words: int = 10000, batch_size: int = 512, test_ratio: float = 0.3,
                 max_words_in_sentence: int = 20):
        if optimizer is None:
            optimizer = Adam(0.01)
        if loss_function is None:
            loss_function = SparseCategoricalCrossentropy(reduction="none")
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.encoder = encoder
        self.decoder = decoder
        self.num_words = num_words
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.trained = False
        self.loss_per_epoch = []
        self.input_tokenizer, self.output_tokenizer = None, None
        self.input_word_index, self.output_word_index = None, None
        self.max_words_in_sentence = max_words_in_sentence

    def summary(self):
        print("Model Summary Encoder:")
        self.encoder.summary()
        print("Model Summary Decoder:")
        self.decoder.summary()

    def calculate_loss(self, actuals, predictions):
        mask = tf.math.logical_not(tf.math.equal(actuals, 0))
        loss = self.loss_function(actuals, predictions)
        mask = tf.cast(mask, dtype=loss.dtype)
        return tf.reduce_mean(loss * mask)

    def _train_step(self, input, output):
        with tf.GradientTape() as g:
            loss = self._forward_pass(input=input, output=output)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = g.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss / int(output.shape[1])

    def _forward_pass(self, input, output):
        loss = 0
        enc_seq_output, enc_hidden = self.encoder(input)
        # this is neccessary so that decoder and encoder will be optimized as one system
        dec_hidden = enc_hidden
        dec_input = input[:, 0]
        for i in range(1, output.shape[1]):
            word_prob, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_seq_output)
            actuals = output[:, i]

            if np.count_nonzero(actuals) == 0:
                break

            loss += self.calculate_loss(actuals, word_prob)

            dec_input = actuals
        return loss

    def train(self, input_language: arraylike, output_language: arraylike, number_epochs: int = 20):
        assert len(input_language) == len(output_language), "input and output language must be equal length, " \
                                                            "since the observations must be matching translations"

        input_word_index, input_sentence, input_tokenizer = tokenize_sentences(input_language,
                                                                               num_words=self.num_words)
        output_word_index, output_sentence, output_tokenizer = tokenize_sentences(output_language,
                                                                                  num_words=self.num_words)
        self.input_tokenizer, self.output_tokenizer = input_tokenizer, output_tokenizer
        self.input_word_index, self.output_word_index = input_word_index, output_word_index

        input_splits, output_splits = create_train_test_splits(input_sentence, output_sentence, ratio=self.test_ratio)
        buffer_size = input_splits[0].shape[0]
        dataset = Dataset.from_tensor_slices((input_splits[0], output_splits[0])).shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(self.batch_size)
        steps_per_epoch = buffer_size // self.batch_size

        for epoch in range(number_epochs):
            epoch_start = time.time()

            loss_per_batch = []
            for (batch, (input, target)) in enumerate(dataset.take(steps_per_epoch)):
                batch_start = time.time()
                batch_loss = self._train_step(input, target)
                if batch % int(steps_per_epoch / 5) == 0:
                    logging.warning("Epoch: {} \t Batch: {} \t Loss: {:.4f} \t Time taken: {}".
                                 format(epoch + 1, batch, batch_loss, time.time() - batch_start))

                loss_per_batch.append(batch_loss)

            epoch_loss = sum(loss_per_batch) / steps_per_epoch
            self.loss_per_epoch.append(epoch_loss)
            logging_string = "Epoch: {} \t Loss: {} \t Time taken: {}".format(
                epoch + 1, epoch_loss, time.time() - epoch_start
            )
            # if self.test_ratio > 0.0:
            #     val_loss = self._forward_pass(input_splits[1], output_splits[1])
            #     logging_string + " \t Val Loss: {}".format(val_loss)
            logging.info(logging_string)
        self.trained = True

    def dump_graph(self):
        pass

    def load_graph(self):
        pass

    def _predict(self):
        pass

    def translate_sentence(self, sentence: str):
        sentence = clean_sentence(sentence)
        seq = pad_sequences(
            self.input_tokenizer.texts_to_sequences([sentence]),maxlen=self.max_words_in_sentence, padding="post"
        )
        enc_input = tf.convert_to_tensor(seq)
        enc_seq_output, enc_hidden = self.encoder(enc_input)
        dec_input = tf.convert_to_tensor([self.output_word_index["<bos>"]])
        dec_hidden = enc_hidden
        sentence_end_word_id = self.output_word_index["<eos>"]
        output_seq = []
        for i in range(self.max_words_in_sentence * 2):
            word_prob, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_seq_output)
            pred_word_id = tf.argmax(word_prob[0, 0, :]).numpy()
            output_seq.append(pred_word_id)
            if pred_word_id == sentence_end_word_id:
                break

            dec_input = tf.convert_to_tensor([pred_word_id])

        return self.output_tokenizer.sequences_to_texts([output_seq])
