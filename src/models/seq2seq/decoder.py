from abc import abstractmethod
import abc
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


class SentencePredictor(abc.ABC):

    @abstractmethod
    def predict_sentence(self, decoder: Decoder, decoder_input, decoder_hidden, encoder_sequence_output,
                         stop_word_id: int):
        pass


class GreedyPredictor(SentencePredictor):

    def __init__(self, max_words_in_sentence: int):
        self.max_words_in_sentence = max_words_in_sentence

    def predict_sentence(self, decoder: Decoder, decoder_input, decoder_hidden, encoder_sequence_output,
                         stop_word_id: int):
        output_seq = []
        for _ in range(self.max_words_in_sentence):
            word_prob, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_sequence_output)
            pred_word_id = tf.argmax(word_prob[0, 0, :]).numpy()
            output_seq.append(pred_word_id)
            if pred_word_id == stop_word_id:
                break

            decoder_input = tf.convert_to_tensor([pred_word_id])

        return output_seq


class BeamSearchPredictor(SentencePredictor):

    def __init__(self, max_words_in_sentence: int, k: int = 3):
        self.max_words_in_sentence = max_words_in_sentence
        self.k = k

    def predict_sentence(self, decoder: Decoder, decoder_input, decoder_hidden, encoder_sequence_output,
                         stop_word_id: int):
        output_seq = {'depth': 0, "score": 0}
        searched = False
        while not searched:
            self._predict_word(decoder=decoder, decoder_input=decoder_input, decoder_hidden=decoder_hidden,
                               encoder_sequence_output=encoder_sequence_output, stop_word_id=stop_word_id,
                               root=output_seq)

    def _predict_word(self, decoder, decoder_input, decoder_hidden, encoder_sequence_output, stop_word_id, root):
        node = {}
        word_prob, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_sequence_output)
        probs, indices = tf.math.top_k(word_prob[0, 0, :], k=self.k)
        depth = root["depth"] + 1
        for prob, index in zip(probs, indices):
            stop = True if depth >= self.max_words_in_sentence or index == stop_word_id else False
            node[index] = {"score": root["score"] + tf.math.log(prob), "stop": stop, "depth": depth}
        node["hidden_state"] = decoder_hidden
        root.update(node)

    def _get_next_node(self, tree: dict):

