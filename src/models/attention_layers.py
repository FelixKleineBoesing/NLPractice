import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class AttentionDotProduct(Layer):

    def call(self, decoder_hidden_state, encoder_outputs):
        decoder_hidden_state_through_time = tf.expand_dims(decoder_hidden_state, 2)
        attention_scores = tf.matmul(encoder_outputs, decoder_hidden_state_through_time)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)
        weighted_sum = tf.reduce_sum(encoder_outputs * attention_scores, axis = 1)
        return weighted_sum, attention_scores