import abc
from abc import abstractmethod
import numpy as np
import time
import tensorflow as tf

from src.models.seq2seq.decoder import Decoder


class BeamSearchNode:

    def __init__(self, word_id, hidden_state, depth, score, callback, stopped=False):
        self.word_id = word_id
        self.hidden_state = hidden_state
        self.score = score
        self.depth = depth
        self.children = None
        self._children_calculated = None
        self.callback = callback
        self.stopped = stopped
        if not stopped:
            self._all_childs_calculated = False
        else:
            self.callback()
            self._all_childs_calculated = True
        super().__init__()

    @property
    def all_childs_calculated(self):
        return self._all_childs_calculated

    @all_childs_calculated.setter
    def all_childs_calculated(self, value: bool = True):
        self._all_childs_calculated = value
        if value:
            self.callback()
            self.hidden_state = None

    def append_children(self, nodes: list):
        self.children = nodes
        self._all_childs_calculated = False
        self.are_all_childs_calculated()

    def are_all_childs_calculated(self, search_nodes: bool = False):
        if self.children is not None:
            if search_nodes:
                for child in self.children:
                    child.are_all_childs_calculated()
            if all(child.all_childs_calculated for child in self.children):
                self.all_childs_calculated = True


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
        self.max_nodes = k ** (self.max_words_in_sentence - 1)
        self.adjusted_max_nodes = self.max_nodes

    def predict_sentence(self, decoder: Decoder, decoder_input, decoder_hidden, encoder_sequence_output,
                         stop_word_id: int):
        def dummy():
            return None
        root = BeamSearchNode(word_id=None, hidden_state=None, depth=0, score=0, callback=dummy, stopped=False)
        searched = False
        tree = root
        calculated_nodes = 0
        epoch_start = time.time()
        while not searched:
            self._predict_word(decoder=decoder, decoder_input=decoder_input, decoder_hidden=decoder_hidden,
                               encoder_sequence_output=encoder_sequence_output, stop_word_id=stop_word_id,
                               root=tree)
            parent_tree = _get_next_node(root)
            if parent_tree is None:
                break

            decoder_input = tf.convert_to_tensor([parent_tree[1].word_id])
            decoder_hidden = parent_tree[0].hidden_state
            tree = parent_tree[1]
            calculated_nodes += 1
            if calculated_nodes % (self.max_nodes / 100) == 0:
                print("Calculated {} % of maximum nodes".format(calculated_nodes / self.max_nodes))
            if calculated_nodes % 100 == 0:
                print("calculated_nodes: {} Took: {}".format(calculated_nodes, time.time()-epoch_start))
                epoch_start = time.time()

        best_sentence, score = _get_best_sentence_from_tree(root)
        return best_sentence

    def _predict_word(self, decoder, decoder_input, decoder_hidden, encoder_sequence_output, stop_word_id, root):
        nodes = []
        word_prob, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_sequence_output)
        probs, indices = tf.math.top_k(word_prob[0, 0, :], k=self.k)
        depth = root.depth + 1
        for prob, index in zip(probs.numpy(), indices.numpy()):
            stop = True if depth >= self.max_words_in_sentence or index == stop_word_id else False
            nodes.append(
                BeamSearchNode(word_id=index, hidden_state=None, depth=depth,
                               score=root.score + tf.math.log(prob), stopped=stop,
                               callback=root.are_all_childs_calculated)
            )
        root.hidden_state = decoder_hidden
        root.append_children(nodes)


def _get_best_sentence_from_tree(tree: BeamSearchNode):
    if tree.children is not None:
        max_sequence, max_score = None, -np.Inf
        for child in tree.children:
            sequence, score = _get_best_sentence_from_tree(child)
            if score > max_score:
                max_sequence, max_score = sequence, score
        output_sequence = []
        if tree.word_id is not None:
            output_sequence.append(tree.word_id)
        return output_sequence + max_sequence, max_score
    else:
        return [tree.word_id], tree.score


def _get_next_node(tree: BeamSearchNode) -> (BeamSearchNode, BeamSearchNode):
    """

    :param tree:
    :return: (ParentNode, ChildNode)
    """
    if tree.children is not None:
        for child in tree.children:
            if not child.all_childs_calculated:
                if child.children is not None:
                    return _get_next_node(child)
                elif not child.stopped:
                    return tree, child
                else:
                    print("I dont know when this could be reached")
    elif not tree.stopped:
        return None, tree
    else:
        print("All nodes are calculated")
