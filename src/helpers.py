import re
from typing import Union

import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


arraylike = Union[list, np.array, pd.Series, tf.Tensor]


def softmax(X: np.ndarray):
    assert (len(X.shape) == 2), "x must be two dimensional"
    exp_x = np.exp(X - np.max(X))
    return exp_x / np.sum(exp_x, axis=1).reshape(X.shape[0], 1)


def softmax_grad(X: np.ndarray):
    return softmax(X) * (1 - softmax(X))


def cross_entropy(y_hat, y):
    if len(y.shape) == 3:
        return - np.sum([y[:, :, i] * np.log(y_hat) for i in range(y.shape[2])])
    else:
        return - np.sum(y * np.log(y_hat))


def cross_entropy_grad(y_hat, y):
    if len(y.shape) == 3:
        return np.sum([y_hat[:, :, 0] - y[:, :, i] for i in range(y.shape[2])], axis=0)
    else:
        return y_hat - y


class Layer:

    def __init__(self, weights: np.ndarray):
        self.weights = weights
        self.activation = None

    def set_activation(self, activation: np.ndarray):
        self.activation = activation


def clean_sentence(sntc: str):
    sen = sntc.strip('.')

    # insert space between words and punctuations
    sen = re.sub(r"([?.!,¿;।])", r" \1 ", sen)
    sen = re.sub(r'[" "]+', " ", sen)

    sen = re.sub(r"[^a-zA-Z?.!,¿']+", " ", sen)
    sen = sen.lower()

    sen = sen.strip()
    sen = '<bos> ' + sen + ' <eos>'

    sen = ' '.join(sen.split())
    return sen


def clean_and_merge_sentences(german_path: str, english_path: str, output_file_path: str,
                              max_words_in_sentence: int = 20):
    """

    :param german_path:
    :param english_path:
    :param output_file_path:
    :param max_words_in_sentence: excludes all sentences that are too long to keep matrices dense
    :return:
    """
    german, english = [], []
    f = open(german_path, "r", encoding="utf-8")
    g = open(english_path, "r", encoding="utf-8")

    for line_ge, line_en in zip(f.readlines(), g.readlines()):
        cleaned_line_ge = clean_sentence(line_ge)
        cleaned_line_en = clean_sentence(line_en)
        len_ge, len_en = cleaned_line_ge.count(" "), cleaned_line_en.count(" ")

        if cleaned_line_en == "" or cleaned_line_ge == "" or len_ge > max_words_in_sentence - 1 \
                or len_en > max_words_in_sentence - 1:
            continue

        english.append(cleaned_line_en)
        german.append(cleaned_line_ge)

    data = pd.DataFrame({"german": german, "english": english})

    data.to_csv(output_file_path, index=False)


def tokenize_sentences(sentences, num_words: int = 30000, oov_token: str = "<UNK>"):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = pad_sequences(sequences, padding='post')
    return word_index, sequences, tokenizer


def create_train_test_splits(*args: Union[np.ndarray, tf.Tensor, list], ratio: float = 0.3, ):
    assert len(args) > 0, "please supply at least one data container"
    len_arr = get_number_obs(args[0])
    for arg in args:
        assert len_arr == get_number_obs(arg), "All data containers must have the same shape or length"
    test_size = int(ratio * len_arr)
    indices = np.random.choice(np.arange(len_arr), size=test_size, replace=False)
    mask = np.ones(len_arr, dtype=bool)
    mask[indices] = False

    splits = []
    for arg in args:
        if isinstance(arg, list):
            arg = np.array(arg)
            split = arg[mask].to_list(), arg[~mask].to_list()
        else:
            split = arg[mask], arg[~mask]
        splits.append(split)
    if len(args) == 1:
        return splits[0]
    else:
        return splits


def get_number_obs(obj):
    if hasattr(obj, "shape"):
        return obj.shape[0]
    elif hasattr(obj, "__len__"):
        return len(obj)
    else:
        raise ValueError("Could not get number observations. Obj has no attribute length or shape")


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size
