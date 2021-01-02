import re

import numpy as np
import pandas as pd


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
    sen = '<BOS> ' + sen + ' <EOS>'

    sen = ' '.join(sen.split())
    return sen


def clean_and_merge_sentences(german_path: str, english_path: str, output_file_path: str):
    german, english = [], []
    with open(german_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            german.append(clean_sentence(line))

    with open(english_path, "r", encoding="utf-8") as g:
        for line in g.readlines():
            english.append(clean_sentence(line))

    data = pd.DataFrame({"german": german, "english": english})

    data.to_csv(output_file_path, index=False)
