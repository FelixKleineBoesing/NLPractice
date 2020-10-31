from collections import Counter
import numpy as np
from typing import Union


def build_word_indices(corpus):
    word_list = set()
    word_count = Counter()
    for sntc in corpus:
        word_count += Counter(sntc)
        word_list.update(sntc)

    vocab_size = len(word_count)
    # Since we access the index of a word very often, we prebuild indices
    word_index = {w: i for i, w in enumerate(word_list)}
    index_word = {i: w for w, i in word_index.items()}
    return vocab_size, word_index, index_word


def build_cbow_dataset(corpus: Union[list, str], word_index: dict, windows_size: int):
    if isinstance(corpus, str):
        corpus = [[w.lower() for w in sntc.split()] for sntc in corpus.split(".") if len(sntc) > 0]

    target, context = _build_target_context(corpus=corpus, word_index=word_index, windows_size=windows_size)
    return context, target


def build_skipgram_dataset(corpus: Union[list, str], word_index: dict, windows_size: int):
    if isinstance(corpus, str):
        corpus = [[w.lower() for w in sntc.split()] for sntc in corpus.split(".") if len(sntc) > 0]
    target, context = _build_target_context(corpus=corpus, word_index=word_index, windows_size=windows_size)
    target = target.reshape((*target.shape, 1))
    return target, context


def _build_target_context(corpus: Union[list, str], word_index: dict, windows_size: int):
    length_data = np.sum([len(l) for l in corpus])
    X = np.zeros((length_data, len(word_index)))
    y = np.zeros((length_data, len(word_index), windows_size * 2))

    i = 0
    for sentence in corpus:
        for j, word in enumerate(sentence):
            X[i, word_index[word]] = 1
            len_sentence = len(sentence)

            for k in range(1, windows_size + 1):
                if j + k < len_sentence:
                    y[i, word_index[sentence[j + k]], k + (windows_size - 1)] = 1
                if j - k >= 0:
                    y[i, word_index[sentence[j - k]], windows_size - k] = 1
            i += 1

    return X, y