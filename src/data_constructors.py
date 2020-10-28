from collections import Counter
import numpy as np


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


def build_cbow_dataset(corpus: list, word_index: dict, windows_size: int):
    length_data = np.sum([len(l) for l in corpus])
    X = np.zeros((length_data, len(word_index), windows_size * 2))
    y = np.zeros((length_data, len(word_index)))

    i = 0
    for sentence in corpus:
        for j, word in enumerate(sentence):
            y[i, word_index[word]] = 1
            len_sentence = len(sentence)

            for k in range(1, windows_size + 1):
                if j + k < len_sentence:
                    X[i, word_index[sentence[j + k]], k + (windows_size - 1)] = 1
                if j - k >= 0:
                    X[i, word_index[sentence[j - k]], windows_size - k] = 1
            i += 1

    return X, y
