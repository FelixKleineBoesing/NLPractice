
from src.data_constructors import build_word_indices, build_cbow_dataset
from src.network import Word2Vec


def main():
    # - Gandalf the Grey
    txt = "It is not our part to master all the tides of the world, " \
          "but to do what is in us for the succor of those years wherein we are set, " \
          "uprooting the evil in the fields that we know, so that those who live after may have clean earth to till. " \
          "What weather they shall have is not ours to rule."
    txt_test = "This is is test"
    window_size = 2

    corp = [[w.lower() for w in sntc.split()] for sntc in txt_test.split(".") if len(sntc) > 0]

    vocab_size, word_index, index_word = build_word_indices(corp)
    X, y = build_cbow_dataset(corpus=corp, word_index=word_index, windows_size=window_size)

    word2vec = Word2Vec(vocab_size=vocab_size, n_hidden_neurons=2, epochs=10, learning_rate=0.3, batch_size=1)
    w1, w2 = word2vec._hidden_layer.weights.copy(), word2vec._output_layer.weights.copy()
    word2vec.train(X=X, y=y)

    corp = [[w.lower() for w in sntc.split()] for sntc in txt.split(".") if len(sntc) > 0]

    vocab_size, word_index, index_word = build_word_indices(corp)
    X, y = build_cbow_dataset(corpus=corp, word_index=word_index, windows_size=window_size)

    word2vec = Word2Vec(vocab_size=vocab_size, n_hidden_neurons=10, epochs=20, learning_rate=0.2)
    word2vec.train(X=X, y=y)


if __name__ == "__main__":
    main()
