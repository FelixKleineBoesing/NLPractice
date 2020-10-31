
from src.data_constructors import build_word_indices, build_cbow_dataset
from src.word2vec import Word2Vec
from src.optimizer import RMSProp, Adam


def main():
    # - Gandalf the Grey
    txt = "It is not our part to master all the tides of the world, " \
          "but to do what is in us for the succor of those years wherein we are set, " \
          "uprooting the evil in the fields that we know, so that those who live after may have clean earth to till. " \
          "What weather they shall have is not ours to rule."
    txt_test = "This is is test"

    corp = [[w.lower() for w in sntc.split()] for sntc in txt_test.split(".") if len(sntc) > 0]

    word2vec = Word2Vec(n_hidden_neurons=2, epochs=10, learning_rate=0.3, batch_size=2, method="cbow",
                        optimizer=Adam())
    word2vec.train(corpus=corp, window_size=2)

    corp = [[w.lower() for w in sntc.split()] for sntc in txt.split(".") if len(sntc) > 0]

    word2vec = Word2Vec(n_hidden_neurons=10, epochs=50, learning_rate=0.2)
    word2vec.train(corpus=corp, window_size=2)
    print(word2vec.get_similar_words("evil"))


if __name__ == "__main__":
    main()
