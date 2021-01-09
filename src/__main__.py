from pathlib import Path

from src.helpers import clean_and_merge_sentences
from src.models.seq2seq.sentence_predictor import GreedyPredictor
from src.models.word2vec import Word2Vec
from src.models.wrappers import train_seq2seq, predict_seq2seq
from src.optimizer import Adam


def main_word2vec():
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


def main_seq2seq():
    train_seq2seq()
    sentence_predictor = GreedyPredictor(max_words_in_sentence=20)
    predict_seq2seq(sentence_predictor=sentence_predictor)


def main_preprocess():
    clean_and_merge_sentences(german_path=Path("..", "data", "german-english", "german.txt"),
                              english_path=Path("..", "data", "german-english", "english.txt"),
                              output_file_path=Path("..", "data", "german-english", "cleaned_langs.csv"))


if __name__ == "__main__":
    #my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    #tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
    #main_word2vec()
    #main_preprocess()
    main_seq2seq()
