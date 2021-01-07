from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
from src.helpers import clean_and_merge_sentences
from src.models import Encoder
from src.models.seq2seq.decoder import Decoder
from src.models.seq2seq.sentence_predictor import BeamSearchPredictor, GreedyPredictor
from src.models.word2vec import Word2Vec
from src.optimizer import Adam
from src.models.seq2seq.seq2seq import Seq2Seq


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
    share_data = 1.0
    data = pd.read_csv("../data/german-english/cleaned_langs.csv")
    indices = np.random.choice(np.arange(data.shape[0]), size=int(share_data*data.shape[0]))
    data = data.iloc[indices, :]
    german = data["german"].tolist()
    english = data["english"].tolist()

    encoder = Encoder(vocab_size=10000, embedding_dim=128, hidden_units=[512])
    decoder = Decoder(vocab_size=10000, embedding_dim=128, hidden_units=[512])
    sentence_predictor = GreedyPredictor(max_words_in_sentence=20)
    sentence_predictor = BeamSearchPredictor(k=3, max_words_in_sentence=20)

    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder, sentence_predictor=sentence_predictor, batch_size=128,
                      num_words=10000)

    seq2seq.train(english, german, number_epochs=1)

    translated_sentence = seq2seq.translate_sentence("This is my first try of a seq2seq model with tensorflow")
    print(translated_sentence)
    translated_sentence1 = seq2seq.translate_sentence("We welcome you to a calm and warm ambience")
    print(translated_sentence1)


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
