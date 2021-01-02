from pathlib import Path

import pandas as pd

from src.helpers import clean_and_merge_sentences
from src.models.word2vec import Word2Vec
from src.optimizer import Adam
from src.models.seq2seq import Encoder, Decoder, Seq2Seq


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
    data = pd.read_csv("../data/german-english/miniset_langs.csv")
    german = data["german"].tolist()
    english = data["english"].tolist()

    encoder = Encoder(vocab_size=10000, embedding_dim=64, hidden_units=[1024])
    decoder = Decoder(vocab_size=10000, embedding_dim=64, hidden_units=[1024])

    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder)
    seq2seq.train(english, german, number_epochs=20)


def main_preprocess():
    number_records_miniset = 2000
    clean_and_merge_sentences(german_path=Path("..", "data", "german-english", "german.txt"),
                              english_path=Path("..", "data", "german-english", "english.txt"),
                              output_file_path=Path("..", "data", "german-english", "cleaned_langs.csv"))
    data = pd.read_csv(Path("..", "data", "german-english", "cleaned_langs.csv"))
    data.iloc[:number_records_miniset, :].to_csv(Path("..", "data", "german-english", "miniset_langs.csv"),
                                                 index=False)



if __name__ == "__main__":
    #main_word2vec()
    #main_preprocess()
    main_seq2seq()