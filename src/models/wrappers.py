import pandas as pd
import numpy as np

from src.models import Encoder, Decoder, GreedyPredictor, Seq2Seq
from src.models.seq2seq.sentence_predictor import SentencePredictor


def train_seq2seq(share_data: float = 0.1, number_epochs: int = 100,
                  input_path: str = "../data/german-english/cleaned_langs.csv",
                  checkpoint_path: str = "../data/german-english/seq2seq-checkpoints"):
    vocab_size = 30000
    data = pd.read_csv(input_path)
    indices = np.random.choice(np.arange(data.shape[0]), size=int(share_data*data.shape[0]))
    data = data.iloc[indices, :]
    german = data["german"].tolist()
    english = data["english"].tolist()

    encoder = Encoder(vocab_size=vocab_size, embedding_dim=128, hidden_units=[512])
    decoder = Decoder(vocab_size=vocab_size, embedding_dim=128, hidden_units=[512])
    sentence_predictor = GreedyPredictor(max_words_in_sentence=20)

    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder, sentence_predictor=sentence_predictor, batch_size=128,
                      num_words=vocab_size, checkpoint_dir=checkpoint_path, test_ratio=0.05)

    seq2seq.train(english, german, number_epochs=number_epochs)


def predict_seq2seq(sentence_predictor: SentencePredictor = None,
                    checkpoint_path: str = "../data/german-english/seq2seq-checkpoints"):
    if sentence_predictor is None:
        sentence_predictor = GreedyPredictor(max_words_in_sentence=20)
    vocab_size = 30000
    encoder = Encoder(vocab_size=vocab_size, embedding_dim=128, hidden_units=[512])
    decoder = Decoder(vocab_size=vocab_size, embedding_dim=128, hidden_units=[512])

    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder, sentence_predictor=sentence_predictor, batch_size=128,
                      num_words=vocab_size, checkpoint_dir=checkpoint_path)

    translated_sentence = seq2seq.translate_sentence("This is my first try of a seq2seq model with tensorflow")
    print(translated_sentence)
    translated_sentence1 = seq2seq.translate_sentence("We welcome you to a calm and warm ambience")
    print(translated_sentence1)