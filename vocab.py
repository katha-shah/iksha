import nltk
import pickle
from collections import Counter
from tqdm import tqdm
import os

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(vocab_path, vocab_name, captions=["Tamburo re", "babaal na joiye"], threshold=3):
    """Build the vocabulary"""
    counter = Counter()

    print("Tokenizing...")
    for caption in tqdm(captions):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    # vocab.add_word('<pad>') # We use Keras padding utility
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    print("Building vocabulary...")
    for word in tqdm(words):
        vocab.add_word(word)

    vocab_file = os.path.join(vocab_path, vocab_name)
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary to '{}'".format(vocab_file))

    return vocab


def load_vocab(vocab_path, vocab_name):
    """Load the vocabulary"""
    vocab_file = os.path.join(vocab_path, vocab_name)
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    print("Loaded the vocabulary from '{}'".format(vocab_file))
    print("Total vocabulary size: {}".format(len(vocab)))
    return vocab

