import nltk
import pickle
import argparse
from collections import Counter
import json
from tqdm import tqdm


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word, ind=None):
        if not word in self.word2idx:
            if ind == None:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
            else:
                self.word2idx[word] = ind
                self.idx2word[ind] = word

    def __call__(self, word):
        if not word in self.word2idx:
            self.add_word(word)
            return len(self.word2idx)-1
            # return len(self.word2idx)+1
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


if __name__ == '__main__':
    vocab_dir = ".\\vocab.pkl"
    VisualNews_train = ".\\train.json"
    vocab = Vocabulary()

    with open(VisualNews_train, mode='r', encoding='utf-8') as f:
        news_list = json.load(f)

    def tokenizer(x): return x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)

    for news in tqdm(news_list):
        content = news["article"][:300]
        for word in tokenizer(content):
            vocab.add_word(word)
    # vocab.add_word('<unk>', vocab("<unk>"))

    print(len(vocab))
    pickle.dump(vocab, open(vocab_dir, 'wb'))  # 482845
