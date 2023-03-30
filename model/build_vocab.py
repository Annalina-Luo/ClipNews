# import nltk
import pickle
import argparse
from collections import Counter
import json
from tqdm import tqdm

MAX_VOCAB_SIZE = 1000000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


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


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    # with open(file_path, 'r', encoding='UTF-8') as f:
    #     for line in tqdm(f):
    #         lin = line.strip()
    #         if not lin:
    #             continue
    #         content = lin.split('\t')[0]
    #         for word in tokenizer(content):
    #             vocab_dic[word] = vocab_dic.get(word, 0) + 1
    #     vocab_list = sorted([_ for _ in vocab_dic.items() if _[
    #                         1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    #     vocab_dic = {word_count[0]: idx for idx,
    #                  word_count in enumerate(vocab_list)}
    #     vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

    for news in tqdm(file_path):
        content = news["article"]
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_dic['<start>'] = vocab_dic.get(word, 0) + 1
        vocab_dic['<end>'] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[
        1] >= min_freq], key=lambda x: x[1], reverse=True)
    vocab_dic = {word_count[0]: idx for idx,
                 word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

    print(len(vocab))
    pickle.dump(vocab, open(vocab_dir, 'wb'))  # 482845


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

    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    # train_dir = "./THUCNews/data/train.txt"
    # vocab_dir = "./THUCNews/data/vocab1.pkl"
    vocab_dir = ".\\article_vocab.pkl"
    # pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    VisualNews_train = "F:\\NLP\\transform-and-tell\\VisualNews_train.json"
    # emb_dim = 300
    # filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"

    with open(VisualNews_train, mode='r', encoding='utf-8') as f:
        news_list = json.load(f)

    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        def tokenizer(x): return x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        # def tokenizer(x): return [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(
            news_list, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    print(len(word_to_id))
