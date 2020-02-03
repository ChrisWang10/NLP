import torch.nn as nn
import torch
import random
import numpy as np
import pkuseg
import os
from tqdm import tqdm
import pickle
from utils.data_formater import remove_stop_words
from config import Cfg

config = Cfg()
jieba = pkuseg.pkuseg()
UNK, PAD = '<UNK>', '<PAD>'
label_dict = {'职业发展': 0, '学业方面': 1, '心理方面': 2, '恋爱关系': 3}


def build_vocabulary(file_path, min_freq):
    """
    :param file_path:   file content example '你好 北京 职业发展' 最后一个为label
    :param max_size:   词表的最大大小
    :param min_freq:    词的最小频率
    :return:
    """
    vocab = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(enumerate(f.readlines())):
            # print(line)
            line = line[1].strip().split(' ')[:-1]
            if not line:
                continue
            for word in line:
                vocab[word] = vocab.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)
        vocab = {word[0]: id for id, word in enumerate(vocab_list)}
        vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})
    with open('voab.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)


def view_vocab():
    print(pickle.load(open('./voab.pkl', 'rb')))


def build_datasets():
    vocab = pickle.load(open('./voab.pkl', 'rb'))
    print(vocab)
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(enumerate(f.readlines())):
                words_line = []
                content = line[1].strip().split(' ')[:-1]
                if not content:
                    continue
                label = line[1].strip().split(' ')[-1]
                label = label_dict[label]

                if len(content) < pad_size:
                    content.extend([vocab.get(PAD)] * (pad_size - len(content)))
                else:
                    content = content[:pad_size]
                # word 2 id
                for word in content:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, label))
        return contents

    train = load_dataset('../data/corpus.txt', pad_size=32)
    with open('../data/train.pkl', 'wb') as fp:
        pickle.dump(train, fp)
    return train


def build_test_data(sentence):
    vocab = pickle.load(open(r'C:\Users\king\Documents\code\NLP\text_classification\utils\topic_vocab.pkl', 'rb'))
    words_line = []
    seg_list = jieba.cut(sentence)
    result = remove_stop_words(seg_list)
    result = ' '.join(result)
    if len(result) <= 1:
        print('too short!')
    else:
        print('分词，去掉停用词后====={}'.format(result))
        result = result.split(' ')
        if len(result) < 32:
            result.extend([vocab.get(PAD)] * (32 - len(result)))
        else:
            result = result[:32]
        for word in result:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        return [words_line]


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = [v[0] for v in datas]
        x = torch.LongTensor(x)
        y = torch.from_numpy(np.array([_[1] for _ in datas]))
        # seq_len = torch.tensor([_[2] for _ in datas]).to(self.device)
        return (x, 0), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, batch_size, device):
    iter = DatasetIterater(dataset, batch_size, device)
    return iter


def load_embeddings():
    with open(r'C:\Users\king\Documents\code\NLP\text_classification\data\embeddings.pkl', 'rb') as fp:
        embeddings = pickle.load(fp)
    with open('./voab.pkl', 'rb') as fp:
        vocab = pickle.load(fp)
    embeddings = np.zeros((len(vocab), 100))
    for word, idx in tqdm(vocab.items()):
        try:
            vector = embeddings[word]
        except Exception as e:
            print('{} dont have embedding'.format(word))
            vector = np.zeros(100)
        embeddings[idx] = vector
    with open('../data/topic_embeddings.pkl', 'wb') as fp:
        pickle.dump(embeddings, fp)


def gen_topic_train_data():
    import pandas as pd
    import re
    cop = re.compile("[^\u4e00-\u9fa5^.^,^，^a-z^A-Z^0-9]")
    source_root = r'C:\Users\king\Desktop\兔兔的论文\数据'
    data = []
    print('1.整合数据，并分词, 去除停用词')
    for exl in os.listdir(source_root):
        for topic in label_dict:
            try:
                df = pd.read_excel(os.path.join(source_root, exl), sheet_name=topic)
            except:
                continue

            content = df['content']
            for line in tqdm(content):
                if type(line) == str:
                    data.append((remove_stop_words(jieba.cut(cop.sub('', line.strip()))), label_dict[topic]))

    print('2.构建词汇表')
    vocab = {}
    for sample in tqdm(data):
        seg_sentence = sample[0]
        for word in seg_sentence:
            vocab[word] = vocab.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab.items() if _[1] >= 3], key=lambda x: x[1], reverse=True)
    vocab = {word[0]: id for id, word in enumerate(vocab_list)}
    vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})

    with open(os.path.join(config.word2vec_from_scratch, 'topic_vocab.pkl'), 'wb') as fp:
        pickle.dump(vocab, fp)

    print('3.将语料根据词典转换为数')
    numerical_data = []
    for sample in tqdm(data):
        res = [vocab.get(word, vocab.get(UNK)) for word in sample[0]]
        if len(res) < 32:
            res += [vocab[PAD] for _ in range(32 - len(res))]
        else:
            res = res[:32]
        numerical_data.append((res, sample[1]))
    with open(os.path.join(config.train_data, 'topic_train.pkl'), 'wb') as fb:
        pickle.dump(numerical_data, fb)

    print(' 4.建立词向量表')
    word2vec = pickle.load(
        open(os.path.join(config.word2vec_from_scratch, 'embeddings.pkl'), 'rb')
    )
    weight = np.zeros((len(vocab), 100), dtype=float)
    for word, idx in tqdm(vocab.items()):
        try:
            weight[idx, :] = word2vec[word]
        except:
            weight[idx, :] = np.array([0 for _ in range(100)])
    with open('../data/topic_emb_weights.pkl', 'wb') as fp:
        pickle.dump(weight, fp)


if __name__ == '__main__':
    train_path = '../data/corpus.txt'
    emb_dim = 100

    """
    构建词表
    """
    # build_vocabulary(train_path, min_freq=5)
    # word_to_id = pickle.load(open('voab.pkl', 'rb'))
    # print(word_to_id['说'])
    # embeddings = np.random.rand(len(word_to_id), emb_dim)
    # build_datasets()
    # load_embeddings()
    gen_topic_train_data()
