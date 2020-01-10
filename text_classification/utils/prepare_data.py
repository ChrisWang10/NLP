import torch.nn as nn
import torch
import random
from xmlrpc.client import ServerProxy
import numpy as np
import jieba
import os
from tqdm import tqdm
import pickle
from .data_formater import remove_stop_words
from .data_formater import stop_words_dict

server = ServerProxy("http://172.20.7.96:10243")
# os._exit(0)

UNK, PAD = '<UNK>', '<PAD>'
label_dict = {'职业发展': 0, '学业': 1, '心理方面': 2, '恋爱关系': 3}


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
                # print(np.fromstring(server.get_vector(word).data))
                # os._exit(0)
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
    vocab = pickle.load(open(r'C:\Users\v_wangchao3\code\MyProject\MyTask\text_classification\utils\voab.pkl', 'rb'))
    words_line = []
    seg_list = jieba.cut(sentence)
    result = remove_stop_words(seg_list)
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
    with open('./voab.pkl', 'rb') as fp:
        vocab = pickle.load(fp)
    embeddings = np.zeros((len(vocab), 100))
    for word, idx in tqdm(vocab.items()):
        try:
            vector = np.fromstring(server.get_vector(word).data)
        except Exception as e:
            print('{} dont have embedding'.format(word))
            vector = np.random.uniform(-1, 1, 100)
        embeddings[idx] = vector
    with open('../data/embeddings.pl', 'wb') as fp:
        pickle.dump(embeddings, fp)


if __name__ == '__main__':
    train_path = '../data/corpus.txt'
    emb_dim = 300

    """
    构建词表
    """
    # build_vocabulary(train_path, min_freq=5)
    # word_to_id = pickle.load(open('voab.pkl', 'rb'))
    # print(word_to_id['说'])
    # embeddings = np.random.rand(len(word_to_id), emb_dim)
    build_datasets()
    # load_embeddings()
