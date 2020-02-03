# encoding=utf-8
import pickle
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pkuseg
from config import Cfg

config = Cfg()
seg = pkuseg.pkuseg()
cop = re.compile("[^\u4e00-\u9fa5^.^,^，^a-z^A-Z^0-9]")
topic_list = ['恋爱关系', '师生关系', '学业方面', '职业发展', '心理方面']
label_dict = {
    '中性': 0,
    '正向低': 1,
    '正向中': 2,
    '正向高': 3,
    '负向低': 4,
    '负向中': 5,
    '负向高': 6
}
reverse_label_dict = {v: k for k, v in label_dict.items()}
pad_size = 32

target = r'C:\Users\king\Documents\code\NLP\text_classification\data\lstm_train_corpus1.xlsx'
word2vec = pickle.load(open(r'C:\Users\king\Documents\code\NLP\train_word2vec\vec.pkl', 'rb'))


def filter_str(desstr, restr=''):
    # 过滤除中英文及数字以外的其他字符
    res = re.compile("[^\u4e00-\u9fa5-\uFF0C^a-z^A-Z^0-9]")
    return res.sub(restr, desstr)


def data_analysis():
    # 1. 各类别的数据
    # 2. 各类别中极性的数据
    # 3. 所有类别中极性的数据
    df = pd.read_excel(target)
    topics, labels = df['topic'], df['label']
    print('=====================》统计信息《==========================')
    for topic in topic_list:
        topic_count = sum(topics == topic)

        print('{} related  to {}'.format(topic_count, topic))
        topic_related = np.array(list(labels))[topics == topic]
        for label in reverse_label_dict:
            label_count = sum(topic_related == label)
            print('\t{} : {}\n'.format(reverse_label_dict[label], label_count), end=' ')
    print('---------------------------------------------------------')


def prepare_data():
    bbs_labeled = r'C:\Users\king\Desktop\兔兔的论文\数据\1.xlsx'
    douban_labeled = r'C:\Users\king\Desktop\兔兔的论文\数据\2.xlsx'

    if os.path.exists(target):
        print('already exist!')
        return
    corpus = {'content': [], 'topic': [], 'label': []}

    # 1. bbs数据存放于不同的sheet里面
    for t in topic_list:
        try:
            sheet = pd.read_excel(bbs_labeled, sheet_name=t)
        except:
            continue
        content = sheet['内容']
        content_drop_na = [cop.sub('', item.strip()) for item in
                           list(content[[not value for value in content.isnull()]])]
        # print(len(content_drop_na))
        corpus['content'] += content_drop_na
        about = list(pd.Series([t for _ in range(len(content_drop_na))]))
        corpus['topic'] += about
        label = sheet['极性'][:len(content_drop_na)] + \
                pd.Series(['' if value == 0 else value for value in sheet['程度'][:len(content_drop_na)]])

        label = list(pd.Series([label_dict[v.strip().replace('\t', '')] for v in label]))
        corpus['label'] += label

    for t in topic_list:
        try:
            sheet = pd.read_excel(douban_labeled, sheet_name=t)
        except:
            continue
        content = sheet['content']
        content_drop_na = [item.strip() for item in list(content[[not value for value in content.isnull()]])]
        corpus['content'] += content_drop_na

        about = list(pd.Series([t for _ in range(len(content_drop_na))]))
        corpus['topic'] += about

        label = list(pd.Series([0 if v >= 0 else 3 for v in sheet['polarity'][:len(content_drop_na)]]) + \
                     pd.Series([0 if inten == 0 else inten for inten in sheet['intensity'][:len(content_drop_na)]]))

        corpus['label'] += label

    df = pd.DataFrame(corpus)
    df.to_excel(target)
    print('finish!')


"""
1. 合并语料，去掉表情符号，句子分词
2. 根据语料得到词表
2. 确定每一个词汇的索引，对于未知
"""


def pre_process(min_freq):
    corpus = pd.read_excel(target)['content']
    labels = pd.read_excel(target)['label']
    vocab = {}

    label = []
    # 分词，构建语料词典
    for i, line in tqdm(enumerate(corpus)):
        if len(line) == 0:
            continue
        # 只留下中英文，标点和数字
        for seg_words in seg.cut(cop.sub('', line.strip())):
            vocab[seg_words] = vocab.get(seg_words, 0) + 1
        label.append(labels[i])
    vocab_list = sorted([_ for _ in vocab.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)
    print('size of vocabulary {}'.format(len(vocab_list)))

    UNK, PAD = '<UNK>', '<PAD>'
    vocab = {word[0]: idx for idx, word in enumerate(vocab_list)}
    vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})

    print('将词典中的词和词向量关联起来')

    embeddings = np.zeros((len(vocab), 100))
    for word, idx in tqdm(vocab.items()):
        try:
            vector = word2vec[word]
        except Exception:
            print('{} dont have embedding'.format(word))
            vector = np.zeros(100)
        embeddings[idx] = vector
    with open(os.path.join(config.word2vec_from_scratch, '/embeddings.pkl'), 'wb') as fp:
        pickle.dump(embeddings, fp)

    print('存储字典')
    with open(os.path.join(config.word2vec_from_scratch, 'vocab.pkl'), 'wb') as fp:
        pickle.dump(vocab, fp)

    print('将数据转化为数值的形式')
    data = []
    for i, line in tqdm(enumerate(corpus)):
        numeric = []
        for seg_words in seg.cut(cop.sub('', line.strip())):
            numeric.append(vocab.get(seg_words, vocab[UNK]))
        if len(numeric) < pad_size:
            numeric += [vocab[PAD] for _ in range(pad_size - len(numeric))]
        else:
            numeric = numeric[:pad_size]
        try:
            assert len(numeric) == pad_size
        except:
            print(len(numeric), numeric)
            os._exit(0)

        data.append(numeric)
    print(np.array(data).shape)

    with open(os.path.join(config.train_data, 'senti_trainval.pkl'), 'wb') as f:
        pickle.dump([data, label], f)


def gen_embedding():
    vocab = pickle.load(open(os.path.join(config.word2vec_from_scratch, 'vocab.pkl'), 'rb'))
    import torch
    weight = np.zeros((len(vocab), 100), dtype=float)
    for word, idx in vocab.items():
        try:
            weight[idx, :] = word2vec[word]
        except:
            weight[idx, :] = np.array([0 for _ in range(100)])
    with open(os.path.join(config.word2vec_from_scratch, 'emb_weights.pkl'), 'wb') as fp:
        pickle.dump(weight, fp)


def build_lstm_test_data(sentence):
    sentence = sentence
    # print(sentence)
    UNK, PAD = '<UNK>', '<PAD>'
    res = []
    vocab = pickle.load(open(os.path.join(config.word2vec_from_scratch, 'vocab.pkl'), 'rb'))
    embedding = pickle.load(
        open(os.path.join(config.word2vec_from_scratch, 'embeddings.pkl'), 'rb')
    )
    # print('/'.join(seg.cut(cop.sub('', sentence.strip()))))
    for seg_word in seg.cut(cop.sub('', sentence.strip())):
        try:
            res.append(embedding[vocab[seg_word]])
        except:
            res.append(embedding[vocab[UNK]])
    if len(res) < pad_size:
        res += [embedding[vocab[PAD]] for _ in range(pad_size - len(res))]
    else:
        res = res[:pad_size]
    return [res]


def find_best_model():
    best, path = '0', ''
    model_root = config.checkpoint['senti']
    for model_path in os.listdir(model_root):
        accu = model_path.split('_')[-1]
        if accu > best:
            best = accu
            path = os.path.join(model_root, model_path)
    return path


def main():
    data_analysis()
    # prepare_data()
    # data_analysis()
    # pre_process(min_freq=3)
    # gen_embedding()


if __name__ == '__main__':
    main()
