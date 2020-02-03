# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.3  # 随机失活
        self.model_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\checkpoints\topic'
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 4  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = 100
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        with open(r'C:\Users\king\Documents\code\NLP\text_classification\data\topic_emb_weights.pkl', 'rb') as fp:
            pretrained_emmbeddings = pickle.load(fp)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_emmbeddings), freeze=False)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # print(x[0])
        out = self.embedding(x[0].long())
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
