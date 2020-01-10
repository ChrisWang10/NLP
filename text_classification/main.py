import os
import pickle
import torch
from utils.prepare_data import build_iterator
from utils.prepare_data import build_datasets
from textCNN import Config
from textCNN import Model


config = Config()


def train():
    with open('./data/train.pkl', 'rb') as fp:
        train_data = pickle.load(fp)
    train_iter = build_iterator(train_data, 32, 'cpu')

    model = Model(Config())
    print(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            print(trains, labels)
            os._exit(0)



if __name__ == '__main__':
    train()
