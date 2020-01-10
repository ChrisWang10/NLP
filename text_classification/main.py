import os
import random
import pickle
import torch
import torch.nn.functional as F
from sklearn import metrics
import numpy as np

from utils.prepare_data import build_iterator
from utils.prepare_data import build_datasets
from utils.prepare_data import build_test_data
from textCNN import Config
from textCNN import Model

config = Config()


def train():
    with open('./data/train.pkl', 'rb') as fp:
        train_data = pickle.load(fp)
    random.shuffle(train_data)
    length = len(train_data)
    test_data = train_data[:int(0.3 * length)]
    train_data = train_data[int(0.3 * length):]
    train_iter = build_iterator(train_data, config.batch_size, 'cpu')
    test_iter = build_iterator(test_data, config.batch_size, 'cpu')

    model = Model(Config())
    print(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    eval_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            predict = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(predict, labels.long())
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                target = labels.data.cpu()
                predict = torch.max(predict.data, 1)[1].cpu()
                train_accuracy = metrics.accuracy_score(target, predict)
                print('epoch: {}, iter: {} loss: {}==train-accuracy: {}'.format(epoch, i, loss, train_accuracy),
                      end='||')
                eval_accuracy, eval_loss = evaluate(model, test_iter)
                print('test_accuracy: {} test_loss {}'.format(eval_accuracy, eval_loss))
                model_save_path = os.path.join(config.model_save_path, str(eval_accuracy) + '.ckpt')
                if eval_loss < eval_best_loss:
                    eval_best_loss = eval_loss
                    torch.save(model.state_dict(), model_save_path)


def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels.long())
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), confusion
    return acc, loss_total / len(data_iter)


def test(sentence):
    print('要预测的句子是：{}'.format(sentence))
    label_dict = {0: '职业发展', 1: '学业', 2: '心理方面', 3: '恋爱关系'}
    data = build_test_data(sentence)
    model = Model(Config())
    model.load_state_dict(torch.load(
        r'C:\Users\v_wangchao3\code\MyProject\MyTask\text_classification\checkpoint\0.7507057320298339.ckpt'))
    model.eval()
    predict = model((torch.LongTensor(data), 0))
    print(label_dict[int(torch.max(predict.data, 1)[1].cpu())])


if __name__ == '__main__':
    # train()
    # test('好喜欢今天在图书馆遇到的女孩')
    test('北京大学团委学生会招新')
