import os
import random
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np

from utils.prepare_data import build_iterator
from utils.prepare_data import build_test_data
from utils.lstm_attn_utils import build_lstm_test_data
from utils.lstm_attn_utils import find_best_model
from utils.lstm_attn_utils import reverse_label_dict
# from model.lstm_attn import Config
from model.textCNN import Config
from config import Cfg
from model.textCNN import Model
from model.lstm_attn import AttnModel
from corpus_dataloader import MyTrainValData

# from corpus_dataloader import collate_fn

config = Config()
cfg = Cfg()
label_dict = {0: '职业发展', 1: '学业方面', 2: '心理方面', 3: '恋爱关系'}


def train():
    with open(os.path.join(cfg.train_data, 'topic_train.pkl'), 'rb') as fp:
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

    eval_best_loss = float('inf')
    model = model.float()
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
                model_save_path = os.path.join(cfg.checkpoint['topic'], str(eval_accuracy) + '.ckpt')
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
    data = build_test_data(sentence)
    model = Model(Config())
    model.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint['topic'], 'best.ckpt')))
    model.eval()
    model.float()
    predict = model((torch.LongTensor(data), 0))
    print(label_dict[int(torch.max(predict.data, 1)[1].cpu())])


def lstm_attn_train():
    with open(os.path.join(cfg.train_data, 'senti_trainval.pkl', 'rb')) as fp:
        all_data = pickle.load(fp)
    data_size = len(all_data[0])
    random_idx = list(range(data_size))
    random.shuffle(random_idx)

    for_train, for_val = random_idx[:int(0.9 * data_size)], random_idx[:int(0.9 * data_size):]
    train_data, val_data = [np.array(all_data[0])[for_train], np.array(all_data[1])[for_train]], \
                           [np.array(all_data[0])[for_val], np.array(all_data[1])[for_val]]

    print('=============> prepare data <==============')
    train_data_iter = MyTrainValData(train_data)
    val_data_iter = MyTrainValData(val_data)
    train_data_loader = DataLoader(train_data_iter, batch_size=config.batch_size, shuffle=True, )
    val_data_loader = DataLoader(val_data_iter, batch_size=config.batch_size, shuffle=True, )

    print('=============> build model <==============')
    model = AttnModel()
    print(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_accu = 0
    for epoch in range(config.num_epochs):
        print('Epoch:{:0>2}/{:0>2}'.format(epoch + 1, config.num_epochs))
        model.train()
        for i, sample in enumerate(train_data_loader):
            model.zero_grad()
            label = sample['label']
            pred = model(sample['data'].float())
            loss = F.cross_entropy(pred, label)
            loss.backward()
            optimizer.step()

            if i % config.info_freq == 0:
                target = label.data.cpu()
                predict = torch.max(pred.data, 1)[1].cpu()
                train_accuracy = metrics.accuracy_score(target, predict)
                print('epoch: {}, iter: {} loss: {}==train-accuracy: {}'.format(epoch, i, loss, train_accuracy),
                      end='||')
        val_accu = lstm_attn_val(model, val_data_loader)
        print('\tval_accuracy => {}'.format(val_accu))
        if val_accu > best_val_accu:
            best_val_accu = val_accu
            torch.save(model.state_dict(),
                       os.path.join(cfg.checkpoint['senti'], str(epoch) + '_' + str(val_accu) + '.ckpt'))


def lstm_attn_val(model, data_iter):
    model.eval()
    loss_total = 0

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for i, sample in enumerate(data_iter):
            pred = model(sample['data'].float())
            label = sample['label']
            loss = F.cross_entropy(pred, label)
            loss_total += loss

            predict = torch.max(pred.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc


model = AttnModel()
model.load_state_dict(torch.load(find_best_model()))
model.eval()


def lstm_attn_test(sentence):
    # 1.加载词典，对句子分词，将句子转化为数值的形式
    numeric_data = build_lstm_test_data(sentence=sentence)
    weight, pred = model(torch.FloatTensor(numeric_data).float())
    # print(list(weight.squeeze()).index(1))
    predict = torch.max(pred.data, 1)[1].cpu().numpy()

    return reverse_label_dict[predict[0]]


def label_data():
    from utils.lstm_attn_utils import label_dict
    from tqdm import tqdm
    import pandas as pd

    res = {'心理方面': {'content': [], 'polarity': [], 'intensity': []},
           '学业方面': {'content': [], 'polarity': [], 'intensity': []},
           '职业发展': {'content': [], 'polarity': [], 'intensity': []},
           '恋爱关系': {'content': [], 'polarity': [], 'intensity': []}
           }
    with open(os.path.join(Cfg().crawler_save_path, 'new_result_notopic.txt'), 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f.readlines())):
            content, topic = line.split(' ')[0].strip(), line.split(' ')[1].strip()
            res[topic]['content'].append(content)
            senti = label_dict[lstm_attn_test(content)]
            if senti == 0:
                polarity = 0
                intensity = 0
            elif senti <= 3:
                polarity = 1
                intensity = senti
            else:
                polarity = -1
                intensity = senti - 3
            res[topic]['polarity'].append(polarity)
            res[topic]['intensity'].append(intensity)
        with open(os.path.join(cfg.tmp, './tmp.pkl'), 'wb') as fp:
            pickle.dump(res, fp)

        writer = pd.ExcelWriter(os.path.join(cfg.labeled, '\output.xlsx'))
        for topic in res:
            pd.DataFrame(res[topic]).to_excel(writer, topic)
        writer.save()


if __name__ == '__main__':
    # train()
    # test('好喜欢今天在图书馆遇到的女孩')
    # test('我发现图书馆自习效率最高')
    # lstm_attn_train()
    # print(lstm_attn_test('我觉得你已经很厉害了'))
    label_data()
