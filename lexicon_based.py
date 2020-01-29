import os
import re
import jieba
import pandas as pd
from tqdm import tqdm

save_path = r'C:\Users\king\Documents\code\NLP\result'
txt_path = r'C:\Users\king\Documents\code\DataMining\tools\result'


class DictBasedSentAnal:
    def __init__(self):
        self.__root_dir = './tools/'
        self.__sent_dict__ = self.__read_dict(self.__root_dir + 'sentiment_score.txt')

    @staticmethod
    def __read_dict(path, encoding='utf-8'):
        sent_dict = {}
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                array = re.split('\s+', line.strip())
                if len(array) == 2:
                    sent_dict[array[0]] = float(array[1])
        return sent_dict

    def analyse(self, sentence):
        score = 0.0
        for words in jieba.cut(sentence):
            score += self.__sent_dict__.get(words, 0)
        return score


def get_result(sentAnal):
    topics = {'职业发展': ['myjob', 'Dilbert521'], '恋爱关系': ['fanjianlove', '151793'],
              '学业方面': ['626766', 'hateschool'], '师生关系': ['evilteacher', 'BADTEACHER']}
    writer = pd.ExcelWriter(os.path.join(save_path, 'result.xlsx'))
    for i, topic in enumerate(topics):

        data = {'content': [], 'polarity': [], 'intensity': []}
        for item in topics[topic]:
            f = open(os.path.join(txt_path, item + '.txt'), 'r')
            for line in tqdm(f.readlines()):
                data['content'].append(line)
                score = sentAnal.analyse(line)
                polarity = int(score / abs(score)) if score != 0 else 0
                intensity = 0 if score == 0 else 1 if abs(score) <= 1 else 2 if abs(score) <= 3 else 3
                # print(score, polarity, intensity)
                data['polarity'].append(polarity)
                data['intensity'].append(intensity)
        df = pd.DataFrame(data=data)
        df.to_excel(excel_writer=writer, sheet_name=topic)
    writer.save()
    writer.close()


if __name__ == '__main__':
    sentAnal = DictBasedSentAnal()
    # print('情感得分\t' + '%.2f' % sentAnal.analyse(''))
    get_result(sentAnal)
