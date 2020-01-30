import os
import re
import jieba
import pandas as pd
from tqdm import tqdm
import math

save_path = r'C:\Users\king\Documents\code\NLP\result'
txt_path = r'C:\Users\king\Documents\code\DataMining\tools\result'
senti_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\ntsu_hownet.txt'
intensity_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\intensity.txt'
negation_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\negation.txt'


class DictBasedSentAnal:
    def __init__(self):
        import pickle
        # self.__root_dir = './tools/'
        self.__ntsu = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\ntsu.txt'
        self.__stop_words = self.get_stop_words()
        self.__intensity_words = pickle.load(
            open(intensity_save_path, 'rb')
        )
        self.__negation_words = pickle.load(
            open(negation_save_path, 'rb')
        )
        self.__sent_dict__ = pickle.load(open(self.__ntsu, 'rb'))

        # self.__sent_dict__ = self.__read_dict(self.__root_dir + 'sentiment_score.txt')

    @staticmethod
    def __read_dict(path, encoding='utf-8'):
        sent_dict = {}
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                array = re.split('\s+', line.strip())
                if len(array) == 2:
                    sent_dict[array[0]] = float(array[1])
        return sent_dict

    @staticmethod
    def get_stop_words():
        import pickle
        with open(r'C:\Users\king\Documents\code\NLP\text_classification\utils\stop_words.txt', 'rb') as fp:
            stop_words_dict = pickle.load(fp)
            return stop_words_dict

    def analyse(self, sentence):
        p_score, p_count, n_score, n_count = 0, 0, 0, 0
        pre = None
        for words in jieba.cut(sentence):
            if self.__stop_words.get(words, 0):
                continue

            senti_value = self.__sent_dict__.get(words, 0)
            if senti_value < 0:
                if pre and




            senti_value = self.__sent_dict__.get(words, 0)
            print(words, senti_value, end=' ')
            if senti_value < 0:
                n_score += senti_value
                n_count += 1
            else:
                p_score += senti_value
                p_count += 1
        print(n_score/n_count, p_score/p_count)
        final = p_score/p_count * (1/(2-math.log(3.5*p_count))) + \
                n_score/n_count * (1/(2-math.log(3.5*n_count)))
        return final


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
    print('情感得分\t' + '%.2f' % sentAnal.analyse('他们都是高收入，高文凭，生活的开开心心，感觉我就像一个废柴'))
    # get_result(sentAnal)
