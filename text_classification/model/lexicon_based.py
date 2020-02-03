# coding=utf-8
import os
import re
import pkuseg

seg = pkuseg.pkuseg()
import pandas as pd
from tqdm import tqdm
import math

save_path = r'C:\Users\king\Documents\code\NLP\result'
txt_path = r'C:\Users\king\Documents\code\DataMining\tools\result'
senti_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\ntsu_hownet.txt'
intensity_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\intensity.txt'
negation_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\negation.txt'
stop_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\stop.txt'


class DictBasedSentAnal:
    def __init__(self):
        import pickle
        self.__root_dir = r'C:\Users\king\Documents\code\NLP\tools'
        self.__ntsu = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\ntsu_hownet.txt'
        self.__stop_words = self.get_stop_words()
        self.__intensity_words = pickle.load(
            open(intensity_save_path, 'rb')
        )
        self.__negation_words = pickle.load(
            open(negation_save_path, 'rb')
        )
        self.__sent_dict__ = pickle.load(open(self.__ntsu, 'rb'))
        self.__sent_dict__['厉害'] = 1
        # self.__sent_dict__ = self.__read_dict(self.__root_dir + '\sentiment_score.txt')

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
        with open(stop_save_path, 'rb') as fp:
            stop_words_dict = pickle.load(fp)
            return stop_words_dict

    def analyse(self, sentence):
        # print(sentence)
        p_score, p_count, n_score, n_count = 0, 0, 0, 0
        final = 0
        intensity, reverse = 1, 0
        # print('/'.join(seg.cut(sentence)))
        for words in seg.cut(sentence):
            # 1. 首先获取程度词，如果是的话，用于作为权重乘以下一个情感词
            if self.__intensity_words.get(words, 0):
                # print(words, '(增强)', end=' ')
                intensity = self.__intensity_words.get(words) * 0.8
                continue

            # 2. 判断当前词是否是否定词
            if self.__negation_words.get(words, 0):
                # print(words, '(否定)', end=' ')
                reverse = self.__negation_words.get(words) * -1
                final -= 0.5
                continue
            if self.__stop_words.get(words, 0):
                continue

            # 3. 获取情感词
            if self.__sent_dict__.get(words, 0):

                senti_value = self.__sent_dict__.get(words, 0)
                # print(words, senti_value, end=' ')
                if reverse:
                    senti_value = max(abs(senti_value) / 2, 0.2) if senti_value < 0 else \
                        min(-senti_value / 2, -0.2)
                if intensity:

                    final += senti_value * intensity
                    intensity = 1
                else:
                    final += senti_value

                if senti_value > 0:
                    p_count += 1
                    p_score += senti_value * intensity
                if senti_value < 0:
                    n_count += 1
                    n_score += senti_value * intensity

        # 如果句子极性是由正负向情感词决定的话，进行修正
        # a. 如果正向均值大于负向均值，就用正向-负向/正向个数
        # b. 反之
        if n_count > 0 and p_count > 0:
            ap, an = p_score / p_count, n_score / n_count
            final = ap + an
            if final == 0:
                if p_count > n_count:
                    res = p_score / (p_count + n_count)
                elif p_count < n_count:
                    res = n_score / (p_count + n_count)
                else:
                    res = 0
                return res
            return final / n_count if final < 0 else final / p_count
        else:
            return final


def get_result(sentAnal):
    topics = {'职业发展': ['myjob', 'Dilbert521'], '恋爱关系': ['fanjianlove', '151793'],
              '学业方面': ['626766', 'hateschool'], '师生关系': ['evilteacher', 'BADTEACHER']}
    writer = pd.ExcelWriter(os.path.join(save_path, 'result1.xlsx'))
    for i, topic in enumerate(topics):
        data = {'content': [], 'polarity': [], 'intensity': []}
        for item in topics[topic]:
            f = open(os.path.join(txt_path, item + '.txt'), 'r')
            for line in tqdm(f.readlines()):
                data['content'].append(line)
                score = sentAnal.analyse(line)
                polarity = int(score / abs(score)) if score != 0 else 0
                intensity = 0 if score == 0 else 1 if abs(score) <= 1 else 2 if abs(score) <= 2 else 3
                # print(score, polarity, intensity)
                data['polarity'].append(polarity)
                data['intensity'].append(intensity)
        df = pd.DataFrame(data=data)
        df.to_excel(excel_writer=writer, sheet_name=topic)
    writer.save()
    writer.close()


def get_excel_result(sentAnal):
    topics = ['职业发展', '恋爱关系',
              '心理方面']
    source_excel = r'C:\Users\king\Documents\code\DataMining\tools\crawler\result_expand.xlsx'

    writer = pd.ExcelWriter(os.path.join(save_path, 'result3.xlsx'))
    for t in topics:
        df = pd.read_excel(source_excel, sheet_name=t)
        sheet = df['content']
        data = {'content': [], 'polarity': [], 'intensity': []}
        for line in tqdm(sheet):
            if type(line) != str:
                continue
            data['content'].append(line.strip())
            score = sentAnal.analyse(line.strip())
            polarity = int(score / abs(score)) if score != 0 else 0
            intensity = 0 if score == 0 else 1 if abs(score) <= 1 else 2 if abs(score) <= 2 else 3
            data['polarity'].append(polarity)
            data['intensity'].append(intensity)
        df = pd.DataFrame(data=data)
        df.to_excel(excel_writer=writer, sheet_name=t)
    writer.save()
    writer.close()


if __name__ == '__main__':
    sentAnal = DictBasedSentAnal()
    # 他又体贴，又温柔，活泼开朗，就是有一点点矮
    # print('情感得分\t' + '%.2f' % sentAnal.analyse(
    #     '北京，30，主业战略，30万副业还能多俩零吧'))
    # get_result(sentAnal)
    get_excel_result(sentAnal)
