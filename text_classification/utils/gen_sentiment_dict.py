import pickle

neg_path = [r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\neg.txt',
            r'C:\Users\king\Desktop\sentiment\neg.txt',
            r'C:\Users\king\Desktop\sentiment\neg1.txt']
pos_path = [r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\pos.txt',
            r'C:\Users\king\Desktop\sentiment\pos.txt',
            r'C:\Users\king\Desktop\sentiment\pos1.txt']
intensity_path = r'C:\Users\king\Desktop\sentiment\intensity.txt'
deny_path = r'C:\Users\king\Desktop\sentiment\deny.txt'
stop_words_path = r'C:\Users\king\Documents\code\NLP\text_classification\stopwords\百度停用词表.txt'

senti_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\ntsu_hownet.txt'
intensity_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\intensity.txt'
negation_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\negation.txt'
stop_save_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\stop.txt'

import os

# ==================================================
# 基础情感词典
# ==================================================
if not os.path.exists(senti_save_path):
    res = {}

    for f_path in neg_path:
        todo = True
        encoding = 'utf-8'
        while todo:
            try:
                with open(f_path, 'r', encoding=encoding) as f:
                    for word in f.readlines():
                        res[word.strip()] = -1
                todo = False
            except:
                encoding = 'gbk' if encoding == 'utf-8' else 'utf-8'

    for f_path in pos_path:
        todo = True
        encoding = 'utf-8'
        try:
            with open(f_path, 'r', encoding=encoding) as f:
                for word in f.readlines():
                    res[word.strip()] = 1
            todo = False
        except:
            encoding = 'gbk' if encoding == 'utf-8' else 'utf-8'

    with open(senti_save_path, 'wb') as fb:
        pickle.dump(res, fb)

# ==================================================
# 程度副词词典
# ==================================================
if not os.path.exists(intensity_save_path):
    res = {}
    todo = True
    encoding = 'utf-8'
    while todo:
        try:
            with open(intensity_path, 'r', encoding=encoding) as f:
                cur = 0
                for line in f.readlines():
                    if cur <= 3:
                        if line[:1].isdigit():
                            cur = int(line[0])
                            continue
                        res[line.strip()] = 4 - cur
                    else:
                        break
            todo = False
        except:
            encoding = 'gbk' if encoding == 'utf-8' else 'utf-8'
    with open(intensity_save_path, 'wb') as fb:
        pickle.dump(res, fb)

# ==================================================
# 否定词词典
# ==================================================
if not os.path.exists(negation_save_path):
    res = {}
    todo = True
    encoding = 'utf-8'
    while todo:
        try:
            with open(deny_path, 'r', encoding=encoding) as f:
                for line in f.readlines():
                    print(line.strip())
                    res[line.strip()] = 1
            todo = False
        except:
            encoding = 'gbk' if encoding == 'utf-8' else 'utf-8'

    with open(negation_save_path, 'wb') as fb:
        pickle.dump(res, fb)

print('Done!')


def test():
    print('Testing...')
    with open(senti_save_path, 'rb') as fp:
        senti_dict = pickle.load(fp)

    with open(intensity_save_path, 'rb') as fp:
        intensity_dict = pickle.load(fp)

    with open(negation_save_path, 'rb') as fp:
        negation_dict = pickle.load(fp)
    print(senti_dict)
    print(intensity_dict)
    print(negation_dict)


if not os.path.exists(stop_save_path):
    res = {}
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            res[line.strip()] = 1
    with open(stop_save_path, 'wb') as fb:
        pickle.dump(res, fb)
print('stop words done!')

for file in pos_path + neg_path:
    encoding = 'utf-8'
    todo = True
    print(file)
    while todo:
        try:
            for line in open(file, 'r', encoding=encoding).readlines():
                print(line.strip())
                if line.strip() == '厉害':
                    print(file)
                    os._exit(0)
            todo = False
        except:
            encoding = 'gbk' if encoding == 'utf-8' else 'utf-8'
