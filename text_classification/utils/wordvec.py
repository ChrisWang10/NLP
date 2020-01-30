from gensim.models import KeyedVectors

pre_trained_model = r'C:\Users\king\Downloads\sgns.baidubaike.bigram-char\sgns.baidubaike.bigram-char'

model = KeyedVectors.load_word2vec_format(pre_trained_model)
print(model.wv['你好'].shape)