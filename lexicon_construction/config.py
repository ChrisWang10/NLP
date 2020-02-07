class Cfg(object):
    def __init__(self):
        self.checkpoint = {
            'topic': r'C:\Users\king\Documents\code\data\checkpoint\topic',
            'senti': r'C:\Users\king\Documents\code\data\checkpoint\senti'
        }

        self.crawler_save_path = r'C:\Users\king\Documents\code\data\crawler'

        self.labeled = r'C:\Users\king\Documents\code\data\labeled'
        self.labeled_by_model = r'C:\Users\king\Documents\code\data\labeled\by_model'

        self.tmp = r'C:\Users\king\Documents\code\data\tmp'

        self.train_data = r'C:\Users\king\Documents\code\data\train_data'

        self.word2vec_from_scratch = r'C:\Users\king\Documents\code\data\word2vec'
