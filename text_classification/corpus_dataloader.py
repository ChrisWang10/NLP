from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import pickle
import os
from config import Cfg


class MyTrainValData(Dataset):
    def __init__(self, data):
        super(MyTrainValData, self).__init__()
        self.data = data
        self.embeddings = pickle.load(
            open(os.path.join(Cfg().word2vec_from_scratch, 'emb_weights.pkl'), 'rb'))

    def __len__(self):
        assert len(self.data[0]) == len(self.data[1])
        return len(self.data[0])

    def __getitem__(self, idx):
        ret = {}
        ret['data'] = self.embeddings[self.data[0][idx]]
        ret['label'] = self.data[1][idx]
        return ret

# def collate_fn(data):
#     data.sort(key=lambda x: len(x), reverse=True)
#     data_length = [len(sq) for sq in data]
#     return data, data_length
