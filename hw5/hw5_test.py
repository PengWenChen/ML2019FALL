import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import re
import torch.nn.init as init
import pickle
from gensim.models import Word2Vec
import sys
import pdb
import spacy
from util import RNN, WordDataset, weight_init, text_preprocess, find_max_len, padding
import spacy


def test_func(test_index, matrix, output_path):
    dummy_col = np.random.uniform(size = matrix.shape[1]).reshape(1, -1)
    dummy_index = matrix.shape[0]
    matrix = np.vstack((matrix, dummy_col))

    input_size = matrix.shape[0]
    batch_size = 64
    hidden_size = 256
    RNN_model = RNN(input_size, hidden_size, batch_size).cuda()
    RNN_model.load_state_dict(torch.load('./model/modelnew_4.pkl'))
    RNN_model.eval()

    test_dataset = WordDataset(data=test_index, train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    f = open(output_path, "w")
    f.write('id,label\n')
    ans = []
    for i, sample in enumerate(test_loader):
        x = sample['data'].cuda()
        output_label = RNN_model(x)
        preds = (output_label>0.5).cpu().numpy()
        for i in range(len(preds)):
            ans.append(preds[i])

    for j in range(len(ans)):
        if ans[j]:
            f.write(str(j)+','+str(1)+'\n')
        else:
            f.write(str(j)+','+str(0)+'\n')

    f.close()
    print('finish')

if __name__=='__main__':
    test_path = sys.argv[1]
    output_path = sys.argv[2]

    test = open(test_path, 'r')
    test = text_preprocess(test)
    max_test_len = find_max_len(test)
    test = padding(test, max_test_len)

    model = Word2Vec.load("word2vec_new.model")
    n = model.wv.vectors.shape[0]
    matrix = model.wv.vectors

    test_index = []
    for i in range(len(test)):
        sentence = test[i]
        tmp = []
        for word in sentence:
            try:
                tmp.append(model.wv.vocab[word].index)
            except:
                tmp.append(n)
        test_index.append(tmp)
    test_index = np.array(test_index)

    test_func(test_index, matrix, output_path)
