import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import sys
import numpy as np
import spacy
import pdb

def text_preprocess(data):
    corpus = []
    nlp = spacy.load('en_core_web_sm')
    tokenizer = spacy.lang.en.English().Defaults().create_tokenizer(nlp)
    for raw_line in data.readlines()[1:]:
        pos = min([raw_line.find(',')])
        sentence = raw_line[pos+1:].replace("@user", "")
        tokens = tokenizer(sentence)
        tmp = []
        for tok in tokens:
            if not tok.is_stop:
                tmp.append(tok.lemma_)
        corpus.append(tmp)
    # pdb.set_trace()
    return corpus

def find_max_len(data):
    list_len = [len(i) for i in data]
    return max(list_len)

def padding(data, max_len):
    for i in range(len(data)):
        while len(data[i]) < max_len:
            data[i].append('<pad>')
            # pdb.set_trace()
    return data


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = 1
        self.embedding = nn.Embedding(input_size, 300)

        self.gru = nn.GRU(300, hidden_size,
                           num_layers=self.num_layers,
                           bidirectional=True, batch_first=True, dropout=0.5)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 1*2, 1),
            nn.Sigmoid()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        embed_input = self.embedding(input.long())
        output, (hidden) = self.gru(embed_input)
        a,b = hidden[0], hidden[1]
        hidden = torch.cat((a,b),1)
        label = (self.fc1(hidden))
        return label.squeeze()

    def init_embedding(self, matrix):
        self.embedding.weight = matrix
        print(self.embedding.weight.requires_grad)


class WordDataset(Dataset):
    def __init__(self, data, train=True, label=None):
        self.data = torch.Tensor(data)
        if train:
            self.label = torch.from_numpy(label).float()
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.train:
            sample = {'data': self.data[idx],
                      'label': self.label[idx]}
        else:
            sample = {'data': self.data[idx]}

        return sample

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
