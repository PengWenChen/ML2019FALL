import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import torch.nn.init as init
import pickle
from gensim.models import Word2Vec
import sys
import pdb
import spacy
from util import RNN, WordDataset, weight_init ,text_preprocess, find_max_len, padding
import spacy
#import matplotlib.pyplot as plt

def plot_history(train_loss, valid_loss, acc_train, acc_valid):
    plt.figure(1)
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig('./pic/loss_history.jpg')
    plt.close()
    # plt.show()

    plt.figure(2)
    plt.plot(acc_train)
    plt.plot(acc_valid)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='upper left')
    plt.savefig('./pic/accuracy_history.jpg')
    plt.close()

def train(matrix, train_index, valid_index, train_label, valid_label):
    dummy_col = np.random.uniform(size = matrix.shape[1]).reshape(1, -1)
    dummy_index = matrix.shape[0]
    matrix = np.vstack((matrix, dummy_col))

    input_size = matrix.shape[0]
    batch_size = 64
    n_epochs = 4
    print_every = 1
    hidden_size = 256
    lr = 0.001
    train_dataset = WordDataset(data=train_index, label=train_label, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = WordDataset(data=valid_index, label=valid_label, train=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model = RNN(input_size, hidden_size, batch_size).cuda()
    model.apply(weight_init)
    model.init_embedding(torch.nn.Parameter(torch.Tensor(matrix).cuda(), requires_grad = True))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.BCELoss()

    model.train()
    ltrain_loss = []
    lvalid_loss = []
    ltrain_acc = []
    lvalid_acc = []
    print('start train')
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for i, sample in enumerate(train_loader):
            x = sample['data'].cuda()
            label = sample['label'].cuda()
            
            optimizer.zero_grad()
            output_label = model(x)
            loss = criterion(output_label, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

            preds = (output_label>0.5).float()
            epoch_acc += torch.sum(preds == label)

        if epoch % print_every == 0:
            model.eval()
            with torch.no_grad():
                valid_acc = 0
                valid_loss = 0
                for i, sample in enumerate(valid_loader):
                    x = sample['data'].cuda()
                    label = sample['label'].cuda()
                    optimizer.zero_grad()
                    output_label = model(x)
                    loss = criterion(output_label, label)
                    #_, preds = torch.max(output_label.data, 1)
                    valid_loss += criterion(output_label, label)
                    preds = (output_label>0.5).float()
                    valid_acc += torch.sum(preds == label)

            print('[ (%d %d%%), Loss:  %.3f, train_Acc: %.5f, valid_Loss: %.3f, valid_Acc: %.5f]' %
                  (
                   epoch,
                   epoch / n_epochs * 100,
                   epoch_loss/len(train_loader),
                   float(epoch_acc) / len(train_loader) / batch_size,
                   valid_loss/len(valid_loader),
                   float(valid_acc) / len(valid_loader) / batch_size))
            ltrain_loss.append(epoch_loss/len(train_loader))
            lvalid_loss.append(valid_loss/len(valid_loader))
            ltrain_acc.append(float(epoch_acc) / len(train_loader) / batch_size)
            lvalid_acc.append(float(valid_acc) / len(valid_loader) / batch_size)
            epoch_loss = epoch_acc = 0
        torch.save(model.state_dict(), './model/modelnew_{}.pkl'.format(epoch))
    #plot_history(ltrain_loss, lvalid_loss, ltrain_acc, lvalid_acc)
    return model

if __name__ == '__main__':
    train_data = open('./data/train_x.csv', 'r')
    train_label = pd.read_csv('./data/train_y.csv')
    train_label = train_label.iloc[:,1]
    train_label = np.asarray(train_label)

    train_data = text_preprocess(train_data)
    max_len = find_max_len(train_data) #62
    train_data = padding(train_data, max_len)


    # read testing data and train a Word2Vec embedding
    test_data = open('./data/test_x.csv', 'r')
    test_data = text_preprocess(test_data)
    max_test_len = find_max_len(test_data)
    test = padding(test_data, max_test_len)
    '''
    train_data = train_data + test_data
    model = Word2Vec(train_data, size=300, window=5, min_count=5, workers=4)
    model.save("word2vec_new.model")
    pdb.set_trace()
    '''
    # Load trained word2vec model
    model = Word2Vec.load("word2vec_new.model")
    n = model.wv.vectors.shape[0] # Take the last number of the bag (of words) to save 'unknown words'.

    # Save every words' index
    train_index = []
    for i in range(len(train_data)):
        sentence = train_data[i]
        tmp = []
        for word in sentence:
            try:
                tmp.append(model.wv.vocab[word].index) # If the word exits in the bag, append the word's index
            except:
                tmp.append(n) # If the word is not in the bag, append the word into the index representing 'unknown'.
        train_index.append(tmp)
    train_index = np.array(train_index)

    matrix = model.wv.vectors
    train_i = train_index[0:10000]
    valid_i = train_index[10000:]
    train_l = train_label[0:10000]
    valid_l = train_label[10000:]
    RNN_model = train(matrix, train_i, valid_i, train_l, valid_l)
