import torch
import sys
import numpy as np
import pandas as pd
from hw3_util import train_hw3,  layer3_CNN
import torch.nn.functional as F
import pdb

'''
python3 hw3_main.py ./data/train_img/ ./data/train.csv train
bash hw3_train.sh <training data folder>  <train.csv>
'''

EPOCH = 20

def train():
    label = pd.read_csv(sys.argv[2],header=0)
    label = np.array(label)
    rng = np.arange(label.shape[0])
    np.random.shuffle(rng)
    train_label = label[rng[:28000]]
    valid_label = label[rng[28000:]]
    train_dataset = train_hw3(sys.argv[1], train_label)
    valid_dataset = train_hw3(sys.argv[1], valid_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = layer3_CNN().to(device)
    else:
        model = layer3_CNN()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCH):
        model.train()
        train_loss = 0
        correct = 0
        for train_batch, (data, label) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            train_loss += F.cross_entropy(output, label).item() 
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
            if train_batch%10 == 0:
                print('Train Epoch: {:0>2d} [{:0>5d}/{:0>5d} ({:.0f}%)]     Loss: {:.6f}\r'.format(
                    epoch, train_batch * len(data), len(train_loader.dataset),
                    100. * train_batch / len(train_loader), loss.item())) 
            if train_batch == len(train_loader)-1:
                print()
                train_loss /= len(train_loader.dataset)
                print('Train set: Average loss: {:.4f}, Accuracy: {:0>4d}/{:0>4d} ({:.0f}%)'.format(
                    train_loss, correct, len(train_loader.dataset),
                    100. * correct / len(train_loader.dataset)))

        model.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in valid_loader:
                if torch.cuda.is_available():
                    data, label = data.to(device), label.to(device)
                output = model(data)
                valid_loss += F.cross_entropy(output, label).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()

        valid_loss /= len(valid_loader.dataset)
        print('valid set: Average loss: {:.4f}, Accuracy: {:0>4d}/{:0>4d} ({:.0f}%)'.format(
            valid_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))
        print()
        #torch.save(model.state_dict(), './model_CNN_{}.pkl'.format(epoch))

if __name__ == '__main__':
    train()