import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import cv2
import numpy as np
import pdb

class train_hw3(Dataset):
    def __init__(self, data_dir, label):
        self.data_dir = data_dir
        self.label = label
    def __getitem__(self, index):
        pic_file = '{:0>5d}.jpg'.format(self.label[index][0])
        img = cv2.imread(os.path.join(self.data_dir, pic_file), cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)
        return torch.FloatTensor(img), self.label[index, 1]
    def __len__(self):
        return self.label.shape[0]

class test_hw3(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.num_pics = 0
        for file in os.listdir(self.data_dir):
            if file[-4:] == '.jpg':
                self.num_pics = self.num_pics + 1
    def __getitem__(self, index):
        pic_file = '{:0>4d}.jpg'.format(index)
        img = cv2.imread(os.path.join(self.data_dir, pic_file), cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)
        return torch.FloatTensor(img), index
    def __len__(self):
        return self.num_pics

class layer3_CNN(nn.Module):
    def __init__(self):
        super(layer3_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            )
        self.fc = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 7)
            )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #pdb.set_trace()
        x=x.view(x.size(0),-1)
        out = self.fc(x)
        return out
