import pdb
import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid as make_grid
from torchvision.utils import save_image as save_image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2
import sys

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
 
        # define: encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)), 
            nn.Conv2d(32, 64, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)),
            nn.Conv2d(64, 128, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)),
            nn.Conv2d(128, 256, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)),
        )
        # define: decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ConvTranspose2d(32, 3, 2, 2),
            nn.Tanh(),
        )
    def forward(self, x):
 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def plot_history(train_loss):
    plt.figure(1)
    plt.plot(train_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.savefig('./pic/loss_history.jpg')
    plt.close()

if __name__ == '__main__':
    trainX_path = sys.argv[1]
    predict_path = sys.argv[2]
    # detect is gpu available.
    use_gpu = torch.cuda.is_available()
 
    autoencoder = Autoencoder()
    
    # load data and normalize to [-1, 1]
    trainX = np.load(trainX_path) # (9000, 32, 32, 3)
    # trainY = np.load('./trainY.npy')
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1 # (9000, 3, 32, 3) normalize to (-1, 1)
    trainX = torch.Tensor(trainX)
    #pdb.set_trace()
    # if use_gpu, send model / data to GPU.
    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()
 
    # Dataloader: train shuffle = True
    train_dataloader = DataLoader(trainX, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(trainX, batch_size=32, shuffle=False)
 
 
    # We set criterion : L1 loss (or Mean Absolute Error, MAE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
 
    plot_loss_train, plot_epoch = [], []

    # Now, we train 20 epochs.
    '''
    for epoch in range(10):
        cumulate_loss = 0
        for x in train_dataloader:
            latent, reconstruct = autoencoder(x)
            loss = criterion(reconstruct, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cumulate_loss = loss.item() * x.shape[0]
            plot_loss_train.append(cumulate_loss)
        print(f'Epoch { "%03d" % epoch }: Loss : { "%.5f" % (cumulate_loss / trainX.shape[0])}')
        #plot_loss_train.append(cumulate_loss)
        #plot_acc_train.append()
        torch.save(autoencoder.state_dict(), './model/model_{}.pkl'.format(epoch))
    plot_history(plot_loss_train)
    '''
    # Collect the latents and stdardize it.
    latents = []
    reconstructs = []
    model = Autoencoder()
    model.load_state_dict(torch.load('./model_9.pkl'))
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    for x in test_dataloader:
        latent, reconstruct = model(x)
        latents.append(latent.cpu().detach().numpy())


    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)

    latents = PCA(n_components=64, whiten=True).fit_transform(latents)
    tsne = TSNE(n_components=2)
    latents = tsne.fit_transform(latents)
    # TSNE_latents = TSNE(verbose=1).fit_transform(latents)

    # num = 0
    # while(num > ):
    result = KMeans(n_clusters = 2, n_init = 10, max_iter = 700, n_jobs = 8).fit(latents).labels_
        # num = np.count_nonzero(result)

    # x_axis = TSNE_latents[:, 0]
    # y_axis = TSNE_latents[:, 1]
    # plt.scatter(x_axis, y_axis, c=trainY.tolist())
    # plt.savefig('./pic/tsne.jpg')

    if np.sum(result[:5]) >= 3:
        result = 1 - result
 
    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(predict_path,index=False)

