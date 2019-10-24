import numpy as np
import pandas as pd
import math
import sys
import pdb
dim = 106

def load_data():
    x_train = pd.read_csv('X_train')
    x_test = pd.read_csv('X_test')

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv('Y_train', header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)

def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
        X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    
    X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_mean), X_std)
     
    return X, X_mean, X_std

def train(x_train, y_train):
    cnt1 = 0
    cnt2 = 0
    
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    
    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
    return mu1, mu2, share_sigma, cnt1, cnt2

def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(share_sigma)

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    # pdb.set_trace()
    # print('w ',w)
    # print('b ',b)
    # print('z ',z)
    pred = sigmoid(z)
    return pred,w,b

def Predict_Save(y_to_predict, mu1, mu2, shared_sigma, N1, N2, w, b, path):
    z = np.dot(w, y_to_predict.T) + b
    pred = sigmoid(z)
    pred = np.around(pred)
    # pdb.set_trace()
    f = open(path, 'w')
    f.write('id,label\n')
    for i in range(pred.shape[0]):
        f.write('{},{}\n'.format(i+1, int(pred[i])))
    f.close()

if __name__ == '__main__':
    # X_train_fpath = './data/X_train'
    # Y_train_fpath = './data/Y_train'
    # X_test_fpath = './data/X_test'
    X_train_fpath = sys.argv[3]
    Y_train_fpath = sys.argv[4]
    X_test_fpath = sys.argv[5]
    output_path = sys.argv[6]
    
    X_train = np.genfromtxt(X_train_fpath, delimiter=',', skip_header=1)
    Y_train = np.genfromtxt(Y_train_fpath, delimiter=',', skip_header=0)

    col = [0,1,3,4,5]
    mu1, mu2, shared_sigma, N1, N2 = train(X_train, Y_train)
    y,w,b = predict(X_train, mu1, mu2, shared_sigma, N1, N2)
    # print('w ',w)
    # print('b ',b)
    y = np.around(y)
    result = (Y_train == y)
    # print('Train acc = %f' % (float(result.sum()) / result.shape[0]))
    #predict x_test
    X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)
    X_test = np.nan_to_num(X_test)

    # output_path='./generative_v1.csv'
    Predict_Save(X_test, mu1, mu2, shared_sigma, N1, N2, w, b, output_path)
    