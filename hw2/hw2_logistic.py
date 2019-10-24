import numpy as np
import pandas as pd
import sys

def load_data():
    #讀檔如果像這樣把路徑寫死交到github上去會馬上死去喔
    #還不知道怎寫請參考上面的連結
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

def normalize(x_train, x_test):
    
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor

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
	b = 0.0
	w = np.zeros(x_train.shape[1])
	lr = 0.05
	epoch = 1000
	b_lr = 0
	w_lr = np.ones(x_train.shape[1])

	for e in range(epoch):
	    z = np.dot(x_train, w) + b
	    pred = sigmoid(z)
	    loss = y_train - pred

	    b_grad = -1*np.sum(loss)
	    w_grad = -1*np.dot(loss, x_train)

	    b_lr += b_grad**2
	    w_lr += w_grad**2


	    b = b-lr/np.sqrt(b_lr)*b_grad
	    w = w-lr/np.sqrt(w_lr)*w_grad

	    if(e+1)%500 == 0:
	        loss = -1*np.mean(y_train*np.log(pred+1e-100) + (1-y_train)*np.log(1-pred+1e-100))
	        print('epoch:{}\nloss:{}\n'.format(e+1,loss))
	return w, b

def Predict_Save(y_to_predict, path):
    f = open(path, 'w')
    f.write('id,label\n')
    w = np.load('logistic_w.npy')
    b = np.load('logistic_b.npy')
    for i in range(y_to_predict.shape[0]):
        ans = np.dot(y_to_predict[i],w) + b
        ans = sigmoid(ans)
        if ans>0.5:
            f.write('{},{}\n'.format(i+1, 1))
        else:
            f.write('{},{}\n'.format(i+1, 0))

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
    X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)

    # x_train, y_train, x_test = load_data()
    # x_train, x_test = normalize(x_train, x_test)
    w, b = train(X_train, Y_train)
    
    np.save('logistic_w.npy',w)
    np.save('logistic_b.npy',b)

    X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)
    X_test = np.nan_to_num(X_test)
    y_to_predict, X_mean, X_std = _normalize_column_normal(X_test, specified_column=col)
    # output_path='./logistic_v1.csv'
    Predict_Save(y_to_predict, output_path)
    #predict x_test