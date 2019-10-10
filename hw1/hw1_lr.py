import numpy as np
import pandas as pd
import sys
def Preprocess(path, train=True):
    data = pd.read_csv(path)
    data.replace("NR",0,inplace=True)
    data.fillna(0,inplace=True)
    data.replace("#",'',inplace=True,regex=True)
    data.replace("x",'',inplace=True,regex=True)
    data.replace("\*",'',inplace=True,regex=True)

    data = data.values # dataframe to array
    data = data[:, 2:] # drop column 0 and 1
    data = data.astype(float) # change element type from str to float
    
    if train==False:
        test_list = []
        for i in range(0,data.shape[0],18):
            test_list.append(data[i:i+18][:,0:9].flatten())
        test_list = np.asarray(test_list)
        return test_list
    x_data, y_data = Extract(data)
    return x_data, y_data

def valid(x, y):
    if y <= 2 or y > 100:
        return False
    for i in range(9):
        if x[9,i] <= 2 or x[9,i] > 100:
            return False
    return True

def Extract(data):
    '''
        return training data (array)
        return labels (array)
    '''
    temp = [[] for i in range(18)]
    for i in range(data.shape[0]):
        index = i%18
        for ele in data[i]: 
            temp[index].append(ele)
    temp = np.asarray(temp) # temp.shape: (18, 8784) 8784=366*24

    x_data = []
    y_data = []
    for i in range(temp.shape[1]-9):
        x = temp[:, i:i+9]  
        y = temp[9,i+9]
        if valid(x,y):
            x = x.flatten()
            x_data.append(x) # extract training data
            y_data.append(y) # extract PM2.5
    x_data = np.asarray(x_data) # x_data.shape: (8775,162) (datanumber, weighting)
    y_data = np.asarray(y_data) # y_data.shpae: (8775,)

    return x_data, y_data

def LR(x_data,y_data):
    num = x_data.shape[1] 
    b = 0.0
    w = np.ones(num)
    lr = 1.0
    epoch = 100000
    b_lr = 0
    w_lr = np.zeros(num)
    for e in range(epoch):
        error = y_data - b - np.dot(x_data, w)
        b_grad = -2*np.sum(error)  
        w_grad = -2*np.dot(error, x_data)
        b_lr = b_lr + b_grad**2
        w_lr = w_lr + w_grad**2
        mse = np.mean(np.square(error))
        b = b - lr/np.sqrt(b_lr)*b_grad
        w = w - lr/np.sqrt(w_lr)*w_grad
        if (e+1)%10000 == 0:
            print('epoch:{}\nloss{}'.format(e+1, mse))
    np.save('weight_v2', w)
    np.save('bias_v2', b)

def Predict_Save(y_to_predict, output_path):
    # f = open('./output_lr/lr_v2.csv', 'w')
    f = open(output_path, 'w')
    f.write('id,value\n')
    w = np.load('weight_v2.npy')
    b = np.load('bias_v2.npy')
    for i in range(y_to_predict.shape[0]):
        ans = np.dot(y_to_predict[i],w) + b
        f.write('id_{},{}\n'.format(i, ans))
    f.close()


if __name__=='__main__':
    # train_data1_path = "../ml2019fall-hw1/year1-data.csv" # 366 Days
    # train_data2_path = "../ml2019fall-hw1/year2-data.csv" # 365 Days
    # test_data_path = "../ml2019fall-hw1/testing_data.csv"
    # x1,y1 = Preprocess(train_data1_path) #(8775,162) (8775,)
    # x2,y2 = Preprocess(train_data2_path) #(8751,162) (8751,)

    # x = np.concatenate((x1, x2), axis = 0)
    # y = np.concatenate((y1, y2), axis = 0)

    test_data_path = sys.argv[1]
    output_path = sys.argv[2]

    y_to_predict = Preprocess(test_data_path,False) #(500, 162)
    # LR(x,y)
    Predict_Save(y_to_predict, output_path)
