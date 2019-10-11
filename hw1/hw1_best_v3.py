import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import sys
import pickle

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

def sk_LinearRegression(x_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    with open('sklear_LR_v3.pickle','wb') as f:
    	pickle.dump(regressor,f)
    # np.save('sklear_LR_v3',regressor)
    return regressor

def Save(pred, path):
    # f = open('./output/best_v3', 'w')
    f = open(path, 'w')
    f.write('id,value\n')
    for i in range(pred.shape[0]):
        f.write('id_{},{}\n'.format(i, pred[i]))
    f.close()

if __name__=='__main__':
	
    # train_data1_path = "../ml2019fall-hw1/year1-data.csv" # 366 Days
    # train_data2_path = "../ml2019fall-hw1/year2-data.csv" # 365 Days
    # x1,y1 = Preprocess(train_data1_path) #(8775,162) (8775,)
    # x2,y2 = Preprocess(train_data2_path) #(8751,162) (8751,)	
    # x = np.concatenate((x1, x2), axis = 0)
    # y = np.concatenate((y1, y2), axis = 0)
    
    # model = sk_LinearRegression(x,y)

    # test_data_path = "../ml2019fall-hw1/testing_data.csv"

    test_data_path = sys.argv[1]
    output_path = sys.argv[2]

    y_to_predict = Preprocess(test_data_path,train=False) #(500, 162)
    with open('sklear_LR_v3.pickle','rb') as f:
    	model2 = pickle.load(f)
    	pred = model2.predict(y_to_predict)
    	Save(pred,output_path)
        #print(pred)
