import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pdb
import sys
import pickle

def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
        X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    
    X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_mean), X_std)
     
    return X, X_mean, X_std

def sk_GradBoost(X_train, Y_train):
    clf_2 = GradientBoostingClassifier(n_estimators=75, max_depth=9, random_state=0)
    clf_2 = clf_2.fit(X_train, Y_train)
    print(clf_2.score(X_train, Y_train))
    with open('sk_gradboost_v8.pickle','wb') as f:
        pickle.dump(clf_2,f)
    return clf_2

def Save(pred, path):
    # f = open('./data/output/best_random_forest_v8.csv', 'w')
    f = open(path,'w')
    f.write('id,label\n')
    for i in range(pred.shape[0]):
        f.write('{},{}\n'.format(i+1, int(pred[i])))
    f.close()

if __name__=='__main__':
    # X_train_fpath = './data/X_train'
    # Y_train_fpath = './data/Y_train'
    # X_test_fpath = './data/X_test'

    # X_train = np.genfromtxt(X_train_fpath, delimiter=',', skip_header=1)
    # Y_train = np.genfromtxt(Y_train_fpath, delimiter=',', skip_header=0)
    # X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)

    col = [0,1,3,4,5]
    # X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)
    # X_test = np.nan_to_num(X_test)
    # X_test, X_test_mean, X_test_std = _normalize_column_normal(X_test, specified_column=col)
    # clf_2 = sk_GradBoost(X_train, Y_train)
    # pred = clf_2.predict(X_test)
    # Save(pred,'./out.csv')
    test_data_path = sys.argv[5]
    output_path = sys.argv[6]

    X_test = np.genfromtxt(test_data_path, delimiter=',', skip_header=1)
    X_test = np.nan_to_num(X_test)
    X_to_predict, X_test_mean, X_test_std=_normalize_column_normal(X_test, specified_column=col)
    with open('sk_gradboost_v8.pickle','rb') as f:
        model2 = pickle.load(f)
        pred = model2.predict(X_to_predict)
        Save(pred,output_path)

