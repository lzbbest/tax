# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re,sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn import preprocessing



class feature(object):
    def __init__(self,path1,path2):
        self.data = pd.read_csv(path1).ix[:,1:]
        self.label = pd.read_csv(path2)['FLGLCD']
        self.x_train,self.x_test,self.y_train,self.y_test = self.split(0.3)

    def split(self,ratio=0.3): # 一般会把train和test集放在一起做标准化
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.label, test_size=ratio, random_state=0)
        return x_train, x_test, y_train, y_test


    def standard(self,f_list): # standardization, Min-Max, Normalization
        # f_list = ['QNT','SL_TSE','USD_AMT','ZCZJ','SL_NUM']
        #scaler = preprocessing.StandardScaler().fit(train[['QNT','USD_AMT']])
        scaler = preprocessing.StandardScaler().fit(self.x_train[f_list])
        self.x_train[f_list] = scaler.transform(self.x_train[f_list])
        self.x_test[f_list] = scaler.transform(self.x_test[f_list])

    def minmax(self,f_list):# f_list=['Gsdj_year','Sbck_year','First_months']
        min_max_scaler = preprocessing.MinMaxScaler().fit(self.x_train[f_list])
        self.x_train[f_list] = min_max_scaler.transform(self.x_train[f_list])
        self.x_test[f_list] = min_max_scaler.transform(self.x_test[f_list])

    def labelcode(self,f_list): # f_list=['Address']
        le = preprocessing.LabelEncoder().fit(self.x_train[f_list])
        self.x_train[f_list] = le.transform(self.x_train[f_list])
        self.x_test[f_list] = le.transform(self.x_test[f_list])


    def onehot(self,f_list): #f_list=['ZCLXCODE','HYCODE','SWCODE','Address']
        enc = preprocessing.OneHotEncoder().fit(self.x_train[f_list])
        self.x_train[f_list] = enc.transform(self.x_train[f_list])
        self.x_test[f_list] = enc.transform(self.x_test[f_list])

    def run(self):
        self.labelcode(['Address'])
        print(11111)
        #self.onehot(['ZCLXCODE','HYCODE','SWCODE','Address'])
        print(22222)
        self.minmax(['Gsdj_year','Sbck_year','First_months'])
        print(33333)
        self.standard(['QNT','SL_TSE','USD_AMT','ZCZJ','SL_NUM'])

if __name__ == '__main__':
    print('Hello')
    one = feature('../data/X.csv','../data/Y.csv')
    one.run()
    x_train, x_test, y_train, y_test = one.x_train, one.x_test, one.y_train, one.y_test



