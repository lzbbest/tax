# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing

class feature(object):
    def __init__(self,idd,data,label):
        self.id = idd
        self.data = data
        self.label = label
        #self.x_train,self.x_test,self.y_train,self.y_test = self.split(0.3)

    def prepro(self):
        # 'Sbck_year', 'day_diff','TK_num', 'TK_sum','TK_average'
        self.data['SL_TSE'] = self.data['SL_TSE'].apply(lambda x:np.nan if x==0 else x)
        self.data['USD_AMT'] = self.data['USD_AMT'].apply(lambda x:np.nan if x==0 else x)
        self.data['Gsdj_year'] = self.data['Gsdj_year'].apply(lambda x:np.nan if x==0 else x)
        self.data['Sbck_year'] = self.data['Sbck_year'].apply(lambda x:np.nan if x==0 else x)
        self.data['ZCZJ'] = self.data['ZCZJ'].apply(lambda x:np.nan if x==0 else x)
        self.data['day_diff'] = self.data['day_diff'].apply(lambda x:np.nan if x==0 else x)

    def split(self,ratio=0.25): # 一般会把train和test集放在一起做标准化
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.label, test_size=ratio, random_state=3)
        return x_train, x_test, y_train, y_test


    def standard(self,f_list): # standardization, Min-Max, Normalization
        # f_list = ['QNT','SL_TSE','USD_AMT','ZCZJ','SL_NUM']
        #scaler = preprocessing.StandardScaler().fit(train[['QNT','USD_AMT']])
        scaler = preprocessing.StandardScaler().fit(self.data[f_list])
        self.data[f_list] = scaler.transform(self.data[f_list])

    def standard_nan(self,f_list):
        self.data[f_list]=(self.data[f_list]-np.mean(self.data[f_list],axis=0))/np.std(self.data[f_list])
         
    def minmax(self,f_list):# f_list=['Gsdj_year','Sbck_year','First_months']
        min_max_scaler = preprocessing.MinMaxScaler().fit(self.data[f_list])
        self.data[f_list] = min_max_scaler.transform(self.data[f_list])

    def labelcode(self,f_list): # f_list=['Address']
        for fea in f_list:
            print(fea)
            le = preprocessing.LabelEncoder().fit(self.data[fea])
            self.data[fea] = le.transform(self.data[fea])
            #preprocessing.LabelEncoder().fit_transform(self.data[fea])

    def onehot(self,f_list):
        for fea in f_list:
            enc = preprocessing.OneHotEncoder().fit(self.data[f_list])
            self.data[f_list] = enc.transform(self.data[f_list])

    def run(self):
        self.prepro()
        self.labelcode(['FLGLCD','ZCLXCODE','HYCODE','NSRLB','QYLX'])
        print('Label encoding completed')
        #self.onehot(['FLGLCD','ZCLXCODE','HYCODE','NSRLB','QYLX'])
        #print('One-Hot encoding completed')
        self.minmax(['First_months','SB_NUM','SB_YM'])
        self.standard(['QNT','SL_NUM'])
        self.standard_nan(['SL_TSE','USD_AMT','Gsdj_year','Sbck_year','ZCZJ',
                           'day_diff','TK_num','TK_sum','TK_average'])
        print('Normalization completed')



'''
if __name__ == '__main__':
    data = pd.read_csv('../data/data.csv')
    ID,mat,label = data['CPCODE'],data.iloc[:,1:-1],data['label']
    one = feature(ID,mat,label)
    one.run()
    x_train, x_test, y_train, y_test = one.split(0.25)
'''


