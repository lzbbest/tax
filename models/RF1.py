# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json,re,sklearn
import matplotlib.pyplot as plt 
import seaborn as sns

def pre():
    path = 'data/X.csv'
    data = pd.read_csv(path)
    dropline = []
    for i in range(len(data)):
        try:
            np.isnan(data.label[i])
            dropline.append(i)
        except:
            pass
    data = data.drop(dropline)
    return data
data = pre()
'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = data

le = sklearn.preprocessing.LabelEncoder()  
le.fit(df['ADDRESS']) 
df['ADDRESS'] = le.transform(df['ADDRESS'])


x_train,x_test,y_train,y_test = train_test_split(df.ix[:,1:16],df.ix[:,-1],test_size=0.3,random_state=0)

features = df.columns[1:16]
clf = RandomForestClassifier()
#y, _ = pd.factorize(y_train)
clf.fit(x_train, y_train)
clf.predict(x_train)
'''