# -*- coding: utf-8 -*-
import sklearn as sk 
import xgboost as xgb 
import numpy as np 
import pandas as pd 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
#from pandas_ml import ConfusionMatrix # 引入混淆矩阵
from xgboost.sklearn import XGBClassifier
from sklearn.externals import joblib # 保存和读取模型用
import sklearn
import copy
from preprocess import feature

data = pd.read_csv('../data/data.csv')
ID,mat,label = data['CPCODE'],data.iloc[:,1:-1],data['label']
one = feature(ID,mat,label)
one.run()
train_X, test_X, train_y, test_y = one.split(0.25)

# dtrain = xgb.DMatrix(train_X, label = train_y)
# dtest = xgb.DMatrix(test_X, label = test_y)
'''
XGBC = xgb.XGBClassifier(
    gamma = 0.1,                      # Gamma指定了节点分裂所需的最小损失函数下降值，值越大，算法越保守。
    learning_rate = 0.3,              # 学习速率
    max_delta_step = 0,               # 限制每棵树权重改变的最大步长。0为没有限制，越大越保守。可用于样本不平衡的时候。
    max_depth = 5,                    # 树的最大深度
    min_child_weight = 6,             # 最小叶子节点样本权重和。低避免过拟合，太高导致欠拟合。
    missing = None,                   # 如果有缺失值则替换。默认 None 就是 np.nan
    n_estimators = 250,               # 树的数量
    nthread = 8,                      # 并行线程数量
    objective = 'binary:logistic',    # 指定学习任务和相应的学习目标或要使用的自定义目标函数
    #'objective':'multi:softprob',    # 定义学习任务及相应的学习目标
    #'objective':'reg:linear',        # 线性回归
    #'objective':'reg:logistic',      # 逻辑回归
    #'objective':'binary:logistic',   # 二分类的逻辑回归问题，输出为概率
    #'objective':'binary:logitraw',   # 二分类的逻辑回归问题，输出结果为 wTx，wTx指机器学习线性模型f(x)=wTx+b
    #'objective':'count:poisson'      # 计数问题的poisson回归，输出结果为poisson分布
    #'objective':'multi:softmax'      # 让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class
    #'objective':'multi:softprob'     # 和softmax一样，但是输出的是ndata * nclass的向量，
                                      # 可以将该向量reshape成ndata行nclass列的矩阵。
                                      # 每行数据表示样本所属于每个类别的概率。
    reg_alpha = 1,                    # 权重的L1正则化项。默认1
    reg_lambda = 1,                   # 权重的L2正则化项。默认1
    scale_pos_weight = 10000,         # 数字变大，会增加对少量诈骗样本的学习权重，这里10000比较好
    seed = 0,                         # 随机种子
    silent = True,                    # 静默模式开启，不会输出任何信息
    subsample = 0.9,                  # 控制对于每棵树，随机采样的比例。减小会更加保守，避免过拟,过小会导致欠拟合。
    base_score = 0.5)                 # 所有实例的初始预测评分,全局偏差 

bst = XGBC.fit(train_X, train_y)
preds = bst.predict(test_X) # 对测试集作出预测
proba_df = bst.predict_proba(test_X)
print("训练完成！")
print("保存模型...")
joblib.dump(bst,'XGBmodel.pkl') # 保存模型
print("保存成功！")
'''
# 精确度（Precision）：
# P = TP/(TP+FP) ;  反映了被分类器判定的正例中真正的正例样本的比重
#print("精确度（Precision）：", precision_score(test_y, preds, average='macro')) # ?? %
#print("召回率（Recall）：", recall_score(test_y, preds, average='macro')) # ?? %
'''
predicted_y = np.array(preds)
right_y = np.array(test_y)
 
# 混淆矩阵的每一列代表了预测类别，
# 每一列的总数表示预测为该类别的数据的数目；
# 每一行代表了数据的真实归属类别，
# 每一行的数据总数表示该类别的数据实例的数目。
confusion_matrix = ConfusionMatrix(right_y, predicted_y)
# print("Confusion matrix:\n%s" % confusion_matrix)
# confusion_matrix.plot(normalized=True)
# plt.show()
confusion_matrix.print_stats()

a=copy.copy(proba_df)
def score1(a):
    m = np.min(a,axis=1)
    m = m + (1e-15)
    b=np.max(a,axis=1)/m
    b=np.log(b).reshape(-1,1)
    b=sklearn.preprocessing.MinMaxScaler().fit_transform(b)
    return b

def score2(a,la):
    m = np.min(a,axis=1)
    m = m + (1e-15)
    b=np.max(a,axis=1)/m
    #b=np.log(b)#.reshape(-1,1)
    if la==0:
        b = np.log(b).reshape(-1,1)
    else:
        b = (b**la-1)/la # BOx-Cox 变换
    return b

c=score2(a,0)
plt.hist(c)

'''

mat['QNT'].quantile([0.25,0.5,0.75])

























