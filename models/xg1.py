# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import operator



def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

if __name__ == '__main__':
    train = pd.read_csv("../data/X.csv").ix[:,1:]
    cat_sel = [n for n in train.columns if n.startswith('cat')]  # 类别特征数值化
    for column in cat_sel:
        train[column] = pd.factorize(train[column].values, sort=True)[0] + 1

    params = {
        'min_child_weight': 100,
        'eta': 0.02,
        'colsample_bytree': 0.7,
        'max_depth': 12,
        'subsample': 0.7,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': 12
    }
    rounds = 1000
    #y = train['loss']
    y = pd.read_csv('../data/Y.csv')['FLGLCD']
    X = train.drop(['Sbck_year', 'Address'], 1)
    #X = train.drop(['Sbck_year'], 1)

    xgtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(params, xgtrain, num_boost_round=rounds)

    features = [x for x in train.columns if x not in ['Sbck_year', 'Address']]
    ceate_feature_map(features)

    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv("../notebooks/feat_importance.csv", index=False)

    plt.figure()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('Feature Importance')
    plt.xlabel('relative importance in %d iterations' % rounds)
    plt.show()




