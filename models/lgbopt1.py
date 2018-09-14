# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy,csv
from sklearn.model_selection import KFold
from preprocess import feature
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from hyperopt.pyll.stochastic import sample
import lightgbm as lgb
'''
data = pd.read_csv('../data/data.csv')
ID,mat,label = data['CPCODE'],data.iloc[:,1:-1],data['label']
one = feature(ID,mat,label)
one.run()
train_X, test_X, train_y, test_y = one.split(0.25)
'''
train_data = lgb.Dataset(train_X,label=train_y,
                         feature_name=list(train_X.columns), 
                         categorical_feature=['FLGLCD','ZCLXCODE','HYCODE','NSRLB','QYLX'])
test_data = lgb.Dataset(test_X,label=test_y,
                        reference=train_data,
                        feature_name=list(test_X.columns), 
                        categorical_feature=['FLGLCD','ZCLXCODE','HYCODE','NSRLB','QYLX'])


params = {
        'random_state':50,
        'is_unbalance':True,
        'metric':{'binary_logloss','auc'}
        }
'''
#bst=lgb.cv(params,train_data, num_boost_round=1000, nfold=10, early_stopping_rounds=100)
gbm = lgb.train(params,  
                train_data,  
                num_boost_round=len(bst['auc-mean']))
'''
gbm = lgb.train(params,train_data)
ypred = gbm.predict(test_X)


def draw(gbm): # gbm after training
    importance = gbm.feature_importance()  
    names = gbm.feature_name()
    df=pd.DataFrame(importance,columns=['score'])
    df['name'] = names
    df = df.sort_values(by='score')
    df.plot(kind='barh',x='name',y='score',legend=False, figsize=(6, 10))
draw(gbm)
'''
# Create the dataset
train_set = lgb.Dataset(train_X, train_y)

def objective(params, n_folds = 10):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgb.cv(params, train_set, nfold = n_folds, num_boost_round = 10000,
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)

    # Extract the best score
    best_score = max(cv_results['auc-mean'])

    # Loss must be minimized
    loss = 1 - best_score

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

'''


'''
# Default gradient boosting machine classifier
model = lgb.LGBMClassifier(
        random_state=50        
        )

# Define the search space
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt',
                                    'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                 {'boosting_type': 'dart',
                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                 {'boosting_type': 'goss'}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

# boosting type domain
boosti_type = {'boostinngg_type': hp.choice('boosting_type',
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}])}

# Sample from the full space
example = sample(space)

# Dictionary get method with default
subsample = example['boosting_type'].get('subsample', 1.0)

# Assign top-level keys
example['boosting_type'] = example['boosting_type']['boosting_type']
example['subsample'] = subsample

# Algorithm
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()

MAX_EVALS = 500
# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = bayes_trials)
'''
































    
    
