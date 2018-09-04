# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

data = pd.read_csv('stest1.csv')
y_test = data['FLGLCD']
y_scores = data['prob']#.apply(lambda x: 1 if x>=0.5 else 0)

fpr, tpr, thresholds = roc_curve(y_test, y_scores)#, pos_label=2)
roc_auc = auc(fpr, tpr)

# Binary classification plot
plt.figure()
sns.set_style('dark')
lw = 2
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, color='darkorange', lw=lw,
        label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('Receiver operating characteristic',fontsize=30)
plt.legend(loc="lower right",fontsize=15)
plt.show()

