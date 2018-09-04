# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json,re
import matplotlib.pyplot as plt 
import seaborn as sns

f = pd.DataFrame([1,1,1,1,1,1,1,3,7,7,7,7,8,8,6],columns=['table'])
f['attr'] = ['QYLX','ADDRESS','ZCLXCODE','HYCODE','ZCZJ','GSDJZ_RZRQ',
                    'SWCODE','SBCK_DATE','SL_MDSE','SL_TSSB_SC_NUM','SL_ZZS_TS_AMT',
                    'SL_TSSB_NUM','QNT','USD_AMT','ZZS_TS_AMT']


class Data(object):
    def __init__(self,path):
        self.table = self.__rename() # dict        
        self.data = self.__read(path) # list

    def __read(self,path):
        data = []
        with open(path,encoding="utf8") as f:
            for line in f:
                t_data = json.loads(line)
                try:
                    t_data[self.table[1]][0]['CPCODE']
                    data.append(t_data)
                except:
                    pass
        data.sort(key = lambda x:x['id'])
        return data
    
    def __rename(self):
        table={}
        table[1] = 'EnterpriseExpTaxRebateBasicInfo'
        table[2] = 'EnterpriseExpTaxRebateExInfo'
        table[3] = 'EnterpriseExpTaxRebateConfigInfo'
        table[4] = 'EnterpriseExpTaxRebateProgress'
        table[5] = 'EnterpriseExpTaxRebateProgressDetails'
        table[6] = 'EnterpriseExportTaxRefundHistory'
        table[7] = 'EnterpriseTaxRefundApplyAcceptance'
        table[8] = 'EnterpriseTaxRefundApplyAcceptanceDetails'
        table[9] = 'EnterpriseTaxRebateDeclarationInfo'
        return table
    
    def column(self,comp_id,table_id,colname): # all tables in one company
        alltable = self.data[comp_id][self.table[table_id]]# list
        attribute = []
        for t in alltable: # i = dict
            if re.match('\s+',t[colname]):
                pass
            else:
                attribute.append(t[colname])
        return attribute
    
    def allcolumn(self,table_id,colname,func):
        n = len(self.data)
        attr = []
        for i in range(n):
            try:
                temp_list = self.column(i,table_id,colname)
                #####  Aggregation operation
                temp = func(temp_list)
                #####
                attr.append(temp)
            except TypeError as error:
                attr.append([])
        return attr
    
    def process1(self):
        pass
    
    def getCompany(self):
        company = self.allcolumn(1,'CPCODE',lambda x:x)
        company = pd.DataFrame(company,columns=['CPCODE'])
        return company
    
    def getLabel(self):
        label = self.allcolumn(4,'FLGLCD',lambda x:x)
        #label = list(map(lambda x: x[0] if len(x)!=0 else np.nan,label))
        for i in range(len(label)):
            if len(label[i]) != 0:
                for j in range(len(label[i])):
                    if label[i][j] != '':
                        label[i] = label[i][j]
                        break
            else:
                label[i] = np.nan
        return label

def func1(lista):
    #lista = list(map(np.float64,lista))
    new = []
    for j in range(len(lista)):
        try:
            new.append(float(lista[j]))
        except:
            pass
    return np.mean(new)

def func2(lista):
    new = []
    for j in range(len(lista)):
        try:
            new.append(float(lista[j]))
        except:
            pass
    return sum(new)

if __name__ == '__main__':
    #one = Data(b'../data/data.json')
    one = Data(b'../data/nashui8000.json')
    company = one.getCompany()
    label = one.getLabel()
    
    f_Mat = pd.DataFrame(company['CPCODE'])
    for i in range(8):
        f_Mat[f['attr'][i]] = one.allcolumn((f['table'][i]),f['attr'][i],lambda x: x[0] if len(x)!=0 else np.nan)
    
    for i in range(8,12):
        f_Mat[f['attr'][i]] = one.allcolumn((f['table'][i]),f['attr'][i],func1)
    for i in range(12,15):
        f_Mat[f['attr'][i]] = one.allcolumn((f['table'][i]),f['attr'][i],func2)
        
    f_Mat['ADDRESS'] = f_Mat['ADDRESS'].str[3:5]
    f_Mat['ADDRESS'].replace('安区','宝安',inplace = True)
    f_Mat['GSDJZ_RZRQ'] = f_Mat['GSDJZ_RZRQ'].str[:4]
    f_Mat['SBCK_DATE'] = f_Mat['SBCK_DATE'].str[:4]

    for i in range(8,13):
        f_Mat.ix[:,i]=f_Mat.ix[:,i].apply(lambda x: np.nan if type(x)==list else x)
    f_Mat['label'] = label
    f_Mat.to_csv('../data/rawdata2.csv',index=False)





