# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json,re
import matplotlib.pyplot as plt 
import seaborn as sns
'''
f = pd.DataFrame([1,1,1,1,1,1,1,3,7,7,7,7,8,8,6],columns=['table'])
f['attr'] = ['QYLX','ADDRESS','ZCLXCODE','HYCODE','ZCZJ','GSDJZ_RZRQ',
                    'SWCODE','SBCK_DATE','SL_MDSE','SL_TSSB_SC_NUM','SL_ZZS_TS_AMT',
                    'SL_TSSB_NUM','QNT','USD_AMT','ZZS_TS_AMT']
'''

class Data(object):
    def __init__(self,path):
        self.table = self.__rename() # dict        
        self.data = self.__read(path) # list

    def __read(self,path):
        data = {}
        #key=company values=9 tables in DF format
        with open(path,encoding="utf8") as f:
            for line in f:
                t_data = json.loads(line)
                try:
                    name = t_data[self.table[1]][0]['CPCODE']
                    data[name] = {}
                    for i in range(1,10):
                        data[name][self.table[i]] = pd.DataFrame(t_data[self.table[i]]) 
                except:
                    pass
        #data.sort(key = lambda x:x['id'])        
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

if __name__ == '__main__':
    one = Data(b'../data/data.json')





