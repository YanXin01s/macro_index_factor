#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:34:06 2022

@author: Yan Xin
"""
import pandas as pd
link  = "/Users/xinyuexu/Desktop/fof/data/"
from dateutil.parser import parse
from datetime import datetime

import numpy as np


def chineseStr2time(x,y):
   # print(y)
    if y =="年月":
        year,_month = x.split("年")
        month,day = _month.split("月") 
        time = datetime(int(year),int(month),1)
    if y =="年":
        time =datetime(int(x),1,1)
    return time

class macro_data(object):

    def __init__(self):
        self.DB={}
        self.DY=3
        self.loadData()
       # self.loadYdata()
    
   
    
    def loadData(self):
        self.dfParams = pd.read_excel(link+"参数.xlsx",index_col =0)
        for k,v in self.dfParams.iterrows():
            
            df = pd.read_csv(link+v["文件名"],index_col =0)
            time_type= v["备注"]
            
            df.index = df.index.to_series().apply(lambda x:chineseStr2time(x,time_type))
            #print(df.index)
            self.DB[k]=df   
      
    
   

   
        
   

        
   # self.panel = pd.read_excel("静态数据",index_col)
    
    def __getitem__(self,key):
        return self.DB[key] if k in self.DB.keys() else None
      
    def __getattr__(self,key):
        return self.DB[key] if k in self.DB.keys() else None

    
    @property
    def date(self):
       return self.DB["MMI及其他中国经济指数历史表现"].index[-1]
    @property
    def code(self):
        pass
      #  return self.DB("MMI及其他中国经济指数历史表现").index[-1]
    @property
    def code_active(self):
        pass
    
   
kk =macro_data()

DB_new ={}
DB_new_interpolate ={}
for i in kk.DB:
    data = kk.DB[i]
    dx = data.iloc[:,-1]
    dx = (dx-np.mean(dx))/np.std(dx)
    dx[dx>2] = 2
    dx[dx<-2]= -2
    dx = dx.drop_duplicates()
    DB_new[i] = dx
    dx = dx.resample("D",convention="start").interpolate("linear")
    dx=dx.resample("M").last()
    dx.index=dx.index.to_period("M").asfreq("D",how="start").to_timestamp()
    DB_new_interpolate[i]=dx
    
    
  
   




       
 
      