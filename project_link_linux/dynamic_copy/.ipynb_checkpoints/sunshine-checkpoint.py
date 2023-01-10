#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:22:42 2022

@author: xinyuexu
"""

from support_function import *
import os
import numpy as np
from load_macro import *
####读取文件
import matplotlib.pyplot as plt
link="/Users/xinyuexu/Desktop/fof/data/"

index_file_name =link+'数据/指数行情序列.xlsx'
index_book_name = '指数行情序列'
factor_colname = ["MSCI中国A股在岸低波(人民币)",'MSCI中国A股在岸动能(人民币)','MSCI中国A股在岸价值(人民币)',"MSCI中国A股在岸中小盘(人民币)",'MSCI中国A股在岸公司质量(人民币)']
benchmark_name ="MSCI中国A股在岸大盘(人民币)"

index_factor_colname =["MSCI中国A股在岸大盘(人民币)","MSCI中国A股在岸低波(人民币)",'MSCI中国A股在岸动能(人民币)','MSCI中国A股在岸价值(人民币)',"MSCI中国A股在岸中小盘(人民币)",'MSCI中国A股在岸公司质量(人民币)']

data_index_month_logr=file_book_name2data_month_index_factor(index_file_name,index_book_name)



data_factor=data_index_month_logr[factor_colname]
data_index = data_index_month_logr[benchmark_name]

data_index_factor =data_index_month_logr[index_factor_colname]
num =6
fig,axes = plt.subplots(num,1,figsize= (10,50*num/20),sharex=True)

import statsmodels.api as sm
reg = {}
reg_filter = {}
reg_filter_sunshine = {}
n=0
for i in data_index_factor.columns:
    dy = data_index_factor[i]
    dy = dy.rolling(window=36).mean()/dy.rolling(window=36).std()
   # dy.plot()
    for j in kk.DB:
        dx = kk.DB[j].iloc[:,-1]
        dx = (dx-np.mean(dx))/np.std(dx)
        dx[dx>2] = 2
        dx[dx<-2]= -2
        dx = dx.drop_duplicates()
       
        data_ols =pd.concat([dy,dx],axis=1).dropna()
        dx = data_ols[data_ols.columns[1]]
        endog = data_ols[data_ols.columns[0]]
   
    
        exogs=sm.add_constant(dx)
     
        model=sm.OLS(endog,exogs) #最小二乘法
        
        res=model.fit() #拟合数据
        Beta=res.params  #取系数
        #Alpha = res.params[0]
        #epsilon = endog - res.fittedvalues
        t_values = res.tvalues
      
        if abs(t_values[1])>2:
            reg[i+":"+j]=t_values[1]
            reg_filter[i] = j
            reg_filter_sunshine[i] =res.fittedvalues
            print(n)
            #ax = fig.add_subplot(51, xlabel="x", ylabel="y", title="Generated data and underlying model")
            axes[n].plot(dx, endog, "x", label="sampled data")
                        
            axes[n].plot(dx, res.fittedvalues, label="true regression line", lw=2.0)
            axes[n].set_xlabel(j)
            axes[n].set_ylabel(data_ols.columns[0][4:-5])
            axes[n].legend(loc=1)
            n=n+1
            

        
            
        
        
        
        
        