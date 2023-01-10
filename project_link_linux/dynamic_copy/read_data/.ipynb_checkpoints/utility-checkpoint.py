#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:23:26 2022

@author: xinyuexu
"""
import matplotlib.pyplot as plt
#from support_function import *
import warnings
import arviz as az
import pandas as pd
import numpy as np
try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass

warnings.simplefilter(action="ignore", category=FutureWarning)
#%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] =['Arial Unicode MS']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题



def path_to_data_folder():
    return "/Users/xinyuexu/Desktop/FOF/数据仓/wind数据/data/"



def import_alpha_fund():
    
    fund_file_name =path_to_data_folder()+'FOF数据/fund_i_data_July.xlsx'
    ###基金数据的文件名
    dic_Style_fund,dic_fund_Style=fund_file2dic(fund_file_name)
    return dic_Style_fund,dic_fund_Style
#def import_index_factor():
    
    
def import_index_factor():
    
    index_file_name =path_to_data_folder()+'factor_price.xlsx'

    factor_colname = ["MSCI中国A股在岸低波(人民币)",'MSCI中国A股在岸动能(人民币)','MSCI中国A股在岸价值(人民币)',"MSCI中国A股在岸中小盘(人民币)",'MSCI中国A股在岸公司质量(人民币)']
    benchmark_name ="MSCI中国A股在岸大盘(人民币)"
    index_factor_colname =["MSCI中国A股在岸大盘(人民币)","MSCI中国A股在岸低波(人民币)",'MSCI中国A股在岸动能(人民币)','MSCI中国A股在岸价值(人民币)',"MSCI中国A股在岸中小盘(人民币)",'MSCI中国A股在岸公司质量(人民币)']
    
    data_index=pd.read_excel(index_file_name,sheet_name ="Sheet1",index_col=[0])
    data_index=data_index.iloc[:-1]
    data_index["time"] = pd.to_datetime(data_index.index)
    data_index=data_index.set_index("time")   
    data_index_month=data_index.resample("M").last()
    data_index_month.index=data_index_month.index.to_period("M").asfreq("D",how="start").to_timestamp()
    data_index_month_logr=close2logr(data_index_month)
    return data_index_month,data_index_month_logr
    



#data_factor=data_index_month_logr[factor_colname]
#data_index = data_index_month_logr[benchmark_name]

#data_index_factor =data_index_month_logr[index_factor_colname]

    

def close2logr(data_close):
    
    data_logr=data_close.transform(np.log)-data_close.shift(1).transform(np.log)
    return data_logr

    
def fund_file2dic(fund_file_name):
    data_book = pd.ExcelFile(fund_file_name)
    book_name = data_book.sheet_names;
    dic_book2index ={}
    reverse_dic = {}
    for i in book_name:
        data_book_i = pd.read_excel(fund_file_name,skiprows=[0,2,3,4],sheet_name =i,index_col=0,parse_dates=True)
        
        dic_book2index[i] = data_book_i.columns
        for j in data_book_i.columns:
            reverse_dic[j] =i
    return dic_book2index,reverse_dic