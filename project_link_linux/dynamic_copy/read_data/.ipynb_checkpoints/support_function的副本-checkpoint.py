#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 16:13:14 2022

@author: xinyuexu
"""
import numpy as np
import pandas as pd
from datetime import *
import numpy as np
import statsmodels.api as sm
from numpy import mat
import pymc3 as pm
from recombinator.block_bootstrap import circular_block_bootstrap
from typing import Dict, List, Tuple
from typing import TypeVar
import warnings



try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass

warnings.filterwarnings("ignore")


def close2logr(data_close):
    data_logr=data_close.transform(np.log)-data_close.shift(1).transform(np.log)
    return data_logr




def file_book_name2data_month_index_factor(index_file_name,index_book_name):
    data_index=pd.read_excel(index_file_name,sheet_name =index_book_name,skiprows=[0],index_col=[0],parse_dates=['时间'])
    data_index=data_index.iloc[:-1]
    data_index["time"] = pd.to_datetime(data_index.index)
    data_index=data_index.set_index("time")   
    data_index_month=data_index.resample("M").last()
    data_index_month.index=data_index_month.index.to_period("M").asfreq("D",how="start").to_timestamp()
    data_index_month_logr=close2logr(data_index_month)
    return data_index_month_logr




def filename2fundname(filename):
    fund_name=[]
    for i in filename:
        if i[8]=="F":
            fund_i = i[:9]
            #fund_i =i[8]
            fund_name.append(fund_i)
    return fund_name




def data_fund_sample(data_fund,beachmark_index_data):
     
    fund_i_new = pd.concat([data_fund,beachmark_index_data], axis=1)
    fund_i_new = fund_i_new.dropna()
   
   # fund_i_new["超额收益率"] =  
   # fund_i_new = []
    return fund_i_new




 







    
