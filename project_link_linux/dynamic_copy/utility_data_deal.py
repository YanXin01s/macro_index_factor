#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:48:20 2022

@author: xinyuexu
"""

import statsmodels.api as sm
import pandas as pd
from recombinator.block_bootstrap import circular_block_bootstrap
from typing import Dict, List, Tuple
import numpy as np
####计算正交

def orthogonalized_factors(factor,index_name,factor_colname):
    factor = factor.dropna()
    data =factor
    epsilon_all =[]
    Beta_all=[]
    for i in factor_colname:
        exog_vars =index_name
        endog_var = i
     
        exog = data[exog_vars]
        endog =data[endog_var]
        exogs=sm.add_constant(exog)
     
        model=sm.OLS(endog,exogs) #最小二乘法
        res=model.fit() #拟合数据
        Beta=res.params  #取系数
        Alpha = res.params[0]
        t_score = res.tvalues
        epsilon = endog - res.fittedvalues+Alpha
        epsilon_all.append(epsilon)
        Beta_all.append(Beta)
    factor_adj =pd.concat(epsilon_all,axis=1)
    
    factor_adj.columns = factor_colname
    index_factor_adj = pd.concat([data[exog_vars],factor_adj],axis=1)
    return factor_adj,index_factor_adj,Beta_all
####计算boostrap 收益


def Bootstrap_mean_std_IR(factor_adjust,B:int =500,b_star_cb:int =36)-> List[float]:
    """B代表bootstrap的次数，b_star_cb代表时间序列block的长度"""

  
    # number of replications for bootstraps (number of resampled time-series to generate)
  
  
    y_star_cb   = circular_block_bootstrap(factor_adjust, 
                                   block_length=b_star_cb, 
                                   replications=B, 
                                   replace=True)
    
    
    IR_estimate_from_bootstrap = np.zeros((B,np.shape(factor_adjust)[1]))

    for b in range(B):
       
  
       IR_estimate_from_bootstrap[b,:] =  np.mean(y_star_cb[b,:,:],axis=0)/np.std(y_star_cb[b,:,:],axis=0)
    #print(np.mean(factor_adjust*12,axis=0))
    #print(np.std(factor_adjust*12,axis=0))
    return [np.mean(factor_adjust,axis=0)/np.std(factor_adjust,axis=0),np.mean(IR_estimate_from_bootstrap,axis =0),np.std(IR_estimate_from_bootstrap,axis=0)]






