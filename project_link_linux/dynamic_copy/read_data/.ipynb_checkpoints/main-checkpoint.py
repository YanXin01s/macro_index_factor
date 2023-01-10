#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:22:58 2022

@author: xinyuexu
"""

import pymc3 as pm
import numpy as np
from scipy import stats



import os
from utility import *
from utility_data_deal import *
import theano
from theano import tensor as T
#import theano.tensor as tt
# 一般都把 `tensor` 子模块导入并命名为 T





########load 指数数据## factor 数据计算
data_index_factor,data_index,data_factor= import_index_factor()
data_factor_adj,data_index_factor_adj=orthogonalized_factors(data_index_factor,data_index.name,data_factor.columns.to_list())
[IR_mean,IR_sample_mean,IR_sample_std]=Bootstrap_mean_std_IR(data_index_factor_adj)
data_index_factor_adj.index=data_index_factor_adj.index.to_period("M")

#########市场指数

dic_Style_fund,dic_fund_Style=fund_file2dic()
data_alpha=query_alpha_fund_type_data("大盘价值")
data_alpha.index= data_alpha.index.to_period("M")

######宏观指数
from load_macro import DB_new_interpolate
data = pd.DataFrame(DB_new_interpolate).ffill().iloc[:,[1,2,3]].dropna()
data.index=data.index.to_period("M")





#########


df_data = pd.DataFrame(columns = ["Period"]).set_index("Period")
date =pd.date_range(start="2010-02-01",end="2022-06-01",freq="M").to_period("M")
df_data.index=date
df_data["Test_"] = "True"
df_macro_filter = pd.concat([df_data,data],axis=1).query('Test_=="True"')

df_Factor_filter=pd.concat([df_data,data_index_factor_adj],axis=1).query('Test_=="True"')
data_alpha_filter =pd.concat([df_data,data_alpha],axis=1).query('Test_=="True"')
data_total =pd.concat([df_Factor_filter.drop("Test_",axis=1),data_alpha_filter.drop("Test_",axis=1)],axis=1).T


Df_macro = df_macro_filter.drop("Test_",axis=1).T
Df_Factor=df_Factor_filter.drop("Test_",axis=1).T

prior_mu = np.array([x for x in IR_mean])
prior_std = np.array([x for x in IR_sample_std])


coords = {"Fund": Df_Factor.index, "date": data_total.columns}

with pm.Model() as model12:
    sharpe_ratio= pm.Normal("μ", prior_mu.T, prior_std, shape=(6,))
    sharpe_ratio=T.reshape(sharpe_ratio,(6,1))
    beta = pm.Normal("beta", mu=0, sigma=20,shape=(6,3))
  
     
  
    data_macro = pm.Data("data", Df_macro.values.astype(float))

    sharpe_ratio_t = pm.Deterministic("SR_t",T.tile(sharpe_ratio,(1,148)) + T.dot(beta,data_macro))
    sd = pm.Uniform("sd", 0, 1)
    obs = pm.Normal("obs",mu=sharpe_ratio_t,sigma=T.tile)
    
    # chol_fund, corr_fund, stds_fund = pm.LKJCholeskyCov(
    #     "chol", n=6, eta=3.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
    # )
    # cov_fund = pm.Deterministic("cov_fund", chol_fund.dot(chol_fund.T))
    
   
    
  #  obs = pm.MvNormal("obs",mu=sharpe_ratio_t, chol=Cov,observed=fund_i_factor)

  #  df_macro.to_numpy()+
RANDOM_SEED = 8924
with model12:
    trace = pm.sample(draws=5000,random_seed=RANDOM_SEED, init="advi",chains=1,n_init=500, tune=500,target_accept=0.8,return_inferencedata=True)
 
      
    
   # observed=Df_Factor.values.astype(float)

   



  #  coe =  pm.Normal("coe", 0,3, shape=6)    
  #  β = pm.Normal('β', mu=0, sd=1)
    
    
    
  #  SR = pm.Deterministic("SR",sharpe_ratio.T*stds) 
    
    
    
    
    
    # i_alpha =T.dot(SR[:-1],coe)+SR[-1]
    
    # K = T.set_subtensor(SR[-1],i_alpha)
    
    # i_sigma = cov[-1,-1]+ T.dot(T.dot(coe,cov[:-1,:-1]),coe.T)
    # Cov =T.set_subtensor(cov[-1,-1],i_sigma)
    
    
    # obs = pm.MvNormal("obs",mu=K, chol=Cov,observed=fund_i_factor)
    
   

