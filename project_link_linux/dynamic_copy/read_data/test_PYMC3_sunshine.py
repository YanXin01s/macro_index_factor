#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:19:05 2022

@author: xinyuexu
"""

import pymc3 as pm
import matplotlib.pyplot as plt

from scipy import stats
import arviz as az
import warnings
import os
from support_function import *
import theano

# 一般都把 `tensor` 子模块导入并命名为 T
from theano import tensor as T

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
import theano.tensor as tt


link="/Users/xinyuexu/Desktop/fof/data/"

filename = os.listdir(link+'数据/风格1')





########读取指数和因子
index_file_name =link+'数据/指数行情序列.xlsx'
index_book_name = '指数行情序列'
factor_colname = ["MSCI中国A股在岸低波(人民币)",'MSCI中国A股在岸动能(人民币)','MSCI中国A股在岸价值(人民币)',"MSCI中国A股在岸中小盘(人民币)",'MSCI中国A股在岸公司质量(人民币)']
benchmark_name ="MSCI中国A股在岸大盘(人民币)"

index_factor_colname =["MSCI中国A股在岸大盘(人民币)","MSCI中国A股在岸低波(人民币)",'MSCI中国A股在岸动能(人民币)','MSCI中国A股在岸价值(人民币)',"MSCI中国A股在岸中小盘(人民币)",'MSCI中国A股在岸公司质量(人民币)']

data_index_month_logr=file_book_name2data_month_index_factor(index_file_name,index_book_name)



data_factor=data_index_month_logr[factor_colname]
data_index = data_index_month_logr[benchmark_name]

data_index_factor =data_index_month_logr[index_factor_colname]


########读取基金数据建立词典
fund_file_name =link+'数据/fund_i_data_July.xlsx'
###基金数据的文件名
dic_Style_fund,dic_fund_Style=fund_file2dic(fund_file_name)
 
##基金数据的查询字典

#####



### 根据指数得到正交化之后的因子:
    
factor_adjust,index_factor_adj = orthogonalized_factors(data_index_factor,benchmark_name,factor_colname)

####读取指数和因子数据计算先验IR：



[IR_mean,IR_sample_mean,IR_sample_std]=Bootstrap_mean_std_IR(index_factor_adj)





benchmark_name=data_index.to_frame().columns[0]
factor_colname = index_factor_adj.columns.to_list()
fund_name = dic_fund_Style.keys()
fund_name_=[i for i in fund_name]


data_fund=query_fund_data(fund_name_[0],dic_fund_Style,fund_file_name)


fund_i_new = data_fund_sample(data_fund,data_index)
fund_i_factor = data_fund_sample(data_fund,index_factor_adj)





prior_mu = np.array([x for x in IR_mean])
prior_std = np.array([x for x in IR_sample_std])

mu_i = 2
sigma_i =0.17
prior_mu_all = np.append(prior_mu,mu_i)



prior_std_all = np.append(prior_std,sigma_i)


from sunshine import reg_filter,kk,DB_new,reg_filter_sunshine








IR_mean_i = IR_mean.copy()
T_i=fund_i_factor.index[0]



for i in reg_filter_sunshine:
    if reg_filter_sunshine[i][T_i]:
        #print(i)
        IR_mean_i[i]= reg_filter_sunshine[i][T_i] 






# with pm.Model() as model12:
#     chol, corr, stds = pm.LKJCholeskyCov(
#         "chol", n=7, eta=3.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
#     )
#     cov = pm.Deterministic("cov", chol.dot(chol.T))
    
    
#     sharpe_ratio= pm.Normal("μ", prior_mu_all.T, prior_std_all, shape=7)
    

    
#     coe =  pm.Normal("coe", 0,3, shape=6)
    
#     β = pm.Normal('β', mu=0, sd=1)
    
    
    
#     SR = pm.Deterministic("SR",sharpe_ratio.T*stds) 
    
#     i_alpha =T.dot(SR[:-1],coe)+SR[-1]
    
#     K = T.set_subtensor(SR[-1],i_alpha)
    
#     i_sigma = cov[-1,-1]+ T.dot(T.dot(coe,cov[:-1,:-1]),coe.T)
#     Cov =T.set_subtensor(cov[-1,-1],i_sigma)
    
    
#     obs = pm.MvNormal("obs",mu=K, chol=Cov,observed=fund_i_factor)
    
   
    
    
    
# RANDOM_SEED = 8924
# with model12:
#     trace = pm.sample(draws=5000,random_seed=RANDOM_SEED, init="advi",chains=1,n_init=500, tune=500,target_accept=0.8,return_inferencedata=True)
 
    
 



        


