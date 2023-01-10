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
def data_panel_resample(data,column_name):
    listname  = data[column_name].unique()
    data_re=[]
    data_result=[]
    for i in listname:
        data_sample_i = data.loc[data[column_name]==i,:]
        data_sample_i_re = data_sample_i.resample("M").last()
        data_sample_i_re["对数收益率"]=data_sample_i_re["收盘价(元)"].transform(np.log)-data_sample_i_re["收盘价(元)"].shift(1).transform(np.log)
        data_re.append(data_sample_i_re)
    data_result=pd.concat(data_re,axis =0)      
    return data_result


def filename2fundname(filename):
    fund_name=[]
    for i in filename:
        if i[8]=="F":
            fund_i = i[:9]
            #fund_i =i[8]
            fund_name.append(fund_i)
    return fund_name


def fund_data_style(fund_name,style="风格1"):
    data_style_all=[]
    for name_i in fund_name:
        data_style=pd.read_excel(link+'数据/'+style+"/"+name_i+".xlsx",index_col=[2],parse_dates=True)
        data_style=data_style[:-4]
        if data_style[:-4].size!=0:
            data_sample_i_re=data_style.resample("M").last()
            data_sample_i_re["对数收益率"]=data_sample_i_re["收盘价(元)"].transform(np.log)-data_sample_i_re["收盘价(元)"].shift(1).transform(np.log)
            data_style_all.append(data_sample_i_re)
    data_all=pd.concat(data_style_all,axis =0)      
    return data_all

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


def fund2_compute_active_return(fund_month_style,beachmark_index_data):
    name_code = fund_month_style["代码"].unique()
    fund_month_style1_bench=[]
    for code_i in name_code:
        fund_i=fund_month_style1[fund_month_style1["代码"]==code_i]
        beachmark_i=beachmark_index_data
        fund_i_new = pd.concat([fund_i,beachmark_index_data], axis=1)
        fund_i =fund_i_new[~fund_i_new["对数收益率"].isna()]
        fund_month_style1_bench.append(fund_i)
    data_all=pd.concat(fund_month_style1_bench,axis =0)  
    return data_all


def fund_data_style(fund_name,style="风格1"):
    data_style_all=[]
    for name_i in fund_name:
        data_style=pd.read_excel(link+'数据/'+style+"/"+name_i+".xlsx",index_col=[2],parse_dates=True)
        data_style=data_style[:-4]
        if data_style[:-4].size!=0:
            data_sample_i_re=data_style.resample("M").last()
            data_sample_i_re["对数收益率"]=data_sample_i_re["收盘价(元)"].transform(np.log)-data_sample_i_re["收盘价(元)"].shift(1).transform(np.log)
            data_style_all.append(data_sample_i_re)
    data_all=pd.concat(data_style_all,axis =0)      
    return data_all



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




def query_fund_data(fund_id,dic_fund_Style,fund_file_name):
    book = dic_fund_Style[fund_id]
    data_book_i = pd.read_excel(fund_file_name,skiprows=[0,2,3,4],sheet_name =book,index_col=0,parse_dates=True)
    data_fund=data_book_i[fund_id]
    data_fund=data_fund.resample("M").last()
    
    data_fund.index=data_fund.index.to_period("M").asfreq("D",how="start").to_timestamp()
    
    data_fund_month_log_r=close2logr(data_fund)
    data_fund_month_log_r = data_fund_month_log_r.dropna()
    
    return data_fund_month_log_r
    

def data_fund_sample(data_fund,beachmark_index_data):
     
    fund_i_new = pd.concat([data_fund,beachmark_index_data], axis=1)
    fund_i_new = fund_i_new.dropna()
   
   # fund_i_new["超额收益率"] =  
   # fund_i_new = []
    return fund_i_new



def data_fund_simple_statistic(fund_i_new):
    fund_colname=fund_i_new.columns[0]
    index_colname= fund_i_new.columns[1]
    fund_i_new = fund_i_new.assign(超额回报率=lambda x:x[fund_colname]-x[index_colname])
    fund_active_ExRet = fund_i_new["超额回报率"].mean()
    fund_active_Sigma = fund_i_new["超额回报率"].std()
    fund_active_sample = np.shape(fund_i_new)[0]
    
    return [fund_active_ExRet,fund_active_Sigma,fund_active_sample]
 



def data_fund_simple_sample(fund_i_new):
    fund_colname=fund_i_new.columns[0]
    index_colname= fund_i_new.columns[1]
    fund_i_new = fund_i_new.assign(超额回报率=lambda x:x[fund_colname]-x[index_colname])

    
    return fund_i_new
  
def data_fund_factor_statistic(fund_i_factor,fund_colname,index_colname,factor_colname):
    #print(fund_i_factor[fund_colname])
  
    data = fund_i_factor.assign(超额回报率=lambda x:x[fund_colname]-x[index_colname])
 
    exog_vars =factor_colname
    endog_var = "超额回报率"
 
    exog = data[exog_vars]
    endog =data[endog_var]
    exogs=sm.add_constant(exog)
 
    model=sm.OLS(endog,exogs) #最小二乘法
    res=model.fit() #拟合数据
    Beta=res.params  #取系数
    Alpha = res.params[0]
    epsilon = endog - res.fittedvalues
    t_values = res.tvalues
    std = np.std(epsilon)
    IR=Alpha/std

    return [Beta,t_values,std]   
  
    
def orthogonalized_factors(factor,index_name,factor_colname):
    factor = factor.dropna()
    data =factor
    epsilon_all =[]
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
    factor_adj =pd.concat(epsilon_all,axis=1)
    
    factor_adj.columns = factor_colname
    index_factor_adj = pd.concat([data[exog_vars],factor_adj],axis=1)
    return factor_adj,index_factor_adj



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



def fund_type_statistic_describe(dic_fund_Style,fund_file_name,data_index,index_factor_adj):
    benchmark_name=data_index.to_frame().columns[0]
    factor_colname = index_factor_adj.columns.to_list()
    fund_name = dic_fund_Style.keys()
    #fund_type = set(dic_fund_Style.values())
    
   # fund_type = np.zeros([len(fund_type),4])
    ### num=4 分别用于记载每种类别中的active return,active risk,IR以及样本数目
    fund_i_ret_risk_IR = np.zeros([len(fund_name),3])
    fund_i_ret_factor_loadings = np.zeros([len(fund_name),len(factor_colname)+1])
    fund_i_ret_factor_tvalues = np.zeros([len(fund_name),len(factor_colname)+1])
    fund_i_ret_factor_sigma = np.zeros([len(fund_name),1])
    #####
    
    n = len(fund_name)
    fund_name_=[i for i in fund_name]
    for i,j in enumerate(fund_name_):
    
        data_fund=query_fund_data(j,dic_fund_Style,fund_file_name)
        if len(data_fund)<12:
            continue
        
        ##########基金数据与指数、风格数据合并
        fund_i_new = data_fund_sample(data_fund,data_index)
        fund_i_factor = data_fund_sample(data_fund,index_factor_adj)
      
        print(str(i)+"/"+str(n))
        print(j)
         ##基金+指数数据的超额收益计算
    
        [fund_active_ExRet,fund_active_Sigma,fund_active_sample]=data_fund_simple_statistic(fund_i_new)
    
   
        [Beta,t_values,std] =data_fund_factor_statistic(fund_i_factor,j,benchmark_name,factor_colname)
        ##基金与风格指数的OLS回归分析
        fund_i_ret_risk_IR[i,:] = fund_active_ExRet,fund_active_Sigma,fund_active_ExRet/fund_active_Sigma
        ########读取个股基金数据建立OLS回归框架
        fund_i_ret_factor_loadings[i,:] = Beta
        fund_i_ret_factor_tvalues[i,:] = t_values
        fund_i_ret_factor_sigma[i,:]=std
    
    Fund_i_describe = np.concatenate([fund_i_ret_risk_IR,fund_i_ret_factor_loadings,fund_i_ret_factor_tvalues,fund_i_ret_factor_sigma],axis=1)

    colnames_beta = [k+"beta系数" for k in factor_colname]
    colnames_tscore = [k+"t_score" for k in factor_colname]
    colnames_fund_i = ["超额收益","超额风险","IR"]
    colnames_fund_all = [colnames_fund_i,["常数回归"],colnames_beta,["常数回归t"],colnames_tscore,["超额收益回归后标准差"]]
    colname_fund_all_= [j for i in colnames_fund_all for j in i]
    Fund_i_describe_ = pd.DataFrame(Fund_i_describe)
    Fund_i_describe_.columns = colname_fund_all_

    Fund_i_describe_= Fund_i_describe_.assign(基金分类=[zz for zz in  dic_fund_Style.values()])
    return Fund_i_describe_


def File_out_table2(Fund_i_describe):
   
    Table2 =Fund_i_describe[["超额收益","超额风险","IR","基金分类"]].groupby("基金分类").mean()
    Table2_1 = Fund_i_describe[["超额收益","基金分类"]].groupby("基金分类").count()
    Table2_1.columns=["样本数目"]
    A = pd.concat([Table2,Table2_1],axis=1)
    A.columns = ["超额收益","超额风险","IR","样本数目"]
    A.to_excel("/Users/xinyuexu/Desktop/fof/data/数据/分析结果/table2.xlsx")
    
    
def File_out_table3(Fund_i_describe):
    Fund_type_table3 = np.zeros([9*2,7])
    
    fund_i_colname_coe = Fund_i_describe.columns[3:10].tolist().append("基金分类")
    fund_i_colname_tscore= Fund_i_describe.columns[10:17].append("基金分类")
    Fund_Type_coe = Fund_i_describe[fund_i_colname_coe].groupby("基金分类").mean()
    Fund_Type_tscore=Fund_i_describe[fund_i_colname_tscore].groupby("基金分类").mean()
    fund_type_list =['大盘价值', '大盘平衡', '大盘成长','中盘价值', '中盘均衡', '中盘成长', '小盘价值', '小盘平衡']
    for i in range(8):
        Fund_type_table3[2*i,:]=Fund_Type_coe[fund_type_list[i]]
        Fund_type_table3[2*i+1,:]=Fund_Type_tscore[fund_type_list[i]]
    Fund_type_table3[17,:] = Fund_Type_coe.mean(axis=1)
    Fund_type_table3[17,:] = Fund_Type_tscore.mean(axis=1) 
    Fund_type_table3.to_excel("/Users/xinyuexu/Desktop/fof/data/数据/分析结果/table3.xlsx")
    
    
    
def File_out_table4(IR_mean,IR_sample_std):
    factor_list = ["大盘指数","低波","动能","价值","市值","质量"]
    IR_total = np.zeros([6,2])
    IR_total[:,0]=IR_mean
    IR_total[:,1]=IR_sample_std
    table4 = pd.DataFrame(IR_total,index=factor_list,columns=["平均IR","IR的标准误"])
    table4.to_excel("/Users/xinyuexu/Desktop/fof/data/数据/分析结果/table4.xlsx")
    
