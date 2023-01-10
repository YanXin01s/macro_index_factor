#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:25:28 2022

@author: xinyuexu
"""
#####未整理完成



####计算超额基准收益



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