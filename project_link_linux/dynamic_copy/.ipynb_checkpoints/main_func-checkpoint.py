#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 17:06:07 2022

@author: xinyuexu
"""



from support_function import *
import os

####读取文件

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
### 文章中table 4


#########


Fund_i_describe= fund_type_statistic_describe(dic_fund_Style,fund_file_name,data_index,index_factor_adj)
#### 文章中table 2
import pandas as pd
Fund_i_describe = pd.read_excel("/Users/xinyuexu/Desktop/fof/data/数据/分析结果/table_total.xlsx")

File_out_table2(Fund_i_describe)
#File_out_table3(Fund_i_describe)
File_out_table4(IR_mean.values,IR_sample_std)
















