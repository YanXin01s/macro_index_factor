#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:10:37 2022

@author: xinyuexu
"""

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

