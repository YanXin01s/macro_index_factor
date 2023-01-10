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

import theano
from theano import tensor as T
#import theano.tensor as tt
# 一般都把 `tensor` 子模块导入并命名为 T







########读取基金数据建立词典
data_index_month,data_index_month_logr= import_index_factor()
#dic_Style_fund,dic_fund_Style=import_alpha_fund()

#import_index_factor()



