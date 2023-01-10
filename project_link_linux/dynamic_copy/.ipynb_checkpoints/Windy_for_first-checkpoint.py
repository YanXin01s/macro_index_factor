from WindPy import *
w.start()
w.isconnected()

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
#######图形模版

import seaborn as sns
cm=sns.color_palette("Spectral", as_cmap=True)
#cm = sns.light_palette("Spectral", as_cmap=True)
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark")
font = ['Songti SC']
parameters = {'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          "font.family" : "sans-serif",
          "font.sans-serif":font,
          'font.size':15,
          "axes.unicode_minus":False}
plt.rcParams.update(parameters)

sys.path.append("/Users/xinyuexu/Public/multi_asset/macro_and_strategy/")
sys.path.append("/Users/xinyuexu/Public/multi_asset/编程学习/financial_course/finance-courses/course_2_advanced_portfolio_construction_and_analysis_with_python/")
import ml_asset_kit as mla


import edhec_risk_kit as erk
from datetime import date
import seaborn as sb
today_for_api  = str(date.today())
