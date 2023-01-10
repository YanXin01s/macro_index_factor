import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from datetime import date
today_for_api  = str(date.today())
import pandas as pd
import numpy as np
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

















import pickle
f2 = open('temp.pkl', 'rb')  
macro_data = pickle.load(f2)  
macro_zscore = pickle.load(f2)  
index_orig = pickle.load(f2)  
index_transformer = pickle.load(f2)  
df_fund = pickle.load(f2)  
df_fund.index=df_fund.index.to_period("M")
f2.close() 

df_ret_ratio=index_orig

# 基于月度数据夏普估计：夏普方法2

#Whitelaw（1997）


index_sharpe=index_orig

df_ret_ratio_mean_monthly=index_sharpe.resample("M").sum()
df_ret_ratio_std_monthly=np.sqrt((df_ret_ratio**2).resample("M").sum())
df_sharpe =df_ret_ratio_mean_monthly/df_ret_ratio_std_monthly

df_sharpe.mean()






import arviz as az
import pymc as pm
import aesara.tensor as at
#import recombinator

from recombinator.block_bootstrap import circular_block_bootstrap

y_star_cb   = circular_block_bootstrap(df_sharpe, 
                               block_length=36, 
                               replications=500, 
                               replace=True)




IR_estimate_from_bootstrap = np.zeros((500,np.shape(y_star_cb)[2]))
for b in range(500):
       IR_estimate_from_bootstrap[b,:] =  np.mean(y_star_cb[b,:,:],axis=0)
IR_mean=IR_estimate_from_bootstrap.mean(axis=0)

IR_sample_std=IR_estimate_from_bootstrap.std(axis=0)

model_begin_date="2012-09-01"

index_transformer_month= index_transformer.resample("M").sum()



df_data = pd.DataFrame(columns = ["Period"]).set_index("Period")
date_model =pd.date_range(start=model_begin_date,end="2022-09-01",freq="M").to_period("M")
df_data.index=date_model
df_data["Test_"] = "True"



df_macro_filter = pd.concat([df_data,macro_zscore.shift(3)],axis=1).query('Test_=="True"')
df_macro_filter_predict = pd.concat([df_data,macro_zscore],axis=1).query('Test_=="True"')

df_Factor_filter=pd.concat([df_data,index_transformer_month],axis=1).query('Test_=="True"')
data_alpha_filter =pd.concat([df_data,df_fund],axis=1).query('Test_=="True"')
data_alpha_filter= data_alpha_filter.dropna(axis=1)

data_total =pd.concat([df_Factor_filter.drop("Test_",axis=1),data_alpha_filter.drop("Test_",axis=1)],axis=1).T


Df_macro = df_macro_filter.drop("Test_",axis=1).T
Df_macro_predict = df_macro_filter_predict.drop("Test_",axis=1).T


Df_Factor=(df_Factor_filter.drop("Test_",axis=1).T)*12
Df_alpha=(data_alpha_filter.drop("Test_",axis=1).T)*12
prior_mu = np.array([x for x in IR_mean])
prior_std = np.array([x for x in IR_sample_std])


RANDOM_SEED = 8924
Df_model_example = Df_macro.head(1)

macro_dim =Df_model_example.shape[0]

fund_dim= Df_Factor.shape[0]

model_dim =data_total.shape[0]

model_dim_T=date_model.shape[0]
alpha_mean__prior =0.36
alpha_std__prior =0.17
LKJ_eta__prior =3
LKJ_st__prior =1.0
LKJ_st_prior =1.0
T_len=model_dim_T
__Df_alpha_model=Df_alpha
alpha_len = __Df_alpha_model.shape[0]






fund_dim= Df_Factor.shape[0]

model_dim =data_total.shape[0]

model_dim_T=date_model.shape[0]




for i in Df_macro.index:
    model_name = "model_total_"+i
    result_name = "trace_alpha_total_"+i
    Df_model_example = Df_macro.loc[[i]]
    
    macro_dim =Df_model_example.shape[0]
    coords_test = {"Fund": Df_Factor.index.values, "date":data_total.columns.to_timestamp().astype("int"),"macro":Df_model_example.index.values,"Alpha":__Df_alpha_model.index.values}
    s="""with pm.Model(coords=coords_test) as {}:
        sharpe_ratio= pm.Normal("μ", prior_mu.T, prior_std, dims=("Fund"))
        sharpe_ratio=at.reshape(sharpe_ratio,(fund_dim,1))
        beta = pm.Normal("beta", mu=0, sigma=20,shape=(fund_dim,macro_dim),dims=("Fund","macro"))
        data_macro = pm.Data("data", Df_model_example.values.astype(float),mutable=True,dims=("macro","date"))
        sharpe_ratio_t = pm.Deterministic("SR_t",at.tile(sharpe_ratio,(1,model_dim_T)) + at.dot(beta,data_macro),dims=("Fund","date"))
        #sd_dist=pm.Exponential("std",1.0,shape=6,dims=("Fund"))
        sd_dist=pm.Exponential("std",1.0,shape=fund_dim,dims=("Fund"))
        sd_dist= at.reshape(sd_dist,(fund_dim,1))
        sd_T =at.tile(sd_dist,(1,model_dim_T)) 
        obs = pm.Normal("obs",mu=sharpe_ratio_t*sd_dist,sigma=sd_dist,observed=Df_Factor,dims=("Fund", "date"))


        ############################################
        ############################################
        ###############################################
        ############################################
        factor_loading = pm.Normal("factor_loading", mu=0, sigma=2,shape=fund_dim,dims=("Fund"))
        factor_loading= at.reshape(factor_loading,(fund_dim,1))


        factor_loading_T =at.tile(factor_loading,(1,T_len))                         
        factor_loading_T=factor_loading_T*sharpe_ratio_t                
        factor_loading_sum= factor_loading_T.sum(axis=0)
        factor_loading_sum =at.reshape(factor_loading_sum,(1,T_len)) 
        factor_loading_sum_stack=at.tile(factor_loading_sum,(alpha_len,1))  

        alpha = pm.Normal("μ_alpha",alpha_mean__prior, alpha_std__prior)
        alpha_n = pm.Normal("IC_alpha",alpha,sigma=1,shape=alpha_len,dims=("Alpha"))
        alpha_n= at.reshape(alpha_n,(alpha_len,1))


        IC_alpha_t = pm.Deterministic("SR_alpha_t",at.tile(alpha_n,(1,T_len)) + factor_loading_sum_stack,dims=("Alpha","date"))

        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", n=alpha_len, eta=LKJ_eta__prior, sd_dist=pm.Exponential.dist(LKJ_st__prior), compute_corr=True
        )






        __stds =  at.reshape(stds,(alpha_len,1)) 
        __stds_t = at.tile(__stds,(1,T_len)) 

        T_n_cov =np.eye(T_len)

        obs_sharpe=pm.MatrixNormal('_IC', mu=IC_alpha_t*__stds_t, rowchol=chol,colcov=T_n_cov,observed=__Df_alpha_model.values,dims=("Alpha","date"))

    
    """.format(model_name)
    exec(s)



    RANDOM_SEED = 8924
    h="""with {}:
        {} = pm.sample(draws=2000,random_seed=RANDOM_SEED, init="advi",chains=2,n_init=500, tune=500,target_accept=0.8,return_inferencedata=True)
    """.format(model_name,result_name)
    exec(h)
    Df_macro_example_predict = Df_macro_predict.loc[[i]]
    e="""with {}:
        pm.set_data({{"data": Df_macro_example_predict.values.astype(float)}})
        {} = pm.sample_posterior_predictive(
            {},
        var_names=["SR_t","obs","_IC"],
        return_inferencedata=True,
        predictions=True,
        extend_inferencedata=True,
        random_seed=RANDOM_SEED,
    )""".format(model_name,result_name,result_name)
    exec(e)
#__Df_alpha_fit_data_model=__Df_alpha_model.values[~np.isnan(__Df_alpha_model.values)]
#alpha_fit_variable = obs_sharpe[~np.isnan(__Df_alpha_model.values)]
#obss=pm.Normal("_IC",alpha_fit_variable,sigma=0.00001,observed=__Df_alpha_fit_data_model)







 az.plot_forest(
    trace_alpha_total_bm,
    var_names=["IC_alpha"],
    #kind="ridgeplot",
    combined=True,
    coords={"Alpha":__Df_alpha_model.index.values},
    labeller=az.labels.NoVarLabeller(),
)
    
    
    
    
    
for i in Df_macro.index:
    files="trace_{}.nc".format(i)
    s="""trace_alpha_total_{}.to_netcdf('{}')""".format(i,files)
    exec(s)
    

    

trace_alpha_total_dp.to_netcdf("trace_dp.nc")

















for i in Df_macro.index:
    files="trace_{}.nc".format(i)
    s="""trace_alpha_total_{}= az.from_netcdf('{}')""".format(i,files)
    exec(s)

for i in Df_macro.index:
    model_name = "model_total_"+i
    result_name = "trace_alpha_total_"+i
    Df_model_example = Df_macro.loc[[i]]
    
    macro_dim =Df_model_example.shape[0]
    coords_test = {"Fund": Df_Factor.index.values, "date":data_total.columns.to_timestamp().astype("int"),"macro":Df_model_example.index.values,"Alpha":__Df_alpha_model.index.values}
    s="""with pm.Model(coords=coords_test) as {}:
        sharpe_ratio= pm.Normal("μ", prior_mu.T, prior_std, dims=("Fund"))
        sharpe_ratio=at.reshape(sharpe_ratio,(fund_dim,1))
        beta = pm.Normal("beta", mu=0, sigma=20,shape=(fund_dim,macro_dim),dims=("Fund","macro"))
        data_macro = pm.Data("data", Df_model_example.values.astype(float),mutable=True,dims=("macro","date"))
        sharpe_ratio_t = pm.Deterministic("SR_t",at.tile(sharpe_ratio,(1,model_dim_T)) + at.dot(beta,data_macro),dims=("Fund","date"))
        #sd_dist=pm.Exponential("std",1.0,shape=6,dims=("Fund"))
        sd_dist=pm.Exponential("std",1.0,shape=fund_dim,dims=("Fund"))
        sd_dist= at.reshape(sd_dist,(fund_dim,1))
        sd_T =at.tile(sd_dist,(1,model_dim_T)) 
        obs = pm.Normal("obs",mu=sharpe_ratio_t*sd_dist,sigma=sd_dist,observed=Df_Factor,dims=("Fund", "date"))


        ############################################
        ############################################
        ###############################################
        ############################################
        factor_loading = pm.Normal("factor_loading", mu=0, sigma=2,shape=fund_dim,dims=("Fund"))
        factor_loading= at.reshape(factor_loading,(fund_dim,1))


        factor_loading_T =at.tile(factor_loading,(1,T_len))                         
        factor_loading_T=factor_loading_T*sharpe_ratio_t                
        factor_loading_sum= factor_loading_T.sum(axis=0)
        factor_loading_sum =at.reshape(factor_loading_sum,(1,T_len)) 
        factor_loading_sum_stack=at.tile(factor_loading_sum,(alpha_len,1))  

        alpha = pm.Normal("μ_alpha",alpha_mean__prior, alpha_std__prior)
        alpha_n = pm.Normal("IC_alpha",alpha,sigma=1,shape=alpha_len,dims=("Alpha"))
        alpha_n= at.reshape(alpha_n,(alpha_len,1))


        IC_alpha_t = pm.Deterministic("SR_alpha_t",at.tile(alpha_n,(1,T_len)) + factor_loading_sum_stack,dims=("Alpha","date"))

        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", n=alpha_len, eta=LKJ_eta__prior, sd_dist=pm.Exponential.dist(i), compute_corr=True
        )






        __stds =  at.reshape(stds,(alpha_len,1)) 
        __stds_t = at.tile(__stds,(1,T_len)) 

        T_n_cov =np.eye(T_len)

        obs_sharpe=pm.MatrixNormal('vals', mu=IC_alpha_t*__stds_t, rowchol=chol,colcov=T_n_cov)

        __Df_alpha_fit_data_model=__Df_alpha_model.values[~np.isnan(__Df_alpha_model.values)]
        alpha_fit_variable = obs_sharpe[~np.isnan(__Df_alpha_model.values)]
        obss=pm.Normal("_IC",alpha_fit_variable,sigma=0.00001,observed=__Df_alpha_fit_data_model)
    """.format(model_name)
    exec(s)
    
    
    
    
    
    
    
    
    
    
    
    
    
for i in Df_macro.index:
    model_name = "model_total_"+i
    result_name = "trace_alpha_total_"+i
    print(i)

    
    
    
    
trace_BMA=[trace_alpha_total_dp,
trace_alpha_total_dy,
trace_alpha_total_ep,
trace_alpha_total_de,
trace_alpha_total_svar,
trace_alpha_total_bm,
trace_alpha_total_ntis,
trace_alpha_total_tbl,
trace_alpha_total_ity,
trace_alpha_total_itr,
trace_alpha_total_tms,
trace_alpha_total_dfy,
trace_alpha_total_infl]






model_BMA=[]
for i in Df_macro.index:
    print( "model_total_"+i)
    model_BMA.append("model_total_"+i)
    
    
    

    
    
    
model_dict = dict(zip(model_BMA, trace_BMA))
comp_fund = az.compare(model_dict,ic="loo",method="BB-pseudo-BMA",b_samples=1000,seed=RANDOM_SEED, scale="log",var_name="obs")
comp_alpha = az.compare(model_dict,ic="loo",method="BB-pseudo-BMA",b_samples=1000,seed=RANDOM_SEED, scale="log",var_name="_IC")



import seaborn as sns
cm=sns.color_palette("Spectral", as_cmap=True)
comp_alpha.style.background_gradient(cmap=cm)




import seaborn as sns
cm=sns.color_palette("Spectral", as_cmap=True)
comp_fund.style.background_gradient(cmap=cm)






az.summary(trace_alpha_total_dp, var_names=["beta"], round_to=2)
az.summary(trace_alpha_total_ep, var_names=["beta"], round_to=2)





import xarray as xr
def weight_predictions_return(idatas, weights=None):
    """
    Generate weighted posterior predictive samples from a list of InferenceData
    and a set of weights.
    Parameters
    ---------
    idatas : list[InferenceData]
        List of :class:`arviz.InferenceData` objects containing the groups `posterior_predictive`
        and `observed_data`. Observations should be the same for all InferenceData objects.
    weights : array-like, optional
        Individual weights for each model. Weights should be positive. If they do not sum up to 1,
        they will be normalized. Default, same weight for each model.
        Weights can be computed using many different methods including those in
        :func:`arviz.compare`.
    Returns
    -------
    idata: InferenceData
        Output InferenceData object with the groups `posterior_predictive` and `observed_data`.
    See Also
    --------
    compare :  Compare models based on PSIS-LOO `loo` or WAIC `waic` cross-validation
    """
    if len(idatas) < 2:
        raise ValueError("You should provide a list with at least two InferenceData objects")

    if not all("predictions" in idata.groups() for idata in idatas):
        raise ValueError(
            "All the InferenceData objects must contain the `predictions` group"
        )

    if not all(idatas[0].observed_data.equals(idata.observed_data) for idata in idatas[1:]):
        raise ValueError("The observed data should be the same for all InferenceData objects")

    if weights is None:
        weights = np.ones(len(idatas)) / len(idatas)
    elif len(idatas) != len(weights):
        raise ValueError(
            "The number of weights should be the same as the number of InferenceData objects"
        )

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    len_idatas = [
        idata.predictions.dims["chain"] * idata.predictions.dims["draw"]
        for idata in idatas
    ]

    if not all(len_idatas):
        raise ValueError("At least one of your idatas has 0 samples")

    new_samples = (np.min(len_idatas) * weights).astype(int)

    new_idatas = [
        az.extract(idata, group="predictions",var_names=["obs","_IC"], num_samples=samples).reset_coords()
        for samples, idata in zip(new_samples, idatas)
    ]

    
    weighted_samples = az.InferenceData(
        predictions=xr.concat(new_idatas, dim="sample"),
        observed_data=idatas[0].observed_data,
    )

    return weighted_samples

import xarray as xr
def weight_predictions_(idatas, weights=None):
    """
    Generate weighted posterior predictive samples from a list of InferenceData
    and a set of weights.
    Parameters
    ---------
    idatas : list[InferenceData]
        List of :class:`arviz.InferenceData` objects containing the groups `posterior_predictive`
        and `observed_data`. Observations should be the same for all InferenceData objects.
    weights : array-like, optional
        Individual weights for each model. Weights should be positive. If they do not sum up to 1,
        they will be normalized. Default, same weight for each model.
        Weights can be computed using many different methods including those in
        :func:`arviz.compare`.
    Returns
    -------
    idata: InferenceData
        Output InferenceData object with the groups `posterior_predictive` and `observed_data`.
    See Also
    --------
    compare :  Compare models based on PSIS-LOO `loo` or WAIC `waic` cross-validation
    """
    if len(idatas) < 2:
        raise ValueError("You should provide a list with at least two InferenceData objects")

    if not all("predictions" in idata.groups() for idata in idatas):
        raise ValueError(
            "All the InferenceData objects must contain the `predictions` group"
        )

    if not all(idatas[0].observed_data.equals(idata.observed_data) for idata in idatas[1:]):
        raise ValueError("The observed data should be the same for all InferenceData objects")

    if weights is None:
        weights = np.ones(len(idatas)) / len(idatas)
    elif len(idatas) != len(weights):
        raise ValueError(
            "The number of weights should be the same as the number of InferenceData objects"
        )

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    len_idatas = [
        idata.predictions.dims["chain"] * idata.predictions.dims["draw"]
        for idata in idatas
    ]

    if not all(len_idatas):
        raise ValueError("At least one of your idatas has 0 samples")

    new_samples = (np.min(len_idatas) * weights).astype(int)

    new_idatas = [
        az.extract(idata, group="predictions",var_names=["SR_t"], num_samples=samples).reset_coords()
        for samples, idata in zip(new_samples, idatas)
    ]

    weighted_samples = az.InferenceData(
        predictions=xr.concat(new_idatas, dim="sample"),
        observed_data=idatas[0].observed_data,
    )

    return weighted_samples




ppc_pred_r=weight_predictions_return(trace_BMA,weights=weights.values)
ppc_pred=weight_predictions_(trace_BMA,weights=weights.values)


for i in Df_macro.index:
    model_name = "model_total_"+i
    result_name = "trace_alpha_total_"+i
   
    sss="""with {}:
 
    pm.sample_posterior_predictive({}, extend_inferencedata=True)
    """.format(model_name,result_name)

    RANDOM_SEED = 8924
    
    exec(sss)
    

    
    
ppc_pred_his=az.weight_predictions(trace_BMA,weights=weights.values)



ppc_pred_r.predictions["obs"]

#az.concat(ppc_pred_r.predictions["obs"], ppc_pred_r.predictions["_IC"])
data_SR=az.extract(ppc_pred,group="predictions", var_names="SR_t", combined=False).mean(axis=2).T.to_dataframe().unstack()
data_obs=az.extract(ppc_pred_r,group="predictions", var_names="obs", combined=False).mean(axis=2).T.to_dataframe().unstack()
data_obs.index =data_obs.index.astype('datetime64[ns]')
data_obs.mean()



data_obs=az.extract(ppc_pred_his,group="posterior_predictive", var_names="obs", combined=False).mean(axis=2).T.to_dataframe().unstack()
data_obs.index =data_obs.index.astype('datetime64[ns]')
(data_obs/12).cumsum().plot()

(Df_Factor/12).T.cumsum().plot()


from matplotlib.dates import date2num
regimelist
threshold=np.quantile(SR_t_std_fig[0],0.9)
regimelist=regime_switch(betas,threshold)
idx=SR_t_std_fig[0]>np.quantile(SR_t_std_fig[0],0.9)
TT_figure_predict[idx]










ppc_pred_r.predictions["obs"]

#az.concat(ppc_pred_r.predictions["obs"], ppc_pred_r.predictions["_IC"])
data_SR=az.extract(ppc_pred,group="predictions", var_names="SR_t", combined=False).mean(axis=2).T.to_dataframe().unstack()






month_interval =(trace_alpha_total_infl.observed_data.date).astype('datetime64[ns]')[3]-(trace_alpha_total_infl.observed_data.date).astype('datetime64[ns]')[0]

fig,ax =plt.subplots(fund_dim,1,figsize=(10,fund_dim*6))

SR_t_mean_fig=ppc_pred.predictions.stack()["SR_t"].mean(axis=2)
SR_t_std_fig=ppc_pred.predictions.stack()["SR_t"].std(axis=2)



TT_figure_predict=(ppc_pred.observed_data.date).astype('datetime64[ns]')+month_interval
TT_figure=(ppc_pred.observed_data.date).astype('datetime64[ns]')
data_fund=np.exp((Df_Factor/12).T.cumsum())
for i in range(fund_dim):
  
    idx=SR_t_std_fig[i]>np.quantile(SR_t_std_fig[i],0.9)
   
    
    ax[i].plot(TT_figure_predict,SR_t_mean_fig[i]+1*SR_t_std_fig[i],':',color="k")
    ax[i].plot(TT_figure_predict,SR_t_mean_fig[i])
    #ax[i].fill_between(TT_figure_predict,T_vol,0,alpha=0.3,color="r")
    
    ax[i].plot(TT_figure_predict,SR_t_mean_fig[i]-1*SR_t_std_fig[i],':',color="k")
    
    
    ax[i].set_title(Df_Factor.index.values[i])
    ax[i].grid()
    #ax[i].set_ylim([-2,2])
    
    ax_right = ax[i].twinx()
    
    #__drawdown =(draw_down_price[i] - draw_down_price[i+"_max"])/draw_down_price[i+"_max"]
    #print(__drawdown)
    ax_right.fill_between(TT_figure,data_fund[Df_Factor.index.values[i]],1,alpha=0.3)
    ax_right.set_ylim([data_fund[Df_Factor.index.values[i]].min(),data_fund[Df_Factor.index.values[i]].max()])
    x_low=TT_figure[idx]-month_interval/3
    x_high=TT_figure[idx]+month_interval/3
    
    #(TT_figure[idx][1:].values-TT_figure[idx][:-1].values)/2678400000000000
    
    
    betas =-SR_t_std_fig[i]
    threshold=-np.quantile(SR_t_std_fig[i],0.9)
    regimelist=regime_switch(betas,threshold)
    curr_reg = np.sign(betas[0]-threshold)
    for m in range(len(regimelist)-1):
        if curr_reg == 1:
            pass
            # uncomment below if we want to color the normal regimes
            #ax.axhspan(0, data.max(), xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
             #         facecolor="green", alpha=0.3)
        else:
            ax_right.axhspan(0, data_fund[Df_Factor.index.values[i]].max(),  xmin=regimelist[m]/regimelist[-1], xmax=regimelist[m+1]/regimelist[-1], 
                       facecolor='gray', alpha=0.5)
        curr_reg = -1 * curr_reg
        
    
  
    