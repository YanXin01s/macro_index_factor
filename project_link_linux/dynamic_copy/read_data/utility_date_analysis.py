import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings(action="ignore", category=MatplotlibDeprecationWarning)


RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")



def Timeseries_orginal_data(data):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, xlabel=r"Time", ylabel=r"{}".format{data.column})
    colors = np.linspace(0, 1, len(prices))
    mymap = plt.get_cmap("viridis")
    sc = ax.scatter(data.index, data.iloc[:,1], c=colors, cmap=mymap, lw=0)
    ticks = colors[:: len(prices) // 10]
    ticklabels = [str(p.date()) for p in data[:: len(data) // 10].index]
    cb = plt.colorbar(sc, ticks=ticks)
    cb.ax.set_yticklabels(ticklabels);
    
def Timeseries_orginal_data(data.posterior)
    fig = plt.figure(figsize=(8, 6), constrained_layout=False)
    ax = plt.subplot(111, xlabel="time", ylabel="alpha", title="Change of alpha over time.")
    ax.plot(trace_rw.posterior.stack(pooled_chain=("chain", "draw"))["alpha"], "r", alpha=0.05)

    ticks_changes = mticker.FixedLocator(ax.get_xticks().tolist())
    ticklabels_changes = [str(p.date()) for p in prices[:: len(prices) // 7].index]
    ax.xaxis.set_major_locator(ticks_changes)
    ax.set_xticklabels(ticklabels_changes)

    fig.autofmt_xdate()