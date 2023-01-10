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



def Timeseries(data):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, xlabel=r"Time", ylabel=r"{}".format{data.column})
    colors = np.linspace(0, 1, len(prices))
    mymap = plt.get_cmap("viridis")
    sc = ax.scatter(data.index, data.iloc[:,1], c=colors, cmap=mymap, lw=0)
    ticks = colors[:: len(prices) // 10]
    ticklabels = [str(p.date()) for p in data[:: len(data) // 10].index]
    cb = plt.colorbar(sc, ticks=ticks)
    cb.ax.set_yticklabels(ticklabels);