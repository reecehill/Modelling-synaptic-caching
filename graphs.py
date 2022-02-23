import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def makeFigure1c(directoryName):
    data = pd.read_csv(directoryName+'/output.csv', delimiter=',',
                       usecols=[2, 3, 5, 6, 15]).groupby('simulationTypeNumber')
    mean = data.mean()
    print(mean['Simulated: energy actually used by learning'].to_numpy())
    y = mean['Simulated: energy actually used by learning'].to_numpy()
    x = mean['n_pattern'].to_numpy() / mean['n_pattern_features'].to_numpy()
    # Setting the figure size and resolution
    fig = plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(x, y, color="red",  linewidth=2.5, linestyle="-")
    plt.yscale('log')
    # Setting the boundaries of the figure
    plt.xlim(x.min()*1, x.max()*1)
    plt.ylim(y.min()*1, y.max()*1)
    plt.xlabel('number of patterns / number of synapses')  # add x-label
    plt.ylabel('energy used (a.u.)')  # add y-label
    fig.savefig(directoryName+"/figure1c.png", dpi=300)  # save figure
    plt.show()
    print('l')
