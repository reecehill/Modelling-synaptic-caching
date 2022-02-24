import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def makeFigure1c(directoryName):
    data = pd.read_csv(directoryName+'/output.csv', delimiter=',', na_values=['inf', 'nan'],
                       usecols=[2, 3, 5, 6, 10, 15]).dropna()
    data = data.where(data['Learning was complete at epoch #'] != False).sort_values('simulationTypeNumber').groupby('simulationTypeNumber')
    means = data[list(['n_pattern', 'n_pattern_features',
                 'Simulated: energy actually used by learning'])
                 ].mean(numeric_only=True).sort_values('n_pattern')

    y = means['Simulated: energy actually used by learning'].to_numpy()
    yMin = y.min() if y.min() > 0  else 0
    yMax = y.max()
    x = means['n_pattern'].to_numpy() / means['n_pattern_features'].to_numpy()
    # Setting the figure size and resolution
    fig = plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(x, y, color="red",  linewidth=2.5, linestyle="-")
    plt.yscale('log')
    # Setting the boundaries of the figure
    plt.xlim(0, 2)
    plt.ylim(yMin, yMax)
    plt.xlabel('number of patterns / number of synapses')  # add x-label
    plt.ylabel('energy used (a.u.)')  # add y-label
    fig.savefig(directoryName+"/figure1c.png", dpi=300)  # save figure
    plt.show()
    print('l')
