import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def showFigures():
    plt.show()

def makeFigure1c(directoryName):
    data = pd.read_csv(directoryName+'/output.csv', delimiter=',', na_values=['inf', 'nan'],
                       usecols=[2, 3, 5, 6, 10, 14, 15]).dropna()
    data = data.where(data['Learning was complete at epoch #'] != False).sort_values(
        'simulationTypeNumber').groupby('simulationTypeNumber')
    means = data[list(['n_pattern', 'n_pattern_features',
                       'Theoretical: minimum energy for learning',
                       'Simulated: energy actually used by learning'])
                 ].mean(numeric_only=True).sort_values('n_pattern')

    y1 = means['Simulated: energy actually used by learning'].to_numpy()
    y2 = means['Theoretical: minimum energy for learning'].to_numpy()
    y1Min = y1.min() if y1.min() > 0 else 0
    y1Max = y1.max()
    x = means['n_pattern'].to_numpy() / means['n_pattern_features'].to_numpy()
    # Setting the figure size and resolution
    fig = plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(x, y1, color="red",  linewidth=1, linestyle="-")
    plt.plot(x, y2, color="green",  linewidth=1, linestyle="-")
    plt.yscale('log')
    # Setting the boundaries of the figure
    plt.xlim(0, 2)
    #plt.ylim(10**3, 10**9)
    plt.xlabel('number of patterns / number of synapses')  # add x-label
    plt.ylabel('energy used (a.u.)')  # add y-label
    fig.savefig(directoryName+"/figure1c.png", dpi=300)  # save figure
    return fig


def makeFigure1d(directoryName):
    data = pd.read_csv(directoryName+'/output.csv', delimiter=',', na_values=['inf', 'nan'],
                       usecols=[2, 3, 5, 6, 10, 11, 12]).dropna()
    data = data.where(data['Learning was complete at epoch #'] != False).sort_values(
        'simulationTypeNumber').groupby('simulationTypeNumber')
    means = data[list(['n_pattern', 'n_pattern_features',
                       'Theoretical: random-walk efficiency',
                       'Simulated: efficiency (m_perc/m_min)'])
                 ].mean(numeric_only=True).sort_values('n_pattern')

    y1 = means['Theoretical: random-walk efficiency'].to_numpy()
    y2 = means['Simulated: efficiency (m_perc/m_min)'].to_numpy()
    y1Min = y1.min() if y1.min() > 0 else 0
    y1Max = y1.max()
    x = means['n_pattern'].to_numpy() / means['n_pattern_features'].to_numpy()
    # Setting the figure size and resolution
    fig = plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(x, y1, color="blue",  linewidth=1, linestyle="-")
    plt.plot(x, y2, color="black",  linewidth=1, linestyle="-")
    plt.yscale('log')
    # Setting the boundaries of the figure
    plt.xlim(0, 2)
    #plt.ylim(10**3, 10**9)
    plt.xlabel('number of patterns / number of synapses')  # add x-label
    plt.ylabel('Inefficiency')  # add y-label
    fig.savefig(directoryName+"/figure1d.png", dpi=300)  # save figure
    return fig


def makeFigure2b(directoryName):
    data = pd.read_csv(directoryName+'/output.csv', delimiter=',', na_values=['inf', 'nan'],
                       usecols=[2, 3, 10, 15, 17, 18, 31]).dropna()
    data = data.where(data['Learning was complete at epoch #'] != False).sort_values(
        'simulationTypeNumber').groupby('simulationTypeNumber')
    means = data[list(['max_size_of_transient_memory',
                       'Simulated: energy actually used by learning',
                       'Energy expended by simulations for consolidations',
                       'Energy expended by simulations for maintenance'])
                 ].mean(numeric_only=True).sort_values('max_size_of_transient_memory')

    y1 = means['Simulated: energy actually used by learning'].to_numpy()
    y2 = means['Energy expended by simulations for consolidations'].to_numpy()
    y3 = means['Energy expended by simulations for maintenance'].to_numpy()
    #y2 = means['Theoretical: minimum energy for learning'].to_numpy()
    y1Min = y1.min() if y1.min() > 0 else 0
    y1Max = y1.max()
    x = means['max_size_of_transient_memory'].to_numpy()
    # Setting the figure size and resolution
    fig = plt.figure(figsize=(10, 6), dpi=300)
    plt.step(x, y1, color="black",  linewidth=1, linestyle="-", label='Total energy')
    plt.step(x, y2, color="blue",  linewidth=1, linestyle="-", label='Consolidation energy')
    plt.step(x, y3, color="orange",  linewidth=1, linestyle="-", label='Maintenance energy')
    #plt.plot(x, y2, color="green",  linewidth=1, linestyle="-")
    #plt.yscale('log')
    # Setting the boundaries of the figure
    plt.xlim(0, 40)
    plt.ylim(0, 3*(10**6))
    plt.xlabel('Consolidation threshold')  # add x-label
    plt.ylabel('Energy used (a.u.)')  # add y-label
    plt.legend()
    fig.savefig(directoryName+"/figure2b.png", dpi=300)  # save figure
    return fig
