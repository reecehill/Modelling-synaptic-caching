import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from os import mkdir
# This code will combine the data collected from simulations that differ only by the SYNAPSE_MEMORY_TYPES used.
# It produces a graph comparable to Fig 2b.

FOLDER_OF_CSV_1 = '20220309-134206'
FOLDER_OF_CSV_2 = '20220309-134554'


def makeFigure2bModified(directoryName):
    data_1 = pd.read_csv(FOLDER_OF_CSV_1+'/output.csv', delimiter=',',
                         na_values=['inf', 'nan']).dropna()

    data_2 = pd.read_csv(FOLDER_OF_CSV_2+'/output.csv', delimiter=',',
                         na_values=['inf', 'nan']).dropna()
    data = pd.concat([data_1, data_2])
    
    # If optimal thresholds have been used, replace with the optimal threshold rather than the word itself.
    rowsWithOptimalThresholdSet = data[data['max_size_of_transient_memory'] == 'optimal']
    rowsWithOptimalThresholdSet['max_size_of_transient_memory'] = rowsWithOptimalThresholdSet['Optimal threshold']
    data[data['max_size_of_transient_memory'] == 'optimal'] = rowsWithOptimalThresholdSet

    usedNumberOfMemoryTypes = data['number of memory types'].unique()
    fig = plt.figure(figsize=(10, 6), dpi=125)

    for usedNumberOfMemoryType in usedNumberOfMemoryTypes:
        groupedData = data.where(data['Learning was complete at epoch #'] != False).where(data['number of memory types'] == usedNumberOfMemoryType).groupby('simulationTypeNumber')
        means = groupedData[list(['max_size_of_transient_memory',
                          'Simulated: energy actually used by learning',
                           'Energy expended by simulations for consolidations',
                           'Energy expended by simulations for maintenance',
                           'Energy expended by simulations for maintenance (before thr)'])
                     ].mean().sort_values('max_size_of_transient_memory')

        y_consolidations = means['Energy expended by simulations for consolidations'].to_numpy(
        )
        y_maintenance = means['Energy expended by simulations for maintenance'].to_numpy(
        )
        y_maintenance_before_thr = means['Energy expended by simulations for maintenance (before thr)'].to_numpy(
        )
        y_total = y_maintenance + y_consolidations
        y_total_thr = y_maintenance + y_maintenance_before_thr + y_consolidations
        x = means['max_size_of_transient_memory'].to_numpy()
        # Setting the figure size and resolution
        #plt.step(x, y_total, color="black",  linewidth=1, linestyle="-", label='Total energy')
        #plt.step(x, y_consolidations, color="blue",  linewidth=1,
        #         linestyle="-", label='Consolidation energy')
        #plt.step(x, y_maintenance, color="orange",  linewidth=1,
        #         linestyle="-", label='Maintenance energy')
        #plt.step(x, y_total, color="black",  linewidth=1,
        #         linestyle="-", label='Total energy')
        #plt.step(x, y_maintenance+y_maintenance_before_thr, color="orange", linewidth=1, linestyle='--', label='Maintenance energy (corrected)')
        plt.step(x, y_total_thr,  linewidth=1, linestyle="-", label='Total energy (corrected): '+str(usedNumberOfMemoryType)+' memory types')

    # plt.yscale('log')
    # Setting the boundaries of the figure
    #plt.ylim(0, 3*10**6)
    plt.xlim(0, 40)
    plt.xlabel('Consolidation threshold')  # add x-label
    plt.ylabel('Energy used (a.u.)')  # add y-label
    plt.legend()
    fig.savefig(directoryName+"/figure2ba.png", dpi=125)  # save figure
    return fig

directoryName = str(datetime.now().strftime("%Y%m%d-%H%M%S"))+'-combined'
mkdir(directoryName)

fig = makeFigure2bModified(directoryName)
