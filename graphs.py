import matplotlib.pyplot as plt
from numpy import nan
import pandas as pd


def getFixedParametersAndGenerateTitle(data, columnsToVary):
    # Get columns for simulation variables that can be varied in parameters, but ought to be fixed for this figure.
    data = data.where(data['learningConcludedAtEpoch'] != False).dropna()
    parametersToFix = [col for col in data if (
        (col.startswith('p_')) & (col not in columnsToVary))]
    #dataToFix = data[columnsToFix].drop_duplicates().to_dict(orient='records')
    simulationsToGraph = data[parametersToFix].drop_duplicates().to_numpy()
    titles = []
    for graphIndex, simulationParameters in enumerate(simulationsToGraph):
        title = ["Parameters in use:"]
        for parameterIndex, parameterValue in enumerate(simulationParameters):
            title.append(str(parametersToFix[parameterIndex])+"="+str(parameterValue))
        titles.append('\n'.join(title))
    dataGroupedByFixedParameters = data.groupby(parametersToFix)
    return parametersToFix, titles, dataGroupedByFixedParameters

    
def showFigures():
    plt.show()

def makeFigure1c(directoryName):
    data = pd.read_csv(directoryName+'/output.csv',
                       delimiter=',', na_values=['inf', 'nan']).dropna()

    parametersToFix, titles, dataGroupedByFixedParameters = getFixedParametersAndGenerateTitle(
        data, columnsToVary=['p_N_PATTERN', 'p_X_PATTERN_FEATURE'])
    if(len(dataGroupedByFixedParameters) > 10):
        print("Skipping making figure 1c as one parameter that should be fixed has more than 10 values.")
        return

    for graphIndex, graphData in enumerate(dataGroupedByFixedParameters.groups):
        for parameterId, parameterToFix in enumerate(graphData):
            data = data.where(data[parametersToFix[parameterId]] == parameterToFix)

        data = data.where(data['learningConcludedAtEpoch'] != False).sort_values(
            'simulationTypeNumber').groupby('simulationTypeNumber')
        means = data[list(['p_N_PATTERN', 'p_X_PATTERN_FEATURE',
                        'theoreticalMinimumEnergy',
                        'simulatedEnergyUsedByPerceptron'])
                    ].mean(numeric_only=True).sort_values('p_N_PATTERN')

        y1 = means['simulatedEnergyUsedByPerceptron'].to_numpy()
        y2 = means['theoreticalMinimumEnergy'].to_numpy()
        y1Min = y1.min() if y1.min() > 0 else 0
        y1Max = y1.max()
        x = means['p_N_PATTERN'].to_numpy(
        ) / means['p_X_PATTERN_FEATURE'].to_numpy()
        # Setting the figure size and resolution
        fig = plt.figure(figsize=(10, 6), dpi=300)
        plt.title(titles[graphIndex], fontsize=8)
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
    data = pd.read_csv(directoryName+'/output.csv',
                       delimiter=',', na_values=['inf', 'nan']).dropna()

    parametersToFix, titles, dataGroupedByFixedParameters = getFixedParametersAndGenerateTitle(
        data, columnsToVary=['p_N_PATTERN', 'p_X_PATTERN_FEATURE'])
    if(len(dataGroupedByFixedParameters) > 10):
        print("Skipping making figure 1c as one parameter that should be fixed has more than 10 values.")
        return

    for graphIndex, graphData in enumerate(dataGroupedByFixedParameters.groups):
        for parameterId, parameterToFix in enumerate(graphData):
            data = data.where(data[parametersToFix[parameterId]] == parameterToFix)
        data = data.where(data['learningConcludedAtEpoch'] != False).sort_values(
            'simulationTypeNumber').groupby('simulationTypeNumber')
        means = data[list(['p_N_PATTERN', 'p_X_PATTERN_FEATURE',
                        'theoreticalRandomWalkEfficiency',
                        'simulatedEfficiency'])
                    ].mean(numeric_only=True).sort_values('p_N_PATTERN')

        y1 = means['theoreticalRandomWalkEfficiency'].to_numpy()
        y2 = means['simulatedEfficiency'].to_numpy()
        y1Min = y1.min() if y1.min() > 0 else 0
        y1Max = y1.max()
        x = means['p_N_PATTERN'].to_numpy(
        ) / means['p_X_PATTERN_FEATURE'].to_numpy()
        # Setting the figure size and resolution
        fig = plt.figure(figsize=(10, 6), dpi=300)
        plt.title(titles[graphIndex], fontsize=8)
        plt.plot(x, y1, color="blue",  linewidth=1, linestyle="-")
        plt.plot(x, y2, color="black",  linewidth=1, linestyle="-")
        plt.yscale('log')
        # Setting the boundaries of the figure
        plt.xlim(0, 2)
        #plt.ylim(10**3, 10**9)
        plt.xlabel('number of patterns / number of synapses')  # add x-label
        plt.ylabel('Inefficiency')  # add y-label
        fig.savefig(directoryName+"/figure1d.png", dpi=125)  # save figure
        return fig


def makeFigure2b(directoryName):
    data = pd.read_csv(directoryName+'/output.csv',
                       delimiter=',', na_values=['inf', 'nan']).dropna()

    parametersToFix, titles, dataGroupedByFixedParameters = getFixedParametersAndGenerateTitle(
        data, columnsToVary=['p_MAX_SIZE_OF_TRANSIENT_MEMORY'])
    if(len(dataGroupedByFixedParameters) > 10):
        print("Skipping making figure 1c as one parameter that should be fixed has more than 10 values.")
        return

    for graphIndex, graphData in enumerate(dataGroupedByFixedParameters.groups):
        for parameterId, parameterToFix in enumerate(graphData):
            data = data.where(data[parametersToFix[parameterId]] == parameterToFix)

        data = data.where(data['learningConcludedAtEpoch'] != False).sort_values(
            'p_MAX_SIZE_OF_TRANSIENT_MEMORY').groupby('simulationTypeNumber')
        means = data[list(['p_MAX_SIZE_OF_TRANSIENT_MEMORY',
                        'simulatedEnergyUsedByPerceptron',
                        'simulatedEnergyForConsolidations',
                        'simulatedEnergyForMaintenance'])
                    ].mean(numeric_only=True).sort_values('p_MAX_SIZE_OF_TRANSIENT_MEMORY')

        y_consolidations = means['simulatedEnergyForConsolidations'].to_numpy()
        y_maintenance = means['simulatedEnergyForMaintenance'].to_numpy()
        y_total = y_maintenance + y_consolidations
        x = means['p_MAX_SIZE_OF_TRANSIENT_MEMORY'].to_numpy()
        # Setting the figure size and resolution
        fig = plt.figure(figsize=(10, 6), dpi=300)
        plt.title(titles[graphIndex], fontsize=8)
        plt.step(x, y_total, color="black",  linewidth=1,
                linestyle="-", label='Total energy')
        plt.step(x, y_consolidations, color="blue",  linewidth=1,
                linestyle="-", label='Consolidation energy')
        plt.step(x, y_maintenance, color="orange",  linewidth=1,
                linestyle="-", label='Maintenance energy')

        # plt.yscale('log')
        # Setting the boundaries of the figure
        #plt.ylim(0, 3*10**6)
        plt.xlim(0, 40)
        plt.xlabel('Consolidation threshold')  # add x-label
        plt.ylabel('Energy used (a.u.)')  # add y-label
        plt.legend()
        fig.savefig(directoryName+"/figure2b.png", dpi=125)  # save figure
        return fig


def makeFigure2c(directoryName):
    data = pd.read_csv(directoryName+'/output.csv',
                       delimiter=',', na_values=['inf', 'nan']).dropna()

    parametersToFix, titles, dataGroupedByFixedParameters = getFixedParametersAndGenerateTitle(
        data, columnsToVary=['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY'])
    if(len(dataGroupedByFixedParameters) > 10):
        print("Skipping making figure 1c as one parameter that should be fixed has more than 10 values.")
        return

    for graphIndex, graphData in enumerate(dataGroupedByFixedParameters.groups):
        for parameterId, parameterToFix in enumerate(graphData):
            data = data.where(
                data[parametersToFix[parameterId]] == parameterToFix)
        data = data.where(data['learningConcludedAtEpoch'] != False).sort_values(
            'simulationTypeNumber').groupby('simulationTypeNumber')
        means = data[list(['theoreticalOptimalThreshold',
                        'p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY'])
                    ].mean(numeric_only=True).sort_values('p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY')

        y1 = means['theoreticalOptimalThreshold'].to_numpy()
        x = means['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY'].to_numpy()
        # Setting the figure size and resolution
        fig = plt.figure(figsize=(10, 6), dpi=300)
        plt.title(titles[graphIndex], fontsize=8)
        plt.step(x, y1,  linewidth=1, linestyle="-")

        # Setting the boundaries of the figure
        plt.xlim(0, 0.1)
        #plt.ylim(10**3, 10**9)
        plt.xlabel('Cost of transient plasticity')  # add x-label
        plt.ylabel('theoreticalOptimalThreshold')  # add y-label
        fig.savefig(directoryName+"/figure2c-"+str(graphIndex) +
                    ".png", dpi=125)  # save figure
        return fig


def makeFigure3a(directoryName):
    data = pd.read_csv(directoryName+'/output.csv',
                       delimiter=',', na_values=['inf', 'nan']).dropna()

    parametersToFix, titles, dataGroupedByFixedParameters = getFixedParametersAndGenerateTitle(
        data, columnsToVary=['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY', 'p_DECAY_RATE_OF_TRANSIENT_MEMORY'])
    if(len(dataGroupedByFixedParameters) > 10):
        print("Skipping making figure 1c as one parameter that should be fixed has more than 10 values.")
        return

    for graphIndex, graphData in enumerate(dataGroupedByFixedParameters.groups):
        fixedData = data
        for fixedParameterId, fixedParameterValue in enumerate(graphData):
            fixedData = fixedData.where(
                fixedData[parametersToFix[fixedParameterId]] == fixedParameterValue)
        usedCostsOfTransientMemory = fixedData['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY'].unique()
        
        fig = plt.figure(figsize=(10, 6), dpi=300)
        plt.title(titles[graphIndex], fontsize=8)
        for costOfTransientMemory in usedCostsOfTransientMemory:
            filteredData = fixedData.where(fixedData['learningConcludedAtEpoch'] != False)\
                .where(fixedData['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY'] == costOfTransientMemory)
            if(filteredData.empty):
                # Empty dataset resulted, probably because the learning was finished for this set.
                continue
            else:
                groupedData = filteredData.groupby('simulationTypeNumber')
            means = groupedData[list(['simulatedEnergyForConsolidationsAndMaintenance',
                                    'p_DECAY_RATE_OF_TRANSIENT_MEMORY', ])
                                ].mean(numeric_only=True).sort_values('p_DECAY_RATE_OF_TRANSIENT_MEMORY')

            x = means['p_DECAY_RATE_OF_TRANSIENT_MEMORY'].to_numpy()
            y = means['simulatedEnergyForConsolidationsAndMaintenance'].to_numpy()
            if(str(costOfTransientMemory) == 'nan'):
                print('k')
            plt.plot(x, y, linewidth=1, linestyle="-",
                    label='c='+str(costOfTransientMemory))

            # plt.yscale('log')
            # Setting the boundaries of the figure
            #plt.xlim(0, 0.004)
            #plt.ylim(10**3, 10**9)
            plt.xlabel('Decay rate')  # add x-label
            plt.ylabel('Energy')  # add y-label
            plt.legend()
            fig.savefig(directoryName+"/figure3a-"+str(graphIndex)+".png", dpi=125)  # save figure



def makeFigure3b(directoryName):
    data = pd.read_csv(directoryName+'/output.csv',
                       delimiter=',', na_values=['inf', 'nan']).dropna()

    parametersToFix, titles, dataGroupedByFixedParameters = getFixedParametersAndGenerateTitle(
        data, columnsToVary=['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY', 'p_DECAY_RATE_OF_TRANSIENT_MEMORY'])
    if(len(dataGroupedByFixedParameters) > 10):
        print("Skipping making figure 1c as one parameter that should be fixed has more than 10 values.")
        return

    for graphIndex, graphData in enumerate(dataGroupedByFixedParameters.groups):
        for parameterId, parameterToFix in enumerate(graphData):
            data = data.where(
                data[parametersToFix[parameterId]] == parameterToFix)
        usedCostsOfTransientMemory = data['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY'].unique()
        fig = plt.figure(figsize=(10, 6), dpi=300)
        plt.title(titles[graphIndex], fontsize=8)
        for costOfTransientMemory in usedCostsOfTransientMemory:
            groupedData = data.where(data['learningConcludedAtEpoch'] != False)\
                .where(data['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY'] == costOfTransientMemory).groupby('simulationTypeNumber')

            means = groupedData[list(['simulatedOptimalThreshold',
                                    'p_DECAY_RATE_OF_TRANSIENT_MEMORY', ])
                                ].mean(numeric_only=True).sort_values('p_DECAY_RATE_OF_TRANSIENT_MEMORY')

            x = means['p_DECAY_RATE_OF_TRANSIENT_MEMORY'].to_numpy()
            y = means['simulatedOptimalThreshold'].to_numpy()

            plt.plot(x, y, linewidth=1, linestyle="-",
                    label='c='+str(costOfTransientMemory))

        plt.xlabel('Decay rate')  # add x-label
        plt.ylabel('Optimal threshold')  # add y-label
        plt.legend()
        fig.savefig(directoryName+"/figure3b.png", dpi=125)  # save figure
        return fig


def makeFigure4b(directoryName):
    # Figure 4b does not work because the ideal consolidation thresholds must be calculated for each cache algorithm and maintenance cost.
    data = pd.read_csv(directoryName+'/output.csv',
                       delimiter=',', na_values=['inf', 'nan']).dropna()

    parametersToFix, titles, dataGroupedByFixedParameters = getFixedParametersAndGenerateTitle(
        data, columnsToVary=['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY', 'p_CACHE_ALGORITHM'])
    if(len(dataGroupedByFixedParameters) > 10):
        print("Skipping making figure 1c as one parameter that should be fixed has more than 10 values.")
        return

    for graphIndex, graphData in enumerate(dataGroupedByFixedParameters.groups):
        for parameterId, parameterToFix in enumerate(graphData):
            data = data.where(data[parametersToFix[parameterId]] == parameterToFix)
        fig = plt.figure(figsize=(10, 6), dpi=300)
        plt.title(titles[graphIndex], fontsize=8)

        usedCacheAlgorithms = data['p_CACHE_ALGORITHM'].unique()
        for cacheAlgorithm in usedCacheAlgorithms:
            groupedData = data.where(data['learningConcludedAtEpoch'] != False)\
                .where(data['p_CACHE_ALGORITHM'] == cacheAlgorithm).groupby('simulationTypeNumber')
            means = groupedData[['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY',
                                'simulatedEnergyForConsolidations',
                                'simulatedEnergyForMaintenance']].mean(numeric_only=True).sort_values('p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY')
            x = means['p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY'].to_numpy()
            y = means['simulatedEnergyForConsolidations'].to_numpy(
            ) + means['simulatedEnergyForMaintenance'].to_numpy()
            plt.plot(x, y, linewidth=1, linestyle="-", label=cacheAlgorithm)

        # plt.yscale('log')
        # Setting the boundaries of the figure
        plt.xlim(0, 0.1)
        #plt.ylim(10**3, 10**9)
        plt.xlabel('Cost of transient plasticity')  # add x-label
        plt.ylabel('Energy')  # add y-label
        plt.legend()
        fig.savefig(directoryName+"/figure4b.png", dpi=300)  # save figure
        return fig
