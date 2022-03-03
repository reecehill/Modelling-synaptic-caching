import system_configuration as conf
# Get the parameters either from main parameters.py, or from a previous simulation
if(conf.RUN_SIMULATION == True):
    import parameters as env
else:
    import importlib
    env = importlib.import_module('parameters', conf.RUN_SIMULATION)

import learning as l
import energy as e
import time
from pandas import DataFrame
from shutil import copyfile
from os import path
from itertools import product


def getAllSimulationPossibilities():
    global TOTAL_SIMULATIONS
    global COMMON_N_PATTERNS_X_PATTERN_FEATURES

    # Only take N_PATTERNS found in X_PATTERN_FEATURES, if instructed.
    if(env.ENSURE_N_PATTERNS_EQUALS_X_PATTERNS_FEATURES):
        COMMON_N_PATTERNS_X_PATTERN_FEATURES = list(
            set(env.N_PATTERNS) & set(env.X_PATTERN_FEATURES))
        env.N_PATTERNS = COMMON_N_PATTERNS_X_PATTERN_FEATURES
        env.X_PATTERN_FEATURES = COMMON_N_PATTERNS_X_PATTERN_FEATURES
    else:
        COMMON_N_PATTERNS_X_PATTERN_FEATURES = []

    # Returns a list of all possible permutations of the parameters for looping later.
    allSimulations = list(product(env.CACHE_ALGORITHMS, env.X_PATTERN_FEATURES, env.N_PATTERNS,
                                  env.LEARNING_RATES, env.MAX_SIZES_OF_TRANSIENT_MEMORY, env.MAINTENANCE_COSTS_OF_TRANSIENT_MEMORY))
    TOTAL_SIMULATIONS = len(allSimulations) * len(env.SEEDS)
    return allSimulations


def simulate(simulationNumber, simulationTypeNumber, totalSimulations, cacheAlgorithm, xPatternFeature, nPattern, learningRate, maxSizeOfTransientMemory, maintenanceCostOfTransientMemory, seed, filePath, directoryName):
    env.setCacheAlgorithm(cacheAlgorithm)
    env.setXPatternFeature(xPatternFeature)
    env.setNPattern(nPattern)
    env.setLearningRate(learningRate)
    env.setMaxSizeOfTransientMemory(maxSizeOfTransientMemory)
    env.setMaintenaceCostOfTransientMemory(maintenanceCostOfTransientMemory)
    env.setSeed(seed)
    env.setWeightModel()  # must always be the last thing to be called!
    start = time.time()
    # Generate datasets
    trainingDatasetX, trainingDatasetY, testingDatasetX, testingDatasetY = l.getDatasets()

    # Use training dataset to estimate weights, where each row of weights is a new epoch.
    epochIndexForConvergence, weightsByEpoch, consolidationsByEpoch = l.trainWeights(
        trainingDatasetX, trainingDatasetY)

    # Get performance metrics for seen data
    trainNLL, trainAccuracy, trainPrecision, trainSensitivity, trainSpecificity = l.testWeights(
        trainingDatasetX, trainingDatasetY, weightsByEpoch)
    # Get performance metrics for UNSEEN data
    testNLL, testAccuracy, testPrecision, testSensitivity, testSpecificity = l.testWeights(
        testingDatasetX, testingDatasetY, weightsByEpoch)

    # Calculate energy expenditure
    theoreticalEfficiency = e.calculateTheoreticalEfficiency()
    metabolicEnergy = e.calculateMetabolicEnergy(weightsByEpoch)
    theoreticalMinimumEnergy = e.calculateTheoreticalMinimumEnergy(
        weightsByEpoch)
    efficiency = e.calculateEfficiency(
        metabolicEnergy, theoreticalMinimumEnergy)
    maintenanceEnergy = e.calculateEnergyFromMaintenance(
        weightsByEpoch)
    consolidationEnergy = e.calculateEnergyFromConsolidations(
        consolidationsByEpoch)
    optimalThreshold = e.calculateOptimalThreshold(epochIndexForConvergence)
    report = {
        simulationNumber:
        {
            'time_elapsed': str(time.time() - start),
            'simulationTypeNumber': simulationTypeNumber,
            'seed': seed,
            'learning_rate': learningRate,
            'n_pattern': nPattern,
            'n_pattern_features': xPatternFeature,  # TODO: notice it is x not n
            'max_epochs': env.MAX_EPOCHS,
            'energy_exponent': env.ENERGY_EXPONENT,
            'preset_simulation': env.PRESET_SIMULATION,
            'Learning was complete at epoch #': epochIndexForConvergence,
            'Theoretical: random-walk efficiency': str(theoreticalEfficiency),
            'Simulated: efficiency (m_perc/m_min)': str(efficiency),
            'Simulated-Theoretical efficiency difference': str(round((efficiency - theoreticalEfficiency), 3)),
            'Theoretical: minimum energy for learning': str(theoreticalMinimumEnergy),
            'Simulated: energy actually used by learning': str(metabolicEnergy),
            'Simulated-Theoretical min difference for learning': str(round((metabolicEnergy - theoreticalMinimumEnergy), 3)),
            'Energy expended by simulations for consolidations': str(consolidationEnergy),
            'Energy expended by simulations for maintenance': str(maintenanceEnergy),
            'Energy expended total': str(round((consolidationEnergy + maintenanceEnergy), 3)),
            '(seen) NLL': str(trainNLL)+'%',
            '(seen) Accuracy': str(trainAccuracy)+'%',
            '(seen) Precision': str(trainPrecision)+'%',
            '(seen) Sensitivity': str(trainSensitivity)+'%',
            '(seen) Specificity': str(trainSpecificity)+'%',
            '(unseen) NLL': str(testNLL)+'%',
            '(unseen) Accuracy': str(testAccuracy)+'%',
            '(unseen) Precision': str(testPrecision)+'%',
            '(unseen) Sensitivity': str(testSensitivity)+'%',
            '(unseen) Specificity': str(testSpecificity)+'%',
            'weights_initialised_as': str(env.WEIGHTS_INITIALISED_AS),
            'cache_algorithm': str(env.CACHE_ALGORITHM),
            'max_size_of_transient_memory': str(maxSizeOfTransientMemory),
            'maintenance_cost_of_transient_memory': str(maintenanceCostOfTransientMemory),
            'Optimal threshold': str(optimalThreshold),
        }
    }

    if path.isfile(filePath):
        mode = 'a'
        header = 0
        columns = list(report[simulationNumber].keys())
    else:
        mode = 'w'
        header = list(report[simulationNumber].keys())
        columns = header

    # Copy the parameters file too (so output can be reproduced)
    # TODO: we lazily remove output.csv to get the folder
    copyfile('parameters.py', str(filePath[:-10])+'parameters.py')
    copyfile('__init__.py', str(filePath[:-10])+'__init__.py')

    DataFrame.from_dict(report).transpose().to_csv(
        filePath, header=header, columns=columns, mode=mode)
    print("Simulation "+str(simulationNumber) +
          " of "+str(totalSimulations))
    timeElapsed = time.time() - start
    print("Finished simulation (time elapsed: "+str(timeElapsed))
