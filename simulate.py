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
                                  env.LEARNING_RATES, env.MAX_SIZES_OF_TRANSIENT_MEMORY, env.MAINTENANCE_COSTS_OF_TRANSIENT_MEMORY,
                                  env.DECAY_TAUS_OF_TRANSIENT_MEMORY))
    TOTAL_SIMULATIONS = len(allSimulations) * len(env.SEEDS)
    return allSimulations


def simulate(simulationNumber, simulationTypeNumber, totalSimulations, cacheAlgorithm, xPatternFeature, nPattern, learningRate, maxSizeOfTransientMemory, maintenanceCostOfTransientMemory, decayTauOfTransientMemory, seed, filePath, directoryName):

    env.setCacheAlgorithm(cacheAlgorithm)
    env.setXPatternFeature(xPatternFeature)
    env.setNPattern(nPattern)
    env.setLearningRate(learningRate)
    env.setMaintenaceCostOfTransientMemory(maintenanceCostOfTransientMemory)
    env.setDecayTauOfTransientMemory(decayTauOfTransientMemory)
    env.setMaxSizeOfTransientMemory(maxSizeOfTransientMemory)
    env.setSeed(seed)
    env.setWeightModel()  # must always be the last thing to be called!

    # TODO: Refactor this so it doesnt have to rerun. If max_sizes is empty, find the optimal.
    if(env.MAX_SIZES_OF_TRANSIENT_MEMORY[0] == 0):
        # This optimisation uses maths that only works if there is no decay!
        env.MAX_SIZE_OF_TRANSIENT_MEMORY = e.calculateTheoreticalOptimalThreshold()
        env.setMaxSizeOfTransientMemory(env.MAX_SIZE_OF_TRANSIENT_MEMORY)
        env.setWeightModel()

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
    simulatedEfficiency = e.calculateSimulatedEfficiency(
        metabolicEnergy, theoreticalMinimumEnergy)
    maintenanceEnergy = e.calculateEnergyFromMaintenance(
        weightsByEpoch)
    consolidationEnergy = e.calculateEnergyFromConsolidations(
        consolidationsByEpoch)
    theoreticalOptimalThreshold = e.calculateTheoreticalOptimalThreshold()
    simulatedOptimalThreshold = e.calculateSimulatedOptimalThreshold(
        epochIndexForConvergence)
    report = {
        simulationNumber:
        {
            # ----
            # Auto-generated values
            # ----

            'timeElapsed': str(time.time() - start),

            # Each simulation type is looped through specified seeds, and describes a particular combination of parameters.
            'simulationTypeNumber': simulationTypeNumber,

            # If an integer, indicates the epoch at which weights converged. If false, learning did not conclude.
            'learningConcludedAtEpoch': epochIndexForConvergence,

            # ----
            # Singular, fixed values that are defined by the user.
            # ----
            'MAX_EPOCH': env.MAX_EPOCHS,
            'ENERGY_EXPONENT': env.ENERGY_EXPONENT,
            'PRESET_SIMULATION': env.PRESET_SIMULATION,
            'WEIGHTS_INITIALISED_AS': str(env.WEIGHTS_INITIALISED_AS),

            # ----
            # Non-fixed parameters that were specified by the user as a list (prefixed with p_ for easier retrieval later)
            # ----
            'p_SEED': str(seed),
            'p_LEARNING_RATE': str(learningRate),
            'p_N_PATTERN': str(nPattern),
            # TODO: notice it is x not n
            'p_X_PATTERN_FEATURE': str(xPatternFeature),
            'p_CACHE_ALGORITHM': str(cacheAlgorithm),
            'p_MAX_SIZE_OF_TRANSIENT_MEMORY': str(maxSizeOfTransientMemory),
            'p_MAINTENANCE_COST_OF_TRANSIENT_MEMORY': str(maintenanceCostOfTransientMemory),
            'p_DECAY_RATE_OF_TRANSIENT_MEMORY': str(decayTauOfTransientMemory),


            #
                # The following report values are divided into values calculated from theory and simulation.
                # They are further divided based on their functions, and whether they apply to scenarios of delay, cache, 
                # both, or neither.
            # 

            # ----
            # Theoretical energy-related values (with cache, without delay)
            # ----
            # Eq. 2
            'theoreticalMinimumEnergy': str(theoreticalMinimumEnergy),

            # sqrt(η**2 (3K/1+cT))
            'theoreticalOptimalThreshold': str(theoreticalOptimalThreshold),

            # M_perc / M_min
            'theoreticalRandomWalkEfficiency': str(theoreticalEfficiency),

            # TODO: Under Eq. 4: N[η2K/θ+13θ(1+cT)]
            'theoreticalTotalEnergyUsingCaching': 0,


            # ----
            # Theoretical energy-related values (with cache, with delay)
            # ----
            # Empty.

            # ----
            # Theoretical energy-related value (without cache, with delay?)
            # ----
            # Empty.
            
            # ----
            # Theoretical energy-related values (without cache, without delay)
            # ----
            # Empty.

            # ----
            # Simulated energy-related values (without cache, without delay)
            # ----

            # Energy used hypothetically with no caching
            'simulatedEnergyUsedByPerceptron': str(metabolicEnergy),

            # ----
            # Simulated energy-related values (with cache, without delay)
            # ----

            # Same as theoretical, but uses T=epochIndexForConvergence
            'simulatedOptimalThreshold': str(simulatedOptimalThreshold),

            # Not explicit in the paper, but is "energy used to learn" - theoreticalMinimumEnergy
            'simulatedEfficiency': str(simulatedEfficiency),

            # M_cons = ∑i∑t|li(t)−li(t−1)|
            'simulatedEnergyForConsolidations': str(consolidationEnergy),

            # M_trans = c∑i∑t|si(t)|.
            'simulatedEnergyForMaintenance': str(maintenanceEnergy),

            # M_cache = M_cons + M_trans
            'simulatedEnergyForConsolidationsAndMaintenance': str(round((consolidationEnergy + maintenanceEnergy), 3)),

            # Calculated performance-related values
            'seenNLL': str(trainNLL)+'%',
            'seenAccuracy': str(trainAccuracy)+'%',
            'seenPrecision': str(trainPrecision)+'%',
            'seenSensitivity': str(trainSensitivity)+'%',
            'seenSpecificity': str(trainSpecificity)+'%',
            'unseenNLL': str(testNLL)+'%',
            'unseenAccuracy': str(testAccuracy)+'%',
            'unseenPrecision': str(testPrecision)+'%',
            'unseenSensitivity': str(testSensitivity)+'%',
            'unseenSpecificity': str(testSpecificity)+'%',
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
