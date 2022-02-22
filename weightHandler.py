
import numpy as np
import parameters as env


def validateParameters():
    global PERCENTAGE_INHIBITORY_WEIGHTS
    print("Skipped validation of parameters (learning)...")


def getSummedWeightsByType(weightsAtTimeT):
    # Sum all types of weights per neurone (i.e. consolidated + transient) to get neurone's current value.
    summedWeightsByType = np.sum(weightsAtTimeT, axis=1)
    return summedWeightsByType


def prepareWeights(trainingDatasetX):
    #minWeight = -1
    #maxWeight = 0
    #initialWeights = (maxWeight - minWeight) * np.random.rand(len(trainingDatasetX[0])) + minWeight
    #initialWeights = [0.0 for i in range(len(trainingDatasetX[0]))]

    # !-- SET +ve/-ve WEIGHTS TO TRY AND MATCH PROPORTIONS FOUND IN BRAIN
    # https://www.brainfacts.org/brain-anatomy-and-function/cells-and-circuits/2021/how-inhibitory-neurons-shape-the-brains-code-100621

    # Generate matrix of following structure:
    # time_1 : [ weight_1, weight_2, ..., weight_(N_WEIGHTS)]
    # ...
    # time_(MAX_EPOCHS) : [ weight_1, weight_2, ..., weight_(N_WEIGHTS)]
    # -----------
    # Where the size of weight_(N_WEIGHTS) is determined by number of MEMORY_MEMORY_TYPES:
    # weight_(N_WEIGHTS) = [ value_consolidated, value_transient ]
    nMemoryTypes = len(env.WEIGHT_MEMORY_TYPES)

    weightsByTime = np.empty((env.MAX_EPOCHS, env.N_WEIGHTS, nMemoryTypes))
    initialWeights = np.empty((0, nMemoryTypes))
    # Use template model of weights (see parameters.py) to create new matrix of random weights
    for neuroneTypeName, neuroneTypeData in env.WEIGHT_MODEL.items():
        nWeights = int(round(
            (neuroneTypeData['percentage_quantity_of_neurones']/100) * env.N_WEIGHTS, 0))
        randomWeights = env.RANDOM_GENERATOR.uniform(
            low=float(neuroneTypeData['min']),
            high=float(neuroneTypeData['max']),
            size=nWeights).reshape(nWeights, 1)
        zeroWeights = np.zeros(shape=(nWeights, nMemoryTypes-1))
        initialWeightsToAdd = np.hstack((randomWeights, zeroWeights))
        initialWeights = np.append(initialWeights, initialWeightsToAdd, axis=0)

    # Due to rounding of nWeights, the matrix may be of incorrect shape. Remove/add row(s) to suit.
    sizeDifference = len(initialWeights) - (env.N_WEIGHTS)
    if(sizeDifference > 0):
    # Matrix is too large, so remove last rows.
        initialWeights = initialWeights[:-sizeDifference, :]
    elif(sizeDifference < 0):
        initialWeights = np.hstack([initialWeights, env.RANDOM_GENERATOR.uniform(
            low=float(neuroneTypeData['min']),
            high=float(neuroneTypeData['max']),
            size=(abs(sizeDifference)))])

    # Set weights at t_0 to randomly shuffled initialWeights (so that -ve and +ve weights are shuffled, if present)
    weightsByTime[0] = initialWeights
    env.RANDOM_GENERATOR.shuffle(weightsByTime[0], axis=0)

    initialWeightsSummedByType = getSummedWeightsByType(weightsAtTimeT=weightsByTime[0])
    indexesOfWeightsByNeuronalType = {
        # Find the indexes for weights that begin positive/negative.
        "excitatory": np.where(initialWeightsSummedByType >= 0),
        "inhibitory": np.where(initialWeightsSummedByType < 0)
    }

    return weightsByTime, indexesOfWeightsByNeuronalType


def prepareConsolidationEvents(weightsByTimeShape):
    # Consolidation describes the movement of weight out of one memory type, into another (e.g. transient -> consolidated).
    # This script only allows memory to move "downwards", according to the value assigned to the memory type.
    # For most cases, consolidation = 0 and transient = 1
    consolidationsByTime = np.zeros(weightsByTimeShape)
    return consolidationsByTime

def consolidateWeightsAboveThreshold(newWeightsAtTimeT, consolidationsAtTimeT):
    # Get the different memory types, starting with the highest valued (as memory moves down the chain, towards consolidation)
    for memoryTypeId in sorted(env.WEIGHT_MEMORY_TYPES, reverse=True):
        # Skip "consolidated" memory types (or memory types without limit)
        if(memoryTypeId == 0 or env.WEIGHT_MEMORY_TYPES[memoryTypeId]['memory_size'] == False):
            continue

        # Get the indexes of weights that have exceeded their memory limit.
        indexesOfWeightsAboveThreshold = np.where(
            abs(newWeightsAtTimeT[:, memoryTypeId]) >= abs(env.WEIGHT_MEMORY_TYPES[memoryTypeId]['memory_size']))[0]
        
        if(len(indexesOfWeightsAboveThreshold) > 0):
            # Add changes to consolidationEvents matrix (which stores the amount each memory will change by this in time step)
            consolidationsAtTimeT[indexesOfWeightsAboveThreshold, memoryTypeId -
                                  1] = newWeightsAtTimeT[indexesOfWeightsAboveThreshold, memoryTypeId]

            # Submit consolidations
            newWeightsAtTimeT = newWeightsAtTimeT + consolidationsAtTimeT

            # Reset values to zero if it was consolidated.
            newWeightsAtTimeT[indexesOfWeightsAboveThreshold, memoryTypeId] = 0

    return newWeightsAtTimeT, consolidationsAtTimeT


def updateWeights(weightsAtTimeT, deltaWeights, neuronalTypes, consolidationsAtTimeT):
    # TODO: this code will fail if weights are not defined as excitatory or inhibitory.
    newWeights = np.zeros(weightsAtTimeT.shape)
    for neuronalType, indexesOfWeightsByNeuronalType in neuronalTypes.items():
        if(env.NEURONES_CAN_CHANGE_TYPE_MID_SIMULATION):
            a_min = abs(np.inf) * -1
            a_max = np.inf
        elif (neuronalType == 'excitatory'):
            a_min = 0.0
            a_max = np.inf
        elif(neuronalType == 'inhibitory'):
            a_min = abs(np.inf) * -1
            a_max = 0.0

        newWeights[indexesOfWeightsByNeuronalType] = weightsAtTimeT[indexesOfWeightsByNeuronalType] + \
            deltaWeights[indexesOfWeightsByNeuronalType]
        newWeights, consolidationsAtTimeT = consolidateWeightsAboveThreshold(
            newWeights, consolidationsAtTimeT)

        try:
            weightsAtTimeT = np.clip(a=newWeights, a_min=a_min, a_max=a_max)
        except:
            print("Neurones must be allowed to switch type (i.e., inhibitory->excitatory) if there is only one neurone type. You may #want to check the NEURONES_TYPES_BEGIN_EITHER_INHIBITORY_OR_EXCITATORY parameter.")
    return weightsAtTimeT, consolidationsAtTimeT
