import parameters as env
import numpy as np
def validateParameters():
    global PERCENTAGE_INHIBITORY_WEIGHTS
    print("Skipped validation of parameters (learning)...")


def getSummedWeightsByType(weightsAtTimeT):
    # Sum all types of weights per synapse (i.e. consolidated + transient) to get synapse's current value.
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
    nMemoryTypes = len(env.SYNAPSE_MEMORY_TYPES)

    weightsByTime = np.zeros((env.MAX_EPOCHS, env.N_WEIGHTS, nMemoryTypes))
    initialWeights = np.empty((0, nMemoryTypes))
    # Use template model of weights (see parameters.py) to create new matrix of random weights
    for synapseTypeName, synapseTypeData in env.WEIGHT_MODEL.items():
        nWeights = int(round(
            (synapseTypeData['percentage_quantity_of_synapses']/100) * env.N_WEIGHTS, 0)+1)


        if(env.WEIGHTS_INITIALISED_AS == 'zeros'):
            randomWeights = np.zeros(shape=(nWeights, 1)) #Represents the initial conditions, applied only to the top most memory Type (i.e. consolidarted)

        elif(env.WEIGHTS_INITIALISED_AS == 'uniform'):
            randomWeights = env.RANDOM_GENERATOR.uniform(low=float(synapseTypeData['min']), high=float(
                synapseTypeData['max']), size=(nWeights, 1))

        elif(env.WEIGHTS_INITIALISED_AS == 'lognormal'):
            randomWeights = env.RANDOM_GENERATOR.lognormal(mean=0, sigma=1, size=(nWeights, 1))
            if(synapseTypeData['max'] == 0):
                randomWeights = randomWeights * -1

        # Set all memory types that are below the top-most (i.e., consolidated) to zero.
        zeroWeights = np.zeros(shape=(nWeights, nMemoryTypes-1))

        # Combine initial conditions with zeroWeights.
        initialWeightsToAdd = np.hstack((randomWeights, zeroWeights))
        initialWeights = np.append(initialWeights, initialWeightsToAdd, axis=0)

    # Due to rounding of nWeights, the matrix may be of incorrect shape. Remove row(s) to suit.
    sizeDifference = len(initialWeights) - (env.N_WEIGHTS)
    if(sizeDifference > 0):
        # Matrix is too large, so remove last rows.
        initialWeights = initialWeights[:-sizeDifference, :]

    # Set weights at t_0 to randomly shuffled initialWeights (so that -ve and +ve weights are shuffled, if present)
    weightsByTime[0] = initialWeights
    env.RANDOM_GENERATOR.shuffle(weightsByTime[0], axis=0)

    initialWeightsSummedByType = getSummedWeightsByType(
        weightsAtTimeT=weightsByTime[0])
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

def consolidateAllWeights(newWeightsAtTimeT, consolidationsAtTimeT):
    # Called when learning has finished.
    # Loop through the different synapses simulated (e.g. excitatory, inhibitory)
    for synapseTypeId, synapseType in env.WEIGHT_SYNAPSE_TYPES.items():
        # Get the different memory types, starting with the highest valued (as memory moves down the chain, towards consolidation)
        for memoryTypeId, memoryType in sorted(synapseType['memoryTypes'].items(), reverse=True):
            # Skip "consolidated" memory types (or memory types without limit)
            if(memoryTypeId == 0):
                continue
            if((isinstance(memoryType['memory_size'], bool)) & (memoryType['memory_size'] == False)):
                print("Memory size set to False. Therefore we assume it has not limit.")
                continue
            # Add changes to consolidationEvents matrix (which stores the amount each memory will change by this in time step)
            consolidationsAtTimeT[:, memoryTypeId -
                                    1] = newWeightsAtTimeT[:, memoryTypeId]
            # Submit consolidations
            newWeightsAtTimeT = newWeightsAtTimeT + consolidationsAtTimeT
            # Reset all values for memory type to zero as they have been consolidated.
            newWeightsAtTimeT[:, memoryTypeId] = 0
    return newWeightsAtTimeT, consolidationsAtTimeT

def consolidateWeightsAboveThreshold(newWeightsAtTimeT, weightsAtTimeTMinus1, consolidationsAtTimeT):
    # Loop through the different synapses simulated (e.g. excitatory, inhibitory)
    for synapseTypeId, synapseType in env.WEIGHT_SYNAPSE_TYPES.items():
        # Get the different memory types, starting with the highest valued (as memory moves down the chain, towards consolidation)
        for memoryTypeId, memoryType in sorted(synapseType['memoryTypes'].items(), reverse=True):
            # Skip "consolidated" memory types (or memory types without limit)
            if(memoryTypeId == 0):
                continue
            if((isinstance(memoryType['memory_size'], bool)) & (memoryType['memory_size'] == False)):
                print("Memory size set to False. Therefore we assume it has not limit.")
                continue

            # Get the indexes of weights that have exceeded their memory limit.
            indexesOfWeightsConsolidated = np.where(
                np.abs(newWeightsAtTimeT[:, memoryTypeId]) > np.abs(memoryType['memory_size']))[0]
            
            if(env.ALSO_CALCULATE_ENERGY_TO_REACH_THRESHOLD):
                weightsAboveThresholdAtTimeT = newWeightsAtTimeT[:,
                                                                memoryTypeId][indexesOfWeightsConsolidated]
                weightsAboveThresholdAtTimeMinus1 = weightsAtTimeTMinus1[:,
                                                                      memoryTypeId][indexesOfWeightsConsolidated]
                env.ENERGY_USED_TO_REACH_THRESHOLD_TALLY += np.sum(np.abs(np.subtract(weightsAboveThresholdAtTimeT,weightsAboveThresholdAtTimeMinus1)))

            # Update weights, moving them up the chain (consolidating) and setting the transient's to zero.
            newWeightsAtTimeT, consolidationsAtTimeT = updateWeightsAccordingToAlgorithm(
                newWeightsAtTimeT, consolidationsAtTimeT, memoryType, memoryTypeId, indexesOfWeightsConsolidated)
            

    return newWeightsAtTimeT, consolidationsAtTimeT


def updateWeightsAccordingToAlgorithm(newWeightsAtTimeT, consolidationsAtTimeT, memoryType, memoryTypeId, indexesOfWeightsConsolidated):
    # Only consolidate individual synapses that have met threshold.
    if(env.CACHE_ALGORITHM == 'local-local'):
        if(len(indexesOfWeightsConsolidated) > 0):
            # Add changes to consolidationEvents matrix (which stores the amount each memory will change by this in time step)
            consolidationsAtTimeT[indexesOfWeightsConsolidated, memoryTypeId -
                                1] = newWeightsAtTimeT[indexesOfWeightsConsolidated, memoryTypeId]
            # Submit consolidations
            newWeightsAtTimeT[indexesOfWeightsConsolidated,
                                memoryTypeId - 1] = newWeightsAtTimeT[indexesOfWeightsConsolidated, memoryTypeId-1] + consolidationsAtTimeT[indexesOfWeightsConsolidated, memoryTypeId - 1]
            # Reset values to zero if consolidated.
            newWeightsAtTimeT[indexesOfWeightsConsolidated,
                memoryTypeId] = 0
    elif(env.CACHE_ALGORITHM == 'local-global'): # Consolidate all synapses once one synapse has hit threshold.
        if(len(indexesOfWeightsConsolidated) > 0):
            negativeWeightIndices = np.where(newWeightsAtTimeT[:, memoryTypeId] < 0)[0]
            amountsFromThreshold = np.abs(newWeightsAtTimeT[:, memoryTypeId]) - np.abs(memoryType['memory_size'])
            amountsFromThreshold[negativeWeightIndices] = amountsFromThreshold[negativeWeightIndices] * -1

            if(env.ONLY_CONSOLIDATE_AMOUNT_ABOVE_THRESHOLD):
                # Add changes to consolidationEvents matrix (which stores the amount each memory has changed by in this time step)
                consolidationsAtTimeT[:, memoryTypeId -
                                    1] = amountsFromThreshold
                # Submit consolidations
                newWeightsAtTimeT[:, memoryTypeId -
                                    1] = newWeightsAtTimeT[:, memoryTypeId - 1] + amountsFromThreshold
                # Reset all values for memory type to the amount remaining after that which is above the threshold has been consolidated (i.e. reset to threshold level).
                newWeightsAtTimeT[:,
                                    memoryTypeId] = newWeightsAtTimeT[:, memoryTypeId] - amountsFromThreshold
            else:
                # Add changes to consolidationEvents matrix (which stores the amount each memory has changed by in this time step)
                consolidationsAtTimeT[:, memoryTypeId -
                                    1] = newWeightsAtTimeT[:, memoryTypeId]
                # Submit consolidations
                newWeightsAtTimeT[:, memoryTypeId -
                                1] = newWeightsAtTimeT[:, memoryTypeId - 1] + \
                                    consolidationsAtTimeT[:, memoryTypeId - 1]
                # Reset all values for memory type to zero as they have been consolidated.
                newWeightsAtTimeT[:, memoryTypeId] = 0
    elif(env.CACHE_ALGORITHM == 'global-global'):
        if(np.sum(abs(newWeightsAtTimeT[:, memoryTypeId])) >= abs(memoryType['memory_size'])):
                # The sum of the values of this memory type have exceeded their threshold.

            # Add changes to consolidationEvents matrix (which stores the amount each memory has changed by in this time step)
            consolidationsAtTimeT[:, memoryTypeId -
                                    1] = newWeightsAtTimeT[:, memoryTypeId]
            # Submit consolidations
            newWeightsAtTimeT[:, memoryTypeId-1] = newWeightsAtTimeT[:, memoryTypeId - 1] + \
                consolidationsAtTimeT[:, memoryTypeId - 1]
            # Reset all values for memory type to zero as they have been consolidated.
            newWeightsAtTimeT[:, memoryTypeId] = 0
    else:
        raise ValueError('Incorrect CACHING_ALGORITHM specified. Please refer to the comments in parameters.py')

    return newWeightsAtTimeT, consolidationsAtTimeT


def updateWeights(weightsAtTimeT, deltaWeights, neuronalTypes):
    newWeights = np.zeros(weightsAtTimeT.shape)
    for neuronalType, indexesOfWeightsByNeuronalType in neuronalTypes.items():
        if(env.SYNAPSES_CAN_CHANGE_TYPE_MID_SIMULATION):
            a_min = abs(np.inf) * -1
            a_max = np.inf
        elif (neuronalType == 'excitatory'):
            a_min = 0.0
            a_max = np.inf
        elif(neuronalType == 'inhibitory'):
            a_min = abs(np.inf) * -1
            a_max = 0.0
        else:
            raise Exception('Synapses must either be able to change to type (i.e., have no limits), or be labelled either excitatory or inhibitory.')

        newWeights[indexesOfWeightsByNeuronalType] = weightsAtTimeT[indexesOfWeightsByNeuronalType] + \
            deltaWeights[indexesOfWeightsByNeuronalType]

        try:
            weightsAtTimeT = np.clip(a=newWeights, a_min=a_min, a_max=a_max)
        except:
            print("Synapses must be allowed to switch type (i.e., inhibitory->excitatory) if there is only one synapse type. You may #want to check the SYNAPSE_TYPES_BEGIN_EITHER_INHIBITORY_OR_EXCITATORY parameter.")
    return weightsAtTimeT


def getDecayedWeights(weightsAtTimeT):
    decayRatesByMemoryType = np.asarray(
        [x['decay_tau'] for x in env.SYNAPSE_MEMORY_TYPES.values()], dtype=np.float32)

    nonZeroDecayRates = np.nonzero(decayRatesByMemoryType)
    if(len(nonZeroDecayRates)==0):
        return weightsAtTimeT
    
    # Set all decay terms to one (i.e. no decay) temporarily
    decayTerms = np.ones(shape = weightsAtTimeT[0].shape)
    
    
    # For those weights that decay, change their decay term (currently one)
        # ... Using exp()
    # decayTerms[nonZeroDecayRates] = np.exp(-1 / decayRatesByMemoryType[nonZeroDecayRates])
        # ... Using 1-decay_rate
    
    
    # 1- decay_rate
    decayTerms[nonZeroDecayRates] = np.subtract(1, decayRatesByMemoryType[nonZeroDecayRates])


    weightsAtTimeTPlusOne = np.zeros(shape=weightsAtTimeT.shape)
    for weightsId, weights in enumerate(weightsAtTimeT):
        #newWeights = np.multiply(weights, decayRatesByMemoryType)
        newWeights = np.multiply(weights, decayTerms)
        weightsAtTimeTPlusOne[weightsId,:] = newWeights


    return weightsAtTimeTPlusOne