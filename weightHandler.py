
import numpy as np
import parameters as env
import copy


def validateParameters():
    global PERCENTAGE_INHIBITORY_WEIGHTS
    print("Skipped validation of parameters (learning)...")


def getInitialWeights(trainingDatasetX):
    #minWeight = -1
    #maxWeight = 0
    #initialWeights = (maxWeight - minWeight) * np.random.rand(len(trainingDatasetX[0])) + minWeight
    #initialWeights = [0.0 for i in range(len(trainingDatasetX[0]))]

    # !-- SET +ve/-ve WEIGHTS TO TRY AND MATCH PROPORTIONS FOUND IN BRAIN
    # https://www.brainfacts.org/brain-anatomy-and-function/cells-and-circuits/2021/how-inhibitory-neurons-shape-the-brains-code-100621

    initialWeights = []
    
    # Use template of weights to create new matrix
    for neuroneTypeName, neuroneTypeData in env.WEIGHT_MODEL.items():
      nWeights = int(round(
          (neuroneTypeData['percentage_quantity_of_neurones']/100) * env.N_WEIGHTS, 0))
      randomWeights = np.random.uniform(
          low=float(neuroneTypeData['min']),
          high=float(neuroneTypeData['max']),
          size=nWeights)

      initialWeights = np.hstack([initialWeights, randomWeights])

    np.random.shuffle(initialWeights)
    initialWeightSigns = np.sign(initialWeights)
    indexesOfWeights = {
        "excitatoryWeights": np.where(initialWeightSigns >= 0),
        "inhibitoryWeights": np.where(initialWeightSigns < 0)
    }

    # Due to rounding, the matrix may not be the correct shape. Remove/add row(s) to suit.
    sizeDifference = len(initialWeights) - (1+env.X_PATTERN_FEATURES)
    if(sizeDifference > 0):
      initialWeights = initialWeights[:-sizeDifference, :]
    elif(sizeDifference < 0):
      initialWeights = np.hstack([initialWeights, np.random.uniform(
          low=float(neuroneTypeData['min']),
          high=float(neuroneTypeData['max']),
          size=(sizeDifference*-1))])
    
    weightHistory = np.zeros([env.MAX_EPOCHS-1, len(initialWeights)])
    allWeights = np.vstack([initialWeights, weightHistory])
    
    
    zeros = np.zeros(allWeights.shape)
   
    allWeights = np.stack((allWeights, zeros), axis=2)

    return allWeights, indexesOfWeights


def getFilteredWeights(weights, filterBy=None):
    return weights[[filterBy]]


def updateCumulativeWeights(weights):
  for neuroneTypeName, neuroneTypeData in weights.items():
    for memoryTypeName, memoryTypeData in neuroneTypeData['items'].items():
      weights[neuroneTypeName]['cumulative'] += memoryTypeData
  return weights

def updateWeights(typeOfWeights, newWeights):
    if (typeOfWeights == 'excitatoryWeights'):
        a_min = 0
        a_max = None
    elif(typeOfWeights == 'inhibitoryWeights'):
        a_min = None
        a_max = 0
    else:
        a_min = None
        a_max = None

        # TODO: this code will fail if weights are not defined as excitatory or inhibitory.
    try:
        updatedWeights = np.clip(
            a=newWeights[typeOfWeights], a_min=a_min, a_max=a_max)
    except:
        updatedWeights = newWeights[typeOfWeights]
    return updatedWeights
