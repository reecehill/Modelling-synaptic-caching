
import numpy as np
import parameters as env

def validateParameters():
  global PERCENTAGE_INHIBITORY_WEIGHTS
  print("Skipped validation of parameters (learning)...")


def getInitialWeights(trainingDatasetX):
  #minWeight = -1
  #maxWeight = 0
  #initialWeights = (maxWeight - minWeight) * np.random.rand(len(trainingDatasetX[0])) + minWeight
  #initialWeights = [0.0 for i in range(len(trainingDatasetX[0]))]

  # !-- SET +ve/-ve WEIGHTS TO TRY AND MATCH PROPORTIONS FOUND IN BRAIN
  #https://www.brainfacts.org/brain-anatomy-and-function/cells-and-circuits/2021/how-inhibitory-neurons-shape-the-brains-code-100621
  nInhibitoryWeights = int(
      round(len(trainingDatasetX[0]) * (env.PERCENTAGE_INHIBITORY_WEIGHTS/100), 0))
  nExcitatoryWeights = len(trainingDatasetX[0]) - nInhibitoryWeights
  randomInhibitoryWeights = np.random.rand(nInhibitoryWeights) * -1
  randomExcitatoryWeights = np.random.rand(nExcitatoryWeights)
  initialWeights = np.concatenate(
      (randomExcitatoryWeights, randomInhibitoryWeights))
  np.random.shuffle(initialWeights)

  initialWeightSigns = np.sign(initialWeights)
  indexesOfWeights = {
      "excitatoryWeights": np.where(initialWeightSigns >= 0),
      "inhibitoryWeights": np.where(initialWeightSigns < 0)
  }

  weightHistory = np.zeros((env.MAX_EPOCHS-1, len(initialWeights)))
  allWeights = np.vstack([initialWeights, weightHistory])

  return allWeights, indexesOfWeights


def getFilteredWeights(epochIndex, weights, filterBy=None):
  return weights[epochIndex, [filterBy]]


def updateWeights(epochIndex, weights, newWeights, indexesOfWeights):
  for typeOfWeights, indexOfWeights in indexesOfWeights.items():
    if (typeOfWeights == 'excitatoryWeights'):
      a_min = 0
      a_max = None
    elif(typeOfWeights == 'inhibitoryWeights'):
      a_min = None
      a_max = 0
    else:
      a_min = 1
      a_max = 0
    weights[epochIndex, [indexOfWeights]] = np.clip(
        a=newWeights[typeOfWeights], a_min=a_min, a_max=a_max)
  return weights
