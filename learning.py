import numpy as np
import parameters as env
import weightHandler as w
import evaluate as eval

def validateParameters():
  w.validateParameters()
  global MAX_EPOCHS
  global LEARNING_RATE
  global N_PATTERNS
  global X_PATTERN_FEATURES
  print("Skipped validation of parameters (learning)...")


def getDatasets():
  def generateDataset():
    # -- GENERATE DATASET
    datasetX = env.RANDOM_GENERATOR.integers(
        2, size=(env.N_PATTERNS, env.X_PATTERN_FEATURES))
    datasetX[datasetX == 0] = -1  # Rewrite zeros to -1.
    
    # TO-DO: Should we add ones to our dataset? If so, how do we handle the shape difference between weights and features?
    #ones = np.ones((env.N_PATTERNS, 1))
    #datasetX = np.hstack([ones, datasetX])

    # Add targets to dataset
    datasetY = env.RANDOM_GENERATOR.integers(2, size=(env.N_PATTERNS, 1))
    return datasetX, datasetY
    
  trainingDatasetX, trainingDatasetY = generateDataset()
  testingDatasetX, testingDatasetY = generateDataset()

  return trainingDatasetX, trainingDatasetY, testingDatasetX, testingDatasetY

# Make a prediction with weights
def predict(pattern, weightsAtTimeT):
  summedWeightsByType = w.getSummedWeightsByType(weightsAtTimeT)
  activation = np.dot(summedWeightsByType, pattern)
  return [1] if activation >= 0.0 else [0]

# Estimate Perceptron weights using stochastic gradient descent
def trainWeights(trainingDatasetX, trainingDatasetY):
  weightsByTime, neuronalTypes = w.prepareWeights(
      trainingDatasetX)
  consolidationsByTime = w.prepareConsolidationEvents(weightsByTime.shape)
  for epochIndex in range(0, env.MAX_EPOCHS):
    sum_mse = 0.0
    for patternIndex, pattern in enumerate(trainingDatasetX):
      prediction = predict(pattern, weightsByTime[epochIndex])
      error = trainingDatasetY[patternIndex] - prediction
      sum_mse += error**2

      # To-do: for now, only the last weight type is added to with delta weight.
      deltaWeights = np.zeros(weightsByTime[epochIndex].shape)
      deltaWeights[:,-1] = env.LEARNING_RATE * (error * pattern)
      weightsByTime[epochIndex], consolidationsByTime[epochIndex] = w.updateWeights(
          weightsByTime[epochIndex], deltaWeights, neuronalTypes, consolidationsByTime[epochIndex])

    # Set the proceding weight timestep to be equal to that of the current timestep (so predictions are not made from zeros).
    if(epochIndex != env.MAX_EPOCHS-1):
      weightsByTime[epochIndex+1] = weightsByTime[epochIndex]
    
    #print('->epochIndex=%d, lrate=%.3f, MSE=%f' %(epochIndex, env.LEARNING_RATE, (sum_mse/len(trainingDatasetX))))
  return weightsByTime, consolidationsByTime


def testWeights(testingDatasetX, testingDatasetY, weights):
  predictedTargets = list()

  for pattern in testingDatasetX:
    predictedTarget = predict(pattern, weights[-1, :])
    predictedTargets.append(predictedTarget)

  predictedTargets = np.array(predictedTargets)
  return eval.evaluatePerformance(testingDatasetY, predictedTargets)