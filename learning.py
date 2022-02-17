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
    datasetX = np.random.randint(
        2, size=[env.N_PATTERNS, env.X_PATTERN_FEATURES])
    datasetX[datasetX == 0] = -1  # Rewrite zeros to -1.
    ones = np.ones((env.N_PATTERNS, 1))
    datasetX = np.hstack([ones, datasetX])

    # Add targets to dataset
    datasetY = np.random.randint(2, size=[env.N_PATTERNS, 1])
    return datasetX, datasetY
    
  trainingDatasetX, trainingDatasetY = generateDataset()
  testingDatasetX, testingDatasetY = generateDataset()

  return trainingDatasetX, trainingDatasetY, testingDatasetX, testingDatasetY

# Make a prediction with weights
def predict(pattern, weights):
  activation = np.dot(weights, pattern)
  return [1] if activation >= 0.0 else [0]

# Estimate Perceptron weights using stochastic gradient descent


def trainWeights(trainingDatasetX, trainingDatasetY):
  weights, indexesOfWeights = w.getInitialWeights(trainingDatasetX)
  for epochIndex in range(1, env.MAX_EPOCHS):
    sum_mse = 0.0
    for patternIndex, pattern in enumerate(trainingDatasetX):
      prediction = predict(pattern, weights[epochIndex])
      error = trainingDatasetY[patternIndex] - prediction
      sum_mse += error**2
      deltaWeights = env.LEARNING_RATE * (error * pattern)
      newWeights = dict()
      for typeOfWeights, indexOfWeights in indexesOfWeights.items():
        # Using their indexes, get excitatory and inhibitory weights and add them to their respective deltaWeights.
        newWeights[typeOfWeights] = w.getFilteredWeights(
            epochIndex, weights, filterBy=indexOfWeights) + deltaWeights[indexOfWeights]

      weights = w.updateWeights(
          epochIndex, weights, newWeights, indexesOfWeights)

    # Set the proceding weight timestep to be equal to that of the current timestep (so predictions are not made from zeros).
    if(epochIndex != env.MAX_EPOCHS-1):
      weights[epochIndex+1] = weights[epochIndex]
    
    print('->epochIndex=%d, lrate=%.3f, MSE=%f' %
          (epochIndex, env.LEARNING_RATE, (sum_mse/len(trainingDatasetX))))
  return weights


def testWeights(testingDatasetX, testingDatasetY, weights):
  predictedTargets = list()

  for pattern in testingDatasetX:
    predictedTarget = predict(pattern, weights[-1, :])
    predictedTargets.append(predictedTarget)

  predictedTargets = np.array(predictedTargets)
  return eval.evaluatePerformance(testingDatasetY, predictedTargets)