import parameters as env
import weightHandler as w
import evaluate as eval
import numpy as np
def validateParameters():
  w.validateParameters()
  global MAX_EPOCHS
  global LEARNING_RATE
  global N_PATTERN
  global X_PATTERN_FEATURE
  print("Skipped validation of parameters (learning)...")

def getDatasets():
  def generateDataset():
    # -- GENERATE DATASET

    datasetX = env.RANDOM_GENERATOR.integers(
        2, size=(env.N_PATTERN, env.X_PATTERN_FEATURE))
    datasetX[datasetX == 0] = -1  # Rewrite zeros to -1.
    
    # TO-DO: Should we add ones to our dataset? If so, how do we handle the shape difference between weights and features?
    ones = np.ones((env.N_PATTERN, 1))
    datasetX = np.hstack([ones, datasetX])

    # Add targets to dataset
    datasetY = env.RANDOM_GENERATOR.integers(2, size=(env.N_PATTERN, 1))
    return datasetX, datasetY
    
  trainingDatasetX, trainingDatasetY = generateDataset()
  testingDatasetX, testingDatasetY = generateDataset()

  return trainingDatasetX, trainingDatasetY, testingDatasetX, testingDatasetY

# Make a prediction with weights
def predict(pattern, weightsAtTimeT):
  summedWeightsByType = w.getSummedWeightsByType(weightsAtTimeT)
  activation = np.dot(summedWeightsByType, pattern)
  return [1] if (activation > 0.0) else [0]

# Estimate Perceptron weights using stochastic gradient descent
def trainWeights(trainingDatasetX, trainingDatasetY):
  weightsByTime, neuronalTypes = w.prepareWeights(
      trainingDatasetX)
  consolidationsByTime = w.prepareConsolidationEvents(weightsByTime.shape)
  epochIndexForConvergence = False
  # Start from 1, so that initial weights are untouched.
  for epochIndex in range(1, env.MAX_EPOCHS):
    sum_mse = 0.0
    # Get current weights as a function of the previous weights (with/without decay)
    weightsByTime[epochIndex] = w.getDecayedWeights(weightsByTime[epochIndex-1])


    for patternIndex, pattern in enumerate(trainingDatasetX):

      # Make predictions based on these new, decayed weights.
      prediction = predict(pattern, weightsByTime[epochIndex])
      error = trainingDatasetY[patternIndex] - prediction
      sum_mse += (error**2)

      # TODO: for now, only the most transient weight type is added to with delta weight.
      deltaWeights = np.zeros(weightsByTime[epochIndex].shape)
      deltaWeights[:,-1] = env.LEARNING_RATE * (error * pattern)

      # Override current weights according to prediction from decayed weights.
      weightsByTime[epochIndex], consolidationsByTime[epochIndex] = w.updateWeights(
          weightsByTime[epochIndex], deltaWeights, neuronalTypes, consolidationsByTime[epochIndex])

    if(sum_mse == 0.0):
      # No weights were changed this epoch. Therefore, assume learning is complete.
      # We pass the current weights and next epoch's consolidations, 
      weightsByTime[epochIndex+1], consolidationsByTime[epochIndex+1] = w.consolidateAllWeights(
          weightsByTime[epochIndex], consolidationsByTime[epochIndex+1])
      epochIndexForConvergence = epochIndex+1
      break
    #print('->epochIndex=%d, lrate=%.3f, MSE=%f' %(epochIndex, env.LEARNING_RATE, (sum_mse/len(trainingDatasetX))))
  return epochIndexForConvergence, weightsByTime, consolidationsByTime


def testWeights(testingDatasetX, testingDatasetY, weights):
  predictedTargets = list()

  for pattern in testingDatasetX:
    predictedTarget = predict(pattern, weights[-1, :])
    predictedTargets.append(predictedTarget)

  predictedTargets = np.array(predictedTargets)
  return eval.evaluatePerformance(testingDatasetY, predictedTargets)