import numpy as np


def validateParameters():
  global MAX_EPOCHS
  global LEARNING_RATE
  global N_PATTERNS
  global X_PATTERN_FEATURES
  print("Skipped validation of parameters (learning)...")


def getDatasets():
  def generateDataset():
    # -- GENERATE DATASET
    datasetX = np.random.randint(2, size=[N_PATTERNS, X_PATTERN_FEATURES])
    datasetX[datasetX == 0] = -1  # Rewrite zeros to -1.

    # Add targets to dataset
    datasetY = np.random.randint(2, size=[N_PATTERNS, 1])
    return datasetX, datasetY
    
  trainingDatasetX, trainingDatasetY = generateDataset()
  testingDatasetX, testingDatasetY = generateDataset()

  return trainingDatasetX, trainingDatasetY, testingDatasetX, testingDatasetY

# Make a prediction with weights
def predict(pattern, weights):
  activation = np.matmul(weights, pattern)
  return 1 if activation >= 0.0 else 0

# Estimate Perceptron weights using stochastic gradient descent


def trainWeights(trainingDatasetX, trainingDatasetY):
	# weights = [0.0 for i in range(len(dataset[0]))]
  minWeight = -1
  maxWeight = 1
  initialWeights = (maxWeight - minWeight) * \
      np.random.rand(len(trainingDatasetX[0])) + minWeight
  weightHistory = np.zeros((MAX_EPOCHS-1, len(initialWeights)))
  weights = np.vstack([initialWeights, weightHistory])
  for epochIndex in range(1, MAX_EPOCHS):
    sum_error = 0.0
    for patternIndex, pattern in enumerate(trainingDatasetX):
      prediction = predict(pattern, weights[epochIndex])
      error = trainingDatasetY[patternIndex] - prediction
      sum_error += error**2
      deltaWeights = LEARNING_RATE * (error * pattern)
      weights[epochIndex] = weights[epochIndex] + deltaWeights
    if(epochIndex != MAX_EPOCHS-1):
      weights[epochIndex+1] = weights[epochIndex]

    print('->epochIndex=%d, lrate=%.3f, error=%.3f' %
          (epochIndex, LEARNING_RATE, sum_error))
  return weights


def evaluatePerformance(actualTargets, predictedTargets):
  truePositives = np.where((actualTargets[:] == 1) &
                           (predictedTargets[:] == 1))[0]
  falsePositives = np.where((actualTargets == 1) &
                         (predictedTargets == 0))[0].size
  trueNegatives = np.where((actualTargets == 0) &
                        (predictedTargets == 0))[0].size
  falseNegatives = np.where((actualTargets == 0) &
                         (predictedTargets == 1))[0].size
  negativeLogLikelihood = -np.mean((actualTargets*predictedTargets) - np.log(1+np.exp(predictedTargets)))
  
  try:
    accuracy = (truePositives + trueNegatives) / actualTargets.size
  except ZeroDivisionError:
    accuracy = 0

  try:
    precision = truePositives / (truePositives + falsePositives)
  except ZeroDivisionError:
    precision = 0
  
  try: 
    sensitivity = truePositives / (truePositives + falseNegatives)
  except ZeroDivisionError:
    sensitivity = 0
  
  try:
    specificity = trueNegatives / (trueNegatives + falsePositives)
  except ZeroDivisionError:
    specificity = 0

  return negativeLogLikelihood, accuracy, precision, sensitivity, specificity

def testWeights(testingDatasetX, testingDatasetY, weights):
  predictedTargets = list()
  
  for pattern in testingDatasetX:
    predictedTarget = predict(pattern, weights[-1,:])
    predictedTargets.append(predictedTarget)
   
  return evaluatePerformance(testingDatasetY, predictedTargets)