import numpy as np


def init(max_epochs=None, learning_rate=None, n_patterns=None, x_pattern_features=None):
  print("Skipped validation of parameters...")
  return max_epochs, learning_rate, n_patterns, x_pattern_features


def getDatasets(n_patterns, x_pattern_features):
  def generateDataset():
    # -- GENERATE DATASET
    dataset = np.random.randint(2, size=[n_patterns, x_pattern_features])
    dataset[dataset == 0] = -1  # Rewrite zeros to -1.

    # Add targets to dataset
    y = np.random.randint(2, size=[n_patterns, 1])
    dataset = np.append(dataset, y, axis=1)
    return dataset

  return generateDataset(), generateDataset()

# Make a prediction with weights


def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]

	return 1 if activation >= 0.0 else 0

# Estimate Perceptron weights using stochastic gradient descent


def trainWeights(dataset, learning_rate, max_epochs):
	# weights = [0.0 for i in range(len(dataset[0]))]
  minWeight = -1
  maxWeight = 1
  weights = (maxWeight - minWeight)*np.random.rand(len(dataset[0])) + minWeight
  for epoch in range(max_epochs):
    sum_error = 0.0
    for row in dataset:
      prediction = predict(row, weights)
      error = row[-1] - prediction
      sum_error += error**2
      weights[0] = weights[0] + learning_rate * error
      for i in range(len(row)-1):
        deltaWeight = learning_rate * (error * row[i])
        weights[i + 1] = weights[i + 1] + deltaWeight
    print('->epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
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

def testWeights(testingDataset, weights):
  actualTargets = testingDataset[:, -1]
  predictedTargets = list()
  
  for row in testingDataset:
    predictedTarget = predict(row, weights)
    predictedTargets.append(predictedTarget)
  
  return evaluatePerformance(actualTargets, predictedTargets)