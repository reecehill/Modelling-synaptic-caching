import nested as n
import numpy as np
max_epochs = 10
learning_rate = 0.1
n_patterns = 500
#x_pattern_features = n_patterns
x_pattern_features = 500


# Validate parameters
max_epochs, learning_rate, n_patterns, x_pattern_features = n.init(
    max_epochs, learning_rate, n_patterns, x_pattern_features)

# Generate datasets
trainingDataset, testingDataset = n.getDatasets(n_patterns, x_pattern_features)

# Use training dataset to estimate weights 
weights = n.trainWeights(trainingDataset, learning_rate, max_epochs)

# Get performance metrics (currently only NLL works)
negativeLogLikelihood, accuracy, precision, sensitivity, specificity = n.testWeights(
    testingDataset, weights)

print(negativeLogLikelihood, accuracy, precision, sensitivity, specificity)
