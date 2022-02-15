import learning as l
import energy as e

l.MAX_EPOCHS = 40
l.LEARNING_RATE = 0.1
l.N_PATTERNS = 1000
#l.x_pattern_features = n_patterns
l.X_PATTERN_FEATURES = 8200  #https://pubmed.ncbi.nlm.nih.gov/2778101/
e.ENERGY_EXPONENT = 1

# Validate parameters
l.validateParameters()
e.validateParameters()

# Generate datasets
trainingDatasetX, trainingDatasetY, testingDatasetX, testingDatasetY = l.getDatasets()

# Use training dataset to estimate weights, where each row of weights is a new epoch.
weightsByEpoch = l.trainWeights(trainingDatasetX, trainingDatasetY)

# Get performance metrics (currently only NLL works)
negativeLogLikelihood, accuracy, precision, sensitivity, specificity = l.testWeights(
    testingDatasetX, testingDatasetY, weightsByEpoch)

# Calculate metabolic energy required for learning
energy = e.calculateMetabolicEnergy(weightsByEpoch)

print(negativeLogLikelihood, accuracy, precision, sensitivity, specificity)
