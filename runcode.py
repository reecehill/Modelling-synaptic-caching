import learning as l
import energy as e


# Generate datasets
trainingDatasetX, trainingDatasetY, testingDatasetX, testingDatasetY = l.getDatasets()

# Use training dataset to estimate weights, where each row of weights is a new epoch.
weightsByEpoch = l.trainWeights(trainingDatasetX, trainingDatasetY)

# Get performance metrics (currently only NLL works)
negativeLogLikelihood, accuracy, precision, sensitivity, specificity = l.testWeights(
    testingDatasetX, testingDatasetY, weightsByEpoch)

# Calculate metabolic energy required for learning
energy = e.calculateMetabolicEnergy(weightsByEpoch)


report = {
    'NLL': str(negativeLogLikelihood)+'%',
    'Accuracy': str(accuracy)+'%',
    'Precision': str(precision)+'%',
    'Sensitivity': str(sensitivity)+'%',
    'Specificity': str(specificity)+'%'
}
print(report)
