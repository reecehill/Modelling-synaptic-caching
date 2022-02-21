import parameters as env
import learning as l
import energy as e
import time

for seed in env.SEEDS:
    start = time.time()

    env.setSeed(seed)
    print("Starting...")
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

    timeElapsed=time.time() - start

    print("Finished (time elapsed: "+str(timeElapsed))
    print(report)
