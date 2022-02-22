import parameters as env
import learning as l
import energy as e
import time
from json import dumps

for seed in env.SEEDS:
    start = time.time()

    env.setSeed(seed)
    print("Starting...")
    # Generate datasets
    trainingDatasetX, trainingDatasetY, testingDatasetX, testingDatasetY = l.getDatasets()

    # Use training dataset to estimate weights, where each row of weights is a new epoch.
    epochIndexForConvergence, weightsByEpoch, consolidationsByEpoch = l.trainWeights(
        trainingDatasetX, trainingDatasetY)

    # Get performance metrics for seen data
    trainNLL, trainAccuracy, trainPrecision, trainSensitivity, trainSpecificity = l.testWeights(
        trainingDatasetX, trainingDatasetY, weightsByEpoch)
    # Get performance metrics for UNSEEN data
    testNLL, testAccuracy, testPrecision, testSensitivity, testSpecificity = l.testWeights(
        testingDatasetX, testingDatasetY, weightsByEpoch)

    # Calculate metabolic energy required for learning
    theoreticalMinimumEnergy = e.calculateTheoreticalMinimumEnergy(weightsByEpoch)
    metabolicEnergy = e.calculateMetabolicEnergy(weightsByEpoch)
    maintenanceEnergy = e.calculateEnergyFromMaintenance(weightsByEpoch)
    consolidationEnergy = e.calculateEnergyFromConsolidations(
        consolidationsByEpoch)


    report = {
        'Learning was complete at epoch #': epochIndexForConvergence,
        'Theoretical Minimum Energy': str(theoreticalMinimumEnergy),
        'Metabolic Energy': str(metabolicEnergy),
        'Energy expended for...': {
            'consolidations': str(consolidationEnergy),
            'maintenance': str(maintenanceEnergy),
        },
        'Seen data...': {
            'NLL': str(trainNLL)+'%',
            'Accuracy': str(trainAccuracy)+'%',
            'Precision': str(trainPrecision)+'%',
            'Sensitivity': str(trainSensitivity)+'%',
            'Specificity': str(trainSpecificity)+'%',
        },
        'Unseen data...': {
            'NLL': str(testNLL)+'%',
            'Accuracy': str(testAccuracy)+'%',
            'Precision': str(testPrecision)+'%',
            'Sensitivity': str(testSensitivity)+'%',
            'Specificity': str(testSpecificity)+'%'
        },

    }

    timeElapsed=time.time() - start

    print("Finished (time elapsed: "+str(timeElapsed))
    
    print(dumps(report, indent=4))
print("Done.")