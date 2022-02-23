import parameters as env
import learning as l
import energy as e
import time
from json import dumps
from pandas import DataFrame
from os import path, mkdir
from datetime import datetime


# -- PREPARE DIRECTORY FOR OUTPUT
directoryName = datetime.now().strftime("%Y%m%d-%H%M%S")
mkdir(directoryName)
filePath = directoryName+'/output.csv'
simulationNumber = 0
for xPatternFeature in env.X_PATTERN_FEATURES:
    env.setXPatternFeature(xPatternFeature)
    print("xPattern = "+str(xPatternFeature))
    for nPattern in env.N_PATTERNS:
        if(((nPattern != xPatternFeature) & env.ENSURE_N_PATTERNS_EQUALS_X_PATTERNS_FEATURES)
        or env.X_PATTERN_FEATURE < nPattern): # Avoid dividing by zero error by ensuring <
            continue
        env.setNPattern(nPattern)
        print("nPattern = "+str(nPattern))
        for learningRate in env.LEARNING_RATES:
            env.setLearningRate(learningRate)
            print("learningRate = "+str(learningRate))
            for seed in env.SEEDS:
                env.setSeed(seed)
                print("seed = "+str(seed))
                start = time.time()
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

                # Calculate energy expenditure
                theoreticalEfficiency = e.calculateTheoreticalEfficiency()
                metabolicEnergy = e.calculateMetabolicEnergy(weightsByEpoch)
                theoreticalMinimumEnergy = e.calculateTheoreticalMinimumEnergy(
                    weightsByEpoch)
                efficiency = e.calculateEfficiency(
                    metabolicEnergy, theoreticalMinimumEnergy)
                maintenanceEnergy = e.calculateEnergyFromMaintenance(
                    weightsByEpoch)
                consolidationEnergy = e.calculateEnergyFromConsolidations(
                    consolidationsByEpoch)

                report = {
                    simulationNumber:
                    {
                        'time_elapsed': str(time.time() - start),
                        'seed': seed,
                        'learning_rate': learningRate,
                        'n_pattern': nPattern,
                        'n_pattern_features': xPatternFeature,  # TODO: notice it is x not n
                        'max_epochs': env.MAX_EPOCHS,
                        'energy_exponent': env.ENERGY_EXPONENT,
                        'preset_simulation': env.PRESET_SIMULATION,
                        'Learning was complete at epoch #': epochIndexForConvergence,
                        'Theoretical: random-walk efficiency': str(theoreticalEfficiency),
                        'Simulated:  efficiency (m_perc/m_min)': str(efficiency),
                        'Simulated-Theoretical efficiency difference': str(round((efficiency - theoreticalEfficiency), 3)),
                        'Theoretical: minimum energy for learning': str(theoreticalMinimumEnergy),
                        'Simulated: energy actually used by learning': str(metabolicEnergy),
                        'Simulated-Theoretical min difference for learning': str(round((metabolicEnergy - theoreticalMinimumEnergy), 3)),
                        'Energy expended by simulations for consolidations': str(consolidationEnergy),
                        'Energy expended by simulations for maintenance': str(maintenanceEnergy),
                        'Energy expended total': str(round((consolidationEnergy + maintenanceEnergy), 3)),
                        '(seen) NLL': str(trainNLL)+'%',
                        '(seen) Accuracy': str(trainAccuracy)+'%',
                        '(seen) Precision': str(trainPrecision)+'%',
                        '(seen) Sensitivity': str(trainSensitivity)+'%',
                        '(seen) Specificity': str(trainSpecificity)+'%',
                        '(unseen) NLL': str(testNLL)+'%',
                        '(unseen) Accuracy': str(testAccuracy)+'%',
                        '(unseen) Precision': str(testPrecision)+'%',
                        '(unseen) Sensitivity': str(testSensitivity)+'%',
                        '(unseen) Specificity': str(testSpecificity)+'%',
                    }
                }

                if path.isfile(filePath):
                    mode = 'a'
                    header = 0
                    columns = list(report[simulationNumber].keys())
                else:
                    mode = 'w'
                    header = list(report[simulationNumber].keys())
                    columns = header

                DataFrame.from_dict(report).transpose().to_csv(
                    filePath, header=header, columns=columns, mode=mode)
                simulationNumber = simulationNumber + 1

timeElapsed = time.time() - start
print("Finished (time elapsed: "+str(timeElapsed))
print("A csv file has been produced and is available at: (location of this script)/"+str(filePath))