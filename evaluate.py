import numpy as np


def evaluatePerformance(actualTargets, predictedTargets):
  truePositives = float(sum((predictedTargets == [1]) & (
      actualTargets == predictedTargets) == True))
  falsePositives = float(sum((predictedTargets == 1) & (
      actualTargets != predictedTargets) == True))
  trueNegatives = float(sum((predictedTargets == 0) & (
      actualTargets == predictedTargets) == True))
  falseNegatives = float(sum((predictedTargets == 0) & (
      actualTargets != predictedTargets) == True))
  negativeLogLikelihood = - \
      np.mean((actualTargets*predictedTargets) -
              np.log(1+np.exp(predictedTargets)))

  try:
    accuracy = round(((truePositives + trueNegatives) /
                     actualTargets.size) * 100, 3)
  except ZeroDivisionError:
    accuracy = 0

  try:
    precision = round(
        (truePositives / (truePositives + falsePositives)) * 100, 3)
  except ZeroDivisionError:
    precision = 0

  try:
    sensitivity = round(
        (truePositives / (truePositives + falseNegatives)) * 100, 3)
  except ZeroDivisionError:
    sensitivity = 0

  try:
    specificity = round(
        (trueNegatives / (trueNegatives + falsePositives)) * 100, 3)
  except ZeroDivisionError:
    specificity = 0

  return negativeLogLikelihood, accuracy, precision, sensitivity, specificity
