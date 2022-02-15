import numpy as np

def validateParameters():
  global ENERGY_EXPONENT
  print("Skipped validation of parameters (energy)...")

def calculateMetabolicEnergy(weightsByEpoch):
  # Synapse weights is currently rowed by epoch, rearrange so that each row is the same synapse but over time
  weightsByTime = np.transpose(weightsByEpoch)
  changeInWeightsPerTimeStep = np.diff(weightsByTime)
  return np.sum(abs(changeInWeightsPerTimeStep)**ENERGY_EXPONENT)
