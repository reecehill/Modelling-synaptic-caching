import numpy as np
import parameters as env

def validateParameters():
  global ENERGY_EXPONENT
  print("Skipped validation of parameters (energy)...")

def calculateMetabolicEnergy(weightsByEpoch):
  # Synapse weights is currently rowed by epoch, rearrange so that each row is the same synapse but over time
  weightsByTime = np.transpose(weightsByEpoch)
  changeInWeightsPerTimeStep = np.diff(weightsByTime)
  return round(np.sum(abs(changeInWeightsPerTimeStep)**env.ENERGY_EXPONENT), 3)

def calculateEnergyFromConsolidations(consolidationsByEpoch):
  # Synapse weight changes are currently rowed by epoch, rearrange so that each row is the same synapse but over time
  consolidationsByEpoch = np.transpose(consolidationsByEpoch)
  totalConsolidationChanges = round(np.sum(consolidationsByEpoch), 3)
  return totalConsolidationChanges