import numpy as np
import parameters as env

def validateParameters():
  global ENERGY_EXPONENT
  print("Skipped validation of parameters (energy)...")


def calculateTheoreticalMinimumEnergy(weightsByEpoch):
  return round(abs(np.sum(weightsByEpoch[-1] - weightsByEpoch[0])), 3)

def calculateMetabolicEnergy(weightsByEpoch):
  # Synapse weights is currently rowed by epoch, rearrange so that each row is the same synapse but over time
  weightsByTime = np.transpose(weightsByEpoch)
  changeInWeightsPerTimeStep = np.diff(weightsByTime)
  return round(np.sum(abs(changeInWeightsPerTimeStep)**env.ENERGY_EXPONENT), 3)


def calculateEfficiency(metabolicEnergy, theoreticalMinimumEnergy):
  return round((metabolicEnergy / theoreticalMinimumEnergy), 3)


def calculateTheoreticalEfficiency():
  K = 2*env.N_PATTERN/(2-env.N_PATTERN/env.X_PATTERN_FEATURE)**2
  m_perceptron = env.N_PATTERN * K * env.LEARNING_RATE
  m_minimum = ((2/np.pi)**(1/2))*env.LEARNING_RATE*(K**(1/2))
  try:
    value = m_perceptron / m_minimum
  except :
    value = 0
  return round(value, 3)

def calculateEnergyFromMaintenance(weightsByEpoch):
  # Synapse weight changes are currently rowed by epoch, rearrange so that each row is the same synapse but over time
  weightsByEpoch = np.transpose(weightsByEpoch)
  summedWeightsForAllTimes = np.sum(np.abs(weightsByEpoch), axis=1)
  summedWeightsForAllTimesAndMemoryTypes = np.sum(
      summedWeightsForAllTimes, axis=1)
  

# TODO: Refactor this loop.
  energyConstantsByMemoryType = []
  for memoryTypeId, memoryTypeData in env.WEIGHT_MEMORY_TYPES.items():
    if(memoryTypeData['cost_of_maintenance'] == None):
      continue
    energyConstantsByMemoryType.append(memoryTypeData['cost_of_maintenance'])
  
  energyConsumedByMaintenance = np.multiply(
      summedWeightsForAllTimesAndMemoryTypes, energyConstantsByMemoryType)
  return round(np.sum(energyConsumedByMaintenance), 3)


def calculateEnergyFromConsolidations(calculateEnergyFromConsolidation):
  # Synapse weight changes are currently rowed by epoch, rearrange so that each row is the same synapse but over time
  calculateEnergyFromConsolidation = np.transpose(
      calculateEnergyFromConsolidation)
  summedConsolidationsForAllTimes = np.sum(
      np.abs(calculateEnergyFromConsolidation), axis=1)
  summedConsolidationsForAllTimesAndMemoryTypes = np.sum(
      summedConsolidationsForAllTimes, axis=1)
  

# TODO: Refactor this loop.
  energyConstantsByMemoryType = []
  for memoryTypeId, memoryTypeData in env.WEIGHT_MEMORY_TYPES.items():
    if(memoryTypeData['cost_of_consolidation'] == None):
      continue
    energyConstantsByMemoryType.append(memoryTypeData['cost_of_consolidation'])
  
  energyConsumedByConsolidation = np.multiply(
      summedConsolidationsForAllTimesAndMemoryTypes, energyConstantsByMemoryType)
  return round(np.sum(energyConsumedByConsolidation), 3)
