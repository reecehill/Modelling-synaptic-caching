import parameters as env
import numpy as np
def validateParameters():
  global ENERGY_EXPONENT
  print("Skipped validation of parameters (energy)...")


def calculateTheoreticalMinimumEnergy(weightsByEpoch):
  return round(np.sum(abs(weightsByEpoch[-1] - weightsByEpoch[0])), 3)

def calculateMetabolicEnergy(weightsByEpoch):
  # Synapse weights is currently rowed by epoch, rearrange so that each row is the same synapse but over time
  weightsByTime = np.transpose(weightsByEpoch)
  changeInWeightsPerTimeStep = np.diff(weightsByTime)
  return round(np.sum(abs(changeInWeightsPerTimeStep)**env.ENERGY_EXPONENT), 3)


def calculateEfficiency(metabolicEnergy, theoreticalMinimumEnergy):
  return round((metabolicEnergy - theoreticalMinimumEnergy), 3)


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
  #∑_t|s_i(t)|)
  summedWeightsForAllTimes = np.sum(np.abs(weightsByEpoch), axis=1)  
  
  #∑_i( ∑_t|s_i(t)|) )
  summedWeightsForAllTimesAndMemoryTypes = np.sum(
      summedWeightsForAllTimes, axis=1)
  

# TODO: Refactor this loop.
  energyConstantsByMemoryType = []
  for memoryTypeId, memoryTypeData in env.WEIGHT_MEMORY_TYPES.items():
    if(memoryTypeData['cost_of_maintenance'] == None):
      continue
    energyConstantsByMemoryType.append(memoryTypeData['cost_of_maintenance'])
  
  # c * ∑_i( ∑_t|s_i(t)|) )
  energyConsumedByMaintenance = np.multiply(
      summedWeightsForAllTimesAndMemoryTypes, energyConstantsByMemoryType)
  return round(np.sum(energyConsumedByMaintenance), 3)


def calculateEnergyFromConsolidations(consolidationsByEpoch):
  # Synapse weight changes are currently rowed by epoch, rearrange so that each row is the same synapse but over time
  consolidationsByTime = np.transpose(
      consolidationsByEpoch)

  # | l_i(t) - l_i (t-1) |
  changeInWeightsPerTimeStep = np.abs(np.diff(consolidationsByTime))

  # ∑_t (| l_i(t) - l_i (t-1) |)
  summedConsolidationsForAllTimes = np.sum(changeInWeightsPerTimeStep, axis=1)

  # ∑_i ∑_t (| l_i(t) - l_i (t-1) |)
  summedConsolidationsForAllTimesAndMemoryTypes = np.sum(summedConsolidationsForAllTimes, axis=1)
  

# TODO: Refactor this loop.
  energyConstantsByMemoryType = []
  for memoryTypeId, memoryTypeData in env.WEIGHT_MEMORY_TYPES.items():
    energyConstantsByMemoryType.append(memoryTypeData['cost_of_consolidation'])
  
  
  # c * ∑_i ∑_t (| l_i(t) - l_i (t-1) |)
  # NOTE: The paper considers c=1 for consolidations. I.e., the cost of consolidation is equal to the change in weight.
  consolidationEnergyByMemoryType = np.multiply(
      summedConsolidationsForAllTimesAndMemoryTypes, energyConstantsByMemoryType)
  
  # The consolidation energy so far, is stored as a vector. With each element representing the consolidation energy required per memory type (e.g., persistent, persisent2, transient). 
  # NOTE: The paper considers only persistent memory as a memory type that accepts consolidation.

  # Get sum of vector.
  summedConsolidationEnergy = np.sum(consolidationEnergyByMemoryType)
  return round(summedConsolidationEnergy, 3)
