# CONSTANT PARAMETERS
from numpy import random


SEEDS = [0]
MAX_EPOCHS = 500  # Max epochs before concluding convergence is not possible
LEARNING_RATE = 0.1
N_PATTERNS = 250

# .x_pattern_features = n_patterns

# The number of pattern features will always be equal to the number of weights.
# Could be 8200, according to https://pubmed.ncbi.nlm.nih.gov/2778101/
X_PATTERN_FEATURES = 250

ENERGY_EXPONENT = 1

# Set simulation parameters according to preset scenarios. Enter an integer, as follows:
# 1: Neurones have behaviour alike that in the paper.
# 2: Prevent neurones from switching between excitatory/inhibitory (and deactivate them/set to zero if they try)
PRESET_SIMULATION  = 1

# Only in effect when neurones are allowed to have transient/consolidated memory types.
MAX_SIZE_OF_TRANSIENT_MEMORY = 0.5


# *-*-*-*-*-*-
# !! STOP !!
# For most simulations, the settings below this line do not need to be edited.
# *-*-*-*-*-*-

if(PRESET_SIMULATION == 1):
    NEURONES_CAN_CHANGE_TYPE_MID_SIMULATION = True
    MEMORY_IS_TRANSIENT_OR_CONSOLIDATED = True
    NEURONES_TYPES_BEGIN_EITHER_INHIBITORY_OR_EXCITATORY = False
elif(PRESET_SIMULATION == 2):
    NEURONES_CAN_CHANGE_TYPE_MID_SIMULATION = False
    MEMORY_IS_TRANSIENT_OR_CONSOLIDATED = True
    NEURONES_TYPES_BEGIN_EITHER_INHIBITORY_OR_EXCITATORY = True

# This dictionary stores the different types of neurones, and is set according to NEURONES_ARE_INHIBITORY_OR_EXCITATORY
if (NEURONES_TYPES_BEGIN_EITHER_INHIBITORY_OR_EXCITATORY):
    WEIGHT_NEURONE_TYPES = {
        0: {
            'name': 'excitatory',
            'max': 1.0,
            'min': 0,
            'default': 0,
            'percentage_quantity_of_neurones': 80,
            'cumulative': [],
            'memoryTypes': {}
        },
        1: {
            'name': 'inhibitory',
            'max': 0,
            'min': -1.0,
            'default': 0,
            'percentage_quantity_of_neurones': 20,
            'cumulative': [],
            'memoryTypes': {}
        }
    }
else:
    WEIGHT_NEURONE_TYPES = {
        0: {
            'name': 'all',
            'max': 1.0,
            'min': -1.0,
            'default': 0,
            'percentage_quantity_of_neurones': 100,
            'cumulative': [],
            'memoryTypes': {}
        },
    }

if(MEMORY_IS_TRANSIENT_OR_CONSOLIDATED):
    WEIGHT_MEMORY_TYPES = {
        0: {
            'name': 'consolidated',
            'memory_size': False,
            'cost_of_maintenance': 1,
            'cost_of_consolidation': 2,
        },
        1: {
            'name': 'transient',
            'memory_size': MAX_SIZE_OF_TRANSIENT_MEMORY,
            'cost_of_maintenance': 0.5,
            'cost_of_consolidation': 0 # The highest memory type does not receive weights for consolidation. 
        }
    }
else:
    WEIGHT_MEMORY_TYPES = {
        0: {
            'name': 'consolidated',
            'memory_size': False
        }
    }


def generateWeightModel():
    WEIGHT_MODEL = {}
    for neurone_name, neurone_info in WEIGHT_NEURONE_TYPES.items():
        WEIGHT_MODEL[neurone_name] = neurone_info
        for weight_memory_type in WEIGHT_MEMORY_TYPES:
            WEIGHT_MODEL[neurone_name]['memoryTypes'][weight_memory_type] = []
    return WEIGHT_MODEL


def setSeed(seed):
    global RANDOM_GENERATOR
    RANDOM_GENERATOR = random.default_rng(seed)


#WEIGHT_MODEL = WEIGHT_NEURONE_TYPES
N_WEIGHTS = X_PATTERN_FEATURES

WEIGHT_MODEL = generateWeightModel()
