# CONSTANT PARAMETERS
import numpy


SEEDS = [0,1,2]
MAX_EPOCHS = 20
LEARNING_RATE = 0.1
N_PATTERNS = 100

# .x_pattern_features = n_patterns

# The number of pattern features will always be equal to the number of weights.
X_PATTERN_FEATURES = 100  # Could be 8200, according to https://pubmed.ncbi.nlm.nih.gov/2778101/

ENERGY_EXPONENT = 1
NEURONES_TYPES_BEGIN_EITHER_INHIBITORY_OR_EXCITATORY = True
NEURONES_CAN_CHANGE_TYPE_MID_SIMULATION = True
MEMORY_IS_TRANSIENT_OR_CONSOLIDATED = True

# *-*-*-*-*-*-
# !! STOP !!
# For most simulations, the settings below this line do not need to be edited.
# *-*-*-*-*-*-

# This dictionary stores the different types of neurones, and is set according to NEURONES_ARE_INHIBITORY_OR_EXCITATORY
if (NEURONES_TYPES_BEGIN_EITHER_INHIBITORY_OR_EXCITATORY):
    WEIGHT_NEURONE_TYPES = {
        'excitatory': {
            'name': 'excitatory',
            'max': 1.0,
            'min': 0,
            'default': 0,
            'percentage_quantity_of_neurones': 80,
            'cumulative': [],
            'memoryTypes': {}
        },
        'inhibitory': {
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
        'excitatory': {
            'name': 'all',
            'max': -1.0,
            'min': 1.0,
            'default': 0,
            'percentage_quantity_of_neurones': 100,
            'cumulative': [],
            'memoryTypes': {}
        },
    }

if(MEMORY_IS_TRANSIENT_OR_CONSOLIDATED):
    WEIGHT_MEMORY_TYPES = ['consolidated', 'transient']
else:
    WEIGHT_MEMORY_TYPES = ['consolidated']

def generateWeightModel():
    WEIGHT_MODEL = {}
    for neurone_name, neurone_info in WEIGHT_NEURONE_TYPES.items():
        WEIGHT_MODEL[neurone_name] = neurone_info
        for weight_memory_type in WEIGHT_MEMORY_TYPES:
            WEIGHT_MODEL[neurone_name]['memoryTypes'][weight_memory_type] = []
    return WEIGHT_MODEL
def setSeed(seed):
    global RANDOM_GENERATOR
    RANDOM_GENERATOR = numpy.random.default_rng(seed)

#WEIGHT_MODEL = WEIGHT_NEURONE_TYPES
N_WEIGHTS = X_PATTERN_FEATURES

WEIGHT_MODEL = generateWeightModel()