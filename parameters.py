# CONSTANT PARAMETERS
PERCENTAGE_INHIBITORY_WEIGHTS = 20
MAX_EPOCHS = 500
LEARNING_RATE = 0.01
N_PATTERNS = 1000
# .x_pattern_features = n_patterns

# The number of pattern features will always be equal to the number of weights.
X_PATTERN_FEATURES = 8200  # https://pubmed.ncbi.nlm.nih.gov/2778101/
ENERGY_EXPONENT = 1

# Split weights into transient and permanent?
USE_TRANSIENT_WEIGHTS = True
USE_CONSOLIDATED_WEIGHTS = True

WEIGHT_MEMORY_TYPES = ['consolidated', 'transient']
WEIGHT_NEURONE_TYPES = {
    'excitatory': {
        'name': 'excitatory',
        'max': 1.0,
        'min': 0,
        'default': 0,
        'generateMemoryTypes': True,
        'percentage_quantity_of_neurones': 80,
        'cumulative': [],
        'items': {}
    },
    'inhibitory': {
        'max': 0,
        'min': -1.0,
        'default': 0,
        'generateMemoryTypes': True,
        'percentage_quantity_of_neurones': 20,
        'cumulative': [],
        'items': {}
    }
}


# NOTHING NEED BE EDITED BELOW THIS LINE -----


def automateVariables():
    global WEIGHT_MODEL
    WEIGHT_MODEL = {}
    for neurone_name, neurone_info in WEIGHT_NEURONE_TYPES.items():
      WEIGHT_MODEL[neurone_name] = neurone_info
      for weight_memory_type in WEIGHT_MEMORY_TYPES:
        if(WEIGHT_MODEL[neurone_name]['generateMemoryTypes']):
          WEIGHT_MODEL[neurone_name]['items'][weight_memory_type] = []

WEIGHT_MODEL = WEIGHT_NEURONE_TYPES
N_WEIGHTS = X_PATTERN_FEATURES

automateVariables()
