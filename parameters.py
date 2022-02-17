# CONSTANT PARAMETERS
PERCENTAGE_INHIBITORY_WEIGHTS = 20
MAX_EPOCHS = 500
LEARNING_RATE = 0.001
N_PATTERNS = 1000
#.x_pattern_features = n_patterns
X_PATTERN_FEATURES = 1000  # https://pubmed.ncbi.nlm.nih.gov/2778101/
ENERGY_EXPONENT = 1

# Split weights into transient and permanent?
CONSOLIDATED_AND_TRANSIENT_WEIGHTS = False

def automateVariables():
  global N_WEIGHT_FORMS
  if(CONSOLIDATED_AND_TRANSIENT_WEIGHTS):
    N_WEIGHT_FORMS = 2
  else:
    N_WEIGHT_FORMS = 1

automateVariables();