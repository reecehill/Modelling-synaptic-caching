This project reproduces and extends upon the results of Li and van Rossum (2020): https://elifesciences.org/articles/50804

# PBM-Project1

# Instructions
1) Set parameters (i.e. seeds, max_epochs) in parameters file.
2) Run runcode.py

# IMPORTANT
This script uses "neuronal types" and "memory/weight types":

Neuronal types: used to define a synapse as excitatory or inhibitory. In essence, it is a wrapper that allows min and max values to be set. Could be used for synapses that have known limited ranges for activity. One potential use could be that a excitatory synapse A could be less active (and thus max=2) whereas another synapse, also excitatory, could have a max set to 5. Inhibition is represented through negative numbers.

Memory types: these are sub-units of a given neuronal type. In practice, they could describe an increase in vesicular release, or more permanent methods of increasing synapse-synapse connectivity such as increasing the number of synapses. In the paper, it is split into transient (temporary memory, i.e., vesicular release) and consolidated (more permanent) memory.

For example, a suitable hierarchy could be:

 - Excitatory synapse
   - Consolidated memory type
   - Transient memory type
 - Inhibitory synapse
   - Consolidated memory type
   - Transient memory type

NB: All synapses must have the same memory types.
