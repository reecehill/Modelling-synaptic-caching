# The simulation can use multiple cores for speed. Enter an integer to specify the maximum percentage of total available cores that the script will use.
# e.g., A value of 80 will lead to 80% of all cores being used.
# Where there are 12 cores available, 12*0.8=9.6. Rounding down means 9 cores will be used.
# Recommend between 50 and 80.
PERCENTAGE_OF_CPU_CORES = 80

# Set to True to run simulation and generate data.
# Alternatively, set to directory name, relative to runcode.py file, that contains .csv file from previous simulations. Usually, this is a timestamp, eg. '20220224-224209'
RUN_SIMULATION = True
