from enum import Enum

class version(Enum):
    HALF_PC = 0 # proposed prefetcher
    FULL_PC = 1
    PAGE = 2

"""
             Settings
"""

# feature
feature = version.HALF_PC
# how far into the future to make correlations to
lookahead = 3
# inputs to filter
kernelSize = 6
# nr of filters
nrFilters = 8
# batch size
trainBatchSize = 256
predBatchSize = 16384
# Number of LRU queues
nrQueues = 500
# nr of page indices per physical page address
outputClasses = 64
# nr of stacked residual blocks
resBlocks = 1
# how many prefetches to issue for each observed memory access (Max = 2)
degree = 2


""" 
            Derived
"""

if feature == version.HALF_PC:
    nrFeatureBits = 24
elif feature == version.FULL_PC:
    nrFeatureBits = 48
elif feature == version.PAGE:
    nrFeatureBits = 36

# nr of inputs to the TCN (nr block index bits = 6)
inputLength = nrFeatureBits + 6
