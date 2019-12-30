"""

    Reduce the tilt series or micrographs within a stack

    Input:  - Micrographs stack
            - *.order file with two columns; tilt angle, dose in e-/A**2
            - Range of valid angles

    Output: - Stack with some micrograph extracted
            - *.order file with only the tilt angles selected
            - *.tlt file with only the tilt angles selected

"""

################# Package import

import os
import copy
import csv
import numpy as np
import sys
import time
import pyseg as ps

###### Global variables

__author__ = 'Antonio Martinez-Sanchez'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/in/rln/tomos'

in_stack = ROOT_PATH + '/syn_11_2_bin2/'