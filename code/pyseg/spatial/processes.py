"""
Set of processes for being in mult-processor mode within spatial package

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 11.01.17
"""

__author__ = 'martinez'

####### HELPING FUNCTIONS

####### PROCESSES

# Radial Averaged Fourier Transform alone process
# points: array with points coordinates
# ids: indexes to points array
# dense: dense float typed tomogram (even dimensions)
# mask: binary mask (even dimensions)
# mpa: shared multiprocessing object with the results
def pr_uni_RAFT(points, ids, dense, mask, mpa):
    # TODO
    pass
