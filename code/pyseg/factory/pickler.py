"""
Utilities for pickling a unpickling PySeg objects

# Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry)
# Date: 22.10.14
"""

__author__ = 'martinez'

try:
    import pickle as pickle
except:
    import pickle

# Unpickle a Generic PySeg object
def unpickle_obj(fname):

    # Load pickable state
    f_pkl = open(fname, 'rb')
    try:
        gen_obj = pickle.load(f_pkl)
    finally:
        f_pkl.close()

    return gen_obj
