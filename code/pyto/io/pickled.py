"""
Contains class Pickled for input of data stored in multiple pickle
files. Each pickle contains data for one experiment (observation).

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: pickled_data.py 336 2008-12-27 17:10:30Z vladan $
"""
from __future__ import unicode_literals
from __future__ import absolute_import

__version__ = "$Revision: 336 $"

import sys
import pickle
import warnings
import logging
import numpy

import pyto.util.attributes
from .multi_data import MultiData


class Pickled(MultiData):
    """
    The main purpose of this class is to read data from one or more experiments
    saved as a pickle file, and organize the data into a structre that hold 
    data from several experiments (..analysis.Observation class). 

    Typical usage:

      pick = Pickled(files='dictionary_of_files')
      for multi, obj, category, identifier in pick.readPropertiesGen(...):
        # put here direct manipulations of the current unpickled object
        obj. ...
      # final multi contains all data
      multi. ...
    """

    def __init__(self, files):
        """
        Initializes files attribute.

        Argument files has to be a dictionary of dictionaries, where ouside
        keys are group names, inside keys experiment identifiers and 
        inside values file names. For example:

        files = {'group_a' : {'exp_1' : file_1,
                              'exp_2' : file_2,
                              ...             },
                 'group_b' : {'exp_5' : file_5,
                              ...             },
                 ...                           }
        """
        super(Pickled, self).__init__(files=files)

        # set attributes
        self.compact = False 

    def getSingle(self, category, identifier, encoding='latin1'):
        """
        Returns object containig data for one experiment (observation), in
        other words reads a pickle file corresponding to the specified 
        category and identifier.

        If the data file does not exist returns None.

        In order to load a python2 generated pickle file that contains 
        numpy arrays from python 3, pickle.load() has to have arg
        encoding='latin1' (here set as the default).

        Arguments:
          - category: observation category
          - identifier: onservation identifier
          - encoding: encoding for pickle.load(), not used for python 2
        """
        try:
            file_ = open(self.files[category][identifier], 'rb')
            if sys.version_info[0] > 2:
                obj = pickle.load(file_, encoding=encoding)
            else:
                obj = pickle.load(file_)

        except IOError:
            logging.warning("File " + self.files[category][identifier] \
                                + " could not be read")
            obj = None
        return obj

