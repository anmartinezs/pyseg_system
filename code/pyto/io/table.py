"""
Contains class Table for read (write in the future?) or results in the 
table form.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from builtins import object

__version__ = "$Revision$"


import string
import re
import numpy

from .multi_data import MultiData


class Table(MultiData):
    """
    Contains methods to read and write tables.

    ToDo: see about ordering by index
    """

    #######################################################
    #
    # Initialization
    #
    #######################################################

    # all profiles are defined here
    profiles_db = {}
    profiles_db['layers'] = {\
        'columns' : [0, 1, 2, 3, 4, 5, 6, 8],
        'names' : ['ids', 'density_mean', 'density_std', 'density_min', 
                   'density_max', 'volume', 'number', 'occupancy'],
        'dtypes' : ['int', 'float', 'float', 'float', 'float', 'int', 'int',
                    'float']
        }
    profiles_db['vesicles'] = {\
        'columns' : list(range(23)),
        'names' : ['ids', 
                   'vesicle_density_mean', 'vesicle_density_std',
                   'vesicle_density_min', 'vesicle_density_max',
                   'membrane_density_mean', 'membrane_density_std',
                   'membrane_density_min', 'membrane_density_max',
                   'interior_density_mean', 'interior_density_std',
                   'interior_density_min', 'interior_density_max',
                   'vesicle_volume', 'membrane_volume', 'vesicle_distance',
                   'vesicle_center_0', 'vesicle_center_1', 'vesicle_center_2',
                   'vesicle_radius_mean', 'vesicle_radius_std', 
                   'vesicle_radius_min', 'vesicle_radius_max'],
        'dtypes' : ['int', 'float', 'float', 'float', 'float',
                    'float', 'float', 'float', 'float',
                    'float', 'float', 'float', 'float',
                    'int', 'int', 'float', 'int', 'int', 'int',
                    'float', 'float', 'float', 'float'],
        'nd' : {'vesicle_center' : ['vesicle_center_0', 
                                    'vesicle_center_1', 'vesicle_center_2']} 
        }

    def __init__(self, files, profile=None):
        """
        Initializes attributes.

        Arguments:
          - files:
          - columns: distionary with column positions as keys and names
          as values 
        """
        super(Table, self).__init__(files=files)

        # set attributes common to all profiles
        self.comment = '#'
        class Tmp(object): pass
        self.singleClass = Tmp
        self.compact = True

        # set profile-dependent attributes
        self.setProfile(profile=profile)

    def setProfile(self, profile):
        """
        Sets attributes needed for reading particular data files (profiles).

        Sets:
          - self.profile: subdictionary of profile_db corresponding to
          argument profile

        Arguments:
          - profile: (string) name of the current profile 
        """
        
        # set current profile (so one can change it if needed)
        self.profile = self.profiles_db[profile]

        #
        self.columns = self.profile['columns']
        self.names = self.profile['names']
        self.dtype = numpy.dtype({\
                'names' : tuple(self.profile['names']),
                'formats' : tuple(self.profile['dtypes'])})
                                     
            
    ##########################################################
    #
    # Other methods
    #
    #########################################################

    def getSingle(self, category, identifier):
        """
        """

        # get file name
        file_name = self.files[category][identifier]
                                     
        # read specified columns
        records = numpy.loadtxt(fname=file_name, dtype=self.dtype, 
                                usecols=self.columns)
        
        # make instance to hold this data
        single_inst = self.singleClass()

        # assign properties
        for nam in self.names:
            setattr(single_inst, nam, records[nam])

        # make nd arrays when needed
        single_inst = self.makeNd(single_inst)

        return single_inst

    def makeNd(self, single):
        """
        Make nd arrays (from arrays of single) according to rules given in
        self.profile['nd'].

        Argument:
          - single
        """
        try:
            for nd_name in self.profile['nd']:
                nd_var = [getattr(single, oned_name) \
                              for oned_name in self.profile['nd'][nd_name]]
                nd_var = numpy.asarray(nd_var)
                nd_var = nd_var.transpose()
                setattr(single, nd_name, nd_var)
        except KeyError:
            pass

        return single

    def reorder(self, ids, arrays=None, names=None):
        """
        Not used in the moment (27.12.08), but might be needed.

        Orders elements of arrays according to the ids array.
        
        Arrays can be specified as a list of ndarrays (argument arrays),
        or by names (argument names).

        Arguments:

        """

        # use attributes if arguments not set
        #if names is None:
        #    names = self.names

        # order ids
        ids = numpy.asarray(ids)
        order = ids.argsort()

        # reorder arrays given directly
        if arrays is not None:
            if isinstance(arrays, numpy.ndarray):
                arrays = [arrays]
            res = [ar[order] for ar in arrays]
                
        # reorder arrays given by names
        if names is not None:
            for nam in names:
                self.__dict__[nam] = self.__dict__[nam][order]

        # return in the same form as in the argument
        if arrays is not None:
            if isinstance(arrays, numpy.ndarray):
                return res[0]
            else:
                return res 

    def getVersion(self, file):
        """
        Finds version of the results file.
        
        First it reads file to find the svn revision number of the script that
        was used to write this results file. Then sets the version of the
        results file to be the highest element of self.versions that is still
        lower than the revision.

        Requires self.versions, which is normally not specified in this class.
        This method should be called from instances of classes that inherit from
        this class and have self.versions.set (such as Connections).

        Argument:
          - file: results file

        Retirns: (int) version
        """

        # open results file
        if isinstance(file, str):
            file = open(file)

        # find results file revision
        for line in file:
            revision = self.enclosedString(string=line, begin='$Revision:',
                                           end='$')
            if revision is None:
                continue
            else:
                revision = int(revision.strip())
                break
        else:
            revision = 0

        # find appropriate version
        reverse_vers = [ver for ver in self.versions]  # copy
        reverse_vers.sort()
        reverse_vers.reverse()
        for ver in reverse_vers:
            if revision >= ver: 
                version = ver
                break

        return version

    def enclosedString(self, string, begin, end):
        """
        Returns part of string between strings begin and end.
        """

        # 
        begin = re.escape(begin)
        end = re.escape(end)

        # find 
        whole_re = re.compile(begin + '.*' + end)
        match = whole_re.findall(string)
        if len(match) == 0:
            return None

        begin_re = re.compile(begin)
        match = begin_re.sub('', match[0])
        end_re = re.compile(end)
        match = end_re.sub('', match)
        
        return match
