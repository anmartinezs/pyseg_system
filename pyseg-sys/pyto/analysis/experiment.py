"""
Defines class Experiment that holds data form one experiment. 

This class is not meant to really hold data (Observations is the main
data-holding class). It is only used for some special purposes.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: experiment.py 1275 2015-12-23 14:31:51Z vladan $
"""

__version__ = "$Revision: 1275 $"


import sys
import warnings
import logging
from copy import copy, deepcopy

import numpy
import scipy


class Experiment(object):
    """
    """

    def __init__(self, identifier=None):
        """
        Initialization

        Argument:
          - identifier: experiment identifier
        """
        self.identifier = identifier
        self.indexed = set()
        self.properties = set()

    def getValue(self, name=None):
        """
        Returns the value of the property specified by arg name.

        Argument:
          - name: (str) property name

        Returns: the value of the specified property
        """
        value = getattr(self, name)
        return value
        
    @classmethod
    def transformByIds(
            cls, ids, new_ids, values, default=-1, mode='vector_1-1'):
        """
        Given an array (arg values) whose elements are ordered according to 
        given ids (arg ids), returns new values array where the elements
        are ordered according to arg new_ids.

        Values can have different form, depending on the arg mode:
          - 'vector_1-1': ids and values directly correspond to each other,
          so vectors is an array of the same length as ids
          - 'square': values correspond to all pairs of ids where order of 
          elements in a pairs matters, so values is a square matrix of the 
          same size as ids
          - 'vector_pair': values correspond all pairs of ids where order of 
          elements in a pair does not matter (just like what's returned from
          scipy.spatial.distance.pdist()), but values is a vector of the 
          of length n(n-1)/2, where n is number (length) of ids.

        If an id exists in new ids but not in ids, the corresponding value 
        is set to arg default. On the other hand, a value corresponding to 
        an id (from ids) that is not in new_ids is ignored.

        Arguments:
          - ids: (list or array) ids
          - new_ids: (list or array) new ids
          - values: (list or array) values corresponding to ids
          - default: (list or array) default value

        Returns new values in the same form as ids. 
        """

        if mode == 'vector_1-1':

            # initialize new values array
            max_new = numpy.max(new_ids)
            new_values_exp = numpy.zeros(max_new+1) + default

            # put old values in
            for id_, val in zip(ids, values):
                new_values_exp[id_] = val

            # compress new values array
            new_values = new_values_exp[new_ids]

        elif mode == 'vector_pair':

            # expand to square
            values_sq = scipy.spatial.distance.squareform(values)
            #values_sq = values_sq + numpy.diag(
            #    numpy.zeros(values_sq.shape[0]) + values_sq.max() + 1)
            
            new_values_sq = cls._transformByIdsSquare(
                ids=ids, new_ids=new_ids, values=values_sq, default=default)

            # back to vector_pair form
            triu_inds = numpy.triu_indices(len(new_ids), 1)
            new_values = new_values_sq[triu_inds]

        elif mode == 'square':
            new_values = cls._transformByIdsSquare(
                ids=ids, new_ids=new_ids, values=values, default=default)

        else:
            raise ValueError(
                "Argument mode: " + mode + " was not understood. Possible "
                + "values are 'vector_1-1', 'vector_pair' and 'square'.")

        return new_values

    @classmethod
    def _transformByIdsSquare(cls, ids, new_ids, values, default=-1):
        """
        """

        # initialize new values array
        max_new = numpy.max(new_ids)
        new_values_exp = numpy.zeros((max_new+1, max_new+1)) + default

        # put old values in the expanded new array
        for (i, j), val in numpy.ndenumerate(values):
            try:
                new_values_exp[ids[i], ids[j]] = val
            except IndexError:
                pass

        # compress new values array
        new_values = new_values_exp[new_ids,:][:,new_ids]

        return new_values

    def doCorrelation(self, xName, yName, test=None, regress=False, 
                      reference=None, new=True, out=sys.stdout, 
                      format_={}, title=''):
        """
        Correlates specified data.

        Uses Observations.doCorrelation to do the job.

        Arguments:
          - xName, yName: property names of data 
          - test: correlation test, currently 'r' (or 'pearson'), 'tau' 
          (or 'kendall')
          - regress: flag indicating if regression (best fit) line is calculated
          - reference:
          - new: flag indicating if new instance is created and returned
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first

        Returns object of this class containing data and results (only if new 
        is True).
        """

        # wrap in Observations
        from observations import Observations
        obs = Observations()
        obs.addExperiment(experiment=self)

        # correlation
        if new:
            obs = obs.doCorrelation(
                xName=xName, yName=yName, test=test, regress=regress, 
                reference=reference, new=new, 
                out=out, format_=format_, title=title)
        else:
            obs.doCorrelation(
                xName=xName, yName=yName, test=test, regress=regress, 
                reference=reference, new=new, 
                out=out, format_=format_, title=title)
            self = obs
            
        corr = obs.getExperiment(identifier=self.identifier)

        if new:
            return corr

    def printData(self, names, out=sys.stdout, format_={}, title=None):
        """
        Prints data of this instance.

        Uses Observations.printData() to do the job.

        Arguments:
          - names: list of property (attribute) names to be printed
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name, None for no output
          names list: ['mean', 'std', 'sem', 'n', 'testValue', 'confidence']
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first
        """

        # wrap in Observations
        from observations import Observations
        obs = Observations()
        obs.addExperiment(experiment=self, identifier='')

        # print
        obs.printData(names=names, out=out, format_=format_, title=title)
        
