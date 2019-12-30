"""
Defines class Groups that can hold data form one or more observations 
(experiments) divided (classified) in groups. 

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: groups.py 1517 2019-02-20 11:33:20Z vladan $
"""

__version__ = "$Revision: 1517 $"


import sys
import os
import warnings
import logging
from copy import copy, deepcopy
import re
import imp

import numpy
import scipy

from observations import Observations
import pyto.util.nested

class Groups(dict):
    """
    Data from any number of observations (also called experiments) is 
    classified into one or more groups. Typically groups are based on 
    different experimental conditions or traits observed in individual 
    observations.

    Each group of observations (class Observations) is accesed as values in
    a dictionary where keys are group names. A group can also be accessed 
    as an attribute of this instance. Consequently, my_group['treated'] and
    my_group.treated are the same.

    This class needes to be subclassed. Data is read using read() method that
    need to be defined in each subclass.
    """
    
    #######################################################
    #
    # Initialization
    #
    #######################################################

    def __init__(self):
        """
        Doesn't do almost anything.
        """

        # define property names for fraction-like data properties
        self._fract_data_names = {
            'fraction':'fraction_data', 'histogram':'histogram_data',
            'probability':'probability_data'}

    def fromList(self, groups, names):
        """
        Adds groups specified in arg groups to this instance.

        Arguments:
          - groups: (list of Observations objects) observations objects that
          form this object
          - names: (list of strings): names of the groups, the length has to be
          the same as the length of arg groups
        """

        if groups is not None:

            # check
            if (names is None) or (len(groups) != len(names)):
                raise ValueError("Args groups and names have to be lists of "
                                 + "same lengths.")

            # add observations
            for group, name in zip(groups, names):
                self[name] = group

    def recast(self):
        """
        Returns a new instance of this class and sets all attributes of the new 
        object to the attributes of obj.

        Useful for objects of this class loaded from a pickle that were pickled
        before some of the current methods of this class were coded. In this
        case the returned object contains the same attributes as the unpickled 
        object but it accepts recently coded methods. 
        """

        # make a new instance
        new = self.__class__()

        # set Observations
        for categ in self.keys():
            new[categ] = self[categ].recast()

        return new

    def __getattr__(self, key):
        """
        Allows accessing an item as if it was an attribute, but not for 
        attributes starting with '_'.
        """
        if key.startswith('_'):
            result = super(Groups, self).__getattr__(key)
        else:
            result = self.__getitem__(key)
        return result

    def __setattr__(self, key, value):
        """
        Allows setting an item as if it was an attribute, but not for 
        attributes starting with '_'.
        """
        if key.startswith('_'):
            super(Groups, self).__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __delattr__(self, key):
        """
        Allows deleting an item as if it was an attribute, but not for 
        attributes starting with '_'.
        """
        if key.startswith('_'):
            super(Groups, self).__delattr__(key)
        else:
            self.__delitem__(key)


    #######################################################
    #
    # Methods that unite, separate or rearrange observations
    #
    #######################################################

    def regroup(self, name, categories=None, identifiers=None):
        """
        Returnes a new instance of this class where the observations of this 
        instance are rearanged into new groups according to values of
        the specified property.

        The values of the specified property are used as the groop names of
        the new (rearanged) object. If these values are not strings (str type),
        they are converted to strings by str(value).

        Arguments:
          - name: name of the property used to distinguish groups
          - categories: only experiments belonging to these categories (group 
          names) are used, if None all categories are used
          - identifiers: only experiments having these experiment identifiers 
          are used, if None all identifiers are used

        Returns: new instance of this class
        """

        # make new instance
        rearranged = self.__class__()
        
        for categ, ident, exp in self.experiments(categories=categories, 
                                                  identifiers=identifiers):

            # add experiment to the appropriate group (make group if needed)
            new_categ = str(exp.getValue(name=name))
            if rearranged.get(new_categ) is None:
                rearranged[new_categ] = Observations()
            rearranged[new_categ].addExperiment(experiment=exp, 
                                                identifier=ident)

        return rearranged

    def pool(self, categories, name):
        """
        Unites specified categories into a new category. 

        The new category shares the data with the original categories. The
        other categories are not modified.

        Arguments:
          - categories: (list) categories to be united
          - name: new category name (key)
        """

        # set data
        self[name] = copy(self[categories[0]])
        for categ in categories[1:]:
             self[name].join(self[categ])

    def addGroups(self, groups, copy=False):
        """
        Adds all groups of the specified Groups object (arg groups) to this
        objects. The groups are added separately. The added data 

        If arg copy is True, the individual groups of arg groups are 
        deepcopied and the copies are added to this instance. Otherwise
        the groups are shared. 

        If group names from this and the specified Groups objects overlap,
        ValueError is raised.

        Modifies this instance. 

        Argument:
          - groups: Groups instance
          - copy: Flag indicating whether the added groups are copied 
        """

        # check for overlap
        overlap = set(self).intersection(groups)
        if len(overlap) > 1:
            raise ValueError("The following groups overlap: " + str(overlap)) 

        # add groups
        for name, one_group in groups.items():
            if copy:
                one_group = deepcopy(one_group)
            self[name] = one_group 

    def experiments(self, categories=None, identifiers=None):
        """
        Generator that yields all identifiers and observations (experiments)
        of this instance.

        If arg identifier is specified, the order of yielded objects follows
        the order of arg identifiers.

        Arguments:
          - categories: only experiments belonging to these categories (group 
          names) are yielded, if None all categories are used
          - categories: only experiments having these experiment identifiers 
          are yielded, if None all identifiers are used

        Yields: group_name, identifier, observation (Experiment)
        """

        # get all identifiers and the corresponding group names, but keep
        # only those specified by arguments
        ident_group = {}
        for group_name, group in self.items():
            if (categories is not None) and (group_name not in categories):
                continue
            for ident in group.identifiers:
                if (identifiers is not None) and (ident not in identifiers):
                    continue
                if ident_group.get(ident) is None:
                    ident_group[ident] = []
                ident_group[ident].append(group_name)

        # make identifiers list unles specified already
        if identifiers is None:
            identifiers = ident_group.keys()

        # yield in the order specified by identifiers
        for ident in identifiers:
            for group_name in ident_group[ident]:
                group = self[group_name]

        #for group_name, group in self.items():
        #    if (categories is not None) and (group_name not in categories):
        #        continue

        #    for ident in group.identifiers:
        #        if (identifiers is not None) and (ident not in identifiers):
        #            continue

                # get and yield experiment
                experiment = group.getExperiment(identifier=ident)
                yield group_name, ident, experiment

    def addData(self, source, names, groups=None, identifiers=None, copy=True):
        """
        Adds properties listed in arg names of another Groups object
        (arg source) to this instance. 

        In arg names is a list, added properties retain their names, and so 
        will overwrite the properties having same names of this instance (if 
        they exist). Otherwise, if arg names is a dictionary, the keys are the 
        property names in source object and values are the corresponding names 
        under which they are added to this object.

        All specified group names (arg groups) and / or identifiers have to 
        exist in both objects. If arg groups / identifiers is None, groups / 
        identifiers of arg source have to exist in this object.

        If arg copy is True, a copy of data is saved to the other object 
        (copy() method for numpy.ndarrays, deepcopy for the rest).

        Arguments:
          - source: (Observations) another object
          - names: list of property names of the source object, or a dictionary
          where keys are the property names in source and values are the
          new names
          - groups: list of group names
          - identifiers: list of experiment identifiers for which the data is
          copied. Identifiers listed here that do not exist among 
          identifiers of source are ignored.  
          - copy: Flag indicating if data is copied 
        """

        # set groups
        if groups is None:
            groups = source.keys()

        # add data to each group
        for group_name in groups:

            self[group_name].addData(source=source[group_name], names=names,
                                     identifiers=identifiers, copy=copy)
        
    def remove(self, identifiers, groups=None):
        """
        Removes all data for all experiments specified by arg identifiers.
        Experiments are removed for the specified groups, or if arg groups
        is None for all groups of this instance.

        Arguments:
          - identifiers: list of experiment identifiers
          - groups: list of group names or None for all groups
        """
        
        # repeat for each group 
        for group_name, group in self.items():

            # skip if current group in specified groups
            if groups is not None:
                if group_name not in groups:
                    continue

            # remove
            for ident in identifiers:
                group.remove(identifier=ident)
       
    def keep(self, groups=None, identifiers=None, removeGroups=False):
        """
        Keeps only the experiments specified by arg identifiers and removes 
        data corresponding to all other experiments. 

        If arg removeGroups is False, experiments are removed only for the 
        specified groups, while the nonspecified groups are not affected. 
        
        Alternatively, if arg removeGroups is True, groups that are not
        specified are (entirely) removed. In other words, only the specified
        experiments of the specified gropus are kept.

        If arg identifiers is None, all identifiers are kept. Similarly, if arg 
        groups is None, the behaviour is the same as if all groups were 
        specified. 

        Arguments:
          - identifiers: list of experiment identifiers
          - groups: list of group names or None for all groups
          - removeGroups: flag indicating if non-specified groups are removed 
         """
        
        # repeat for each group 
        for group_name, group in self.items():

            # skip if current group in specified groups
            if groups is not None:
                if group_name not in groups:
                    if removeGroups:
                        self.pop(group_name)
                    continue

            # keep specified
            if identifiers is not None:
                group.keep(identifiers=identifiers)

    def extract(self, condition):
        """
        Returns an instance of this class that contain only those  
        observations for which the condition is satisfied.

        Argument:
          - condition: object that have the same structure as this instance 
        """
            
        extracted = self.__class__()
        for categ in self:
            extracted[categ] = self[categ].extract(condition=condition[categ])
        return extracted

    def extractIndexed(self, condition):
        """
        Extracts elements of individual observations of this instance, according
        to the arg condition and returns an object of this class containig 
        the extracted elements.

        The structure of arg condition has to correspond to the structure of 
        this instance, that is it has to be a dictionary with the same keys as
        in this instance, each of its values has to be a list of the same
        number of elements as the number of observations, and each of these
        elements has to be a ndarray containing elements corresponding to the
        elements of the indexed properties of this instance. 

        Elements of this instance which correspond to True elements in arg 
        condition are extracted.  

        Elements are extracted from all indexed properties (as specified in 
        self.indexed), while the other properties are copied in full.

        Arguments:
          - condition: condition
        """

        extracted = self.__class__()
        for categ in self:
            extracted[categ] = \
                self[categ].extractIndexed(condition=condition[categ])
        return extracted

    def splitIndexed(self):
        """
        Splits this instance according to the indexed properties.

        Returns a list of instances of this class where each instance contains
        only one value of each of the indexed properties for each category 
        (the returned instances have the same categories as this instance). 
        Other (non-indexed) properties are copied from this to each of the 
        resulting instances.

        If indexed properties of observations comprising this instance contain 
        different number of elements, only the common elements are returned.
        That is, the length of the returned list is the minimum number of
        elements that indexed properties have.

        Returns: list of instances of this class.
        """

        # split each category
        for categ in self:

            # split observations
            split_obs = self[categ].splitIndexed()

            # save values for the current category (initialize when needed)
            for obs_ind in range(len(split_obs)):
                try:
                    split[obs_ind][categ] = split_obs[obs_ind]
                except NameError:
                    split = [self.__class__() for init_ind 
                             in range(len(split_obs))]
                    split[obs_ind][categ] = split_obs[obs_ind]
            
        return split

    def split(self, value, name, categories=None):
        """
        Splits this object into one or more objects depending of the values
        of the attribute specified by arg name and returns the object(s).

        If value is a single number, the returned object if formed from all
        vesicles of all observations and for all categories that have the
        value of property given by arg name between 0 and arg value (limits
        inclusive).

        If value is a list of numbers, they specify bins and a list of
        objects (one for each bin) is returned. Lower bin limits are 
        inclusive, while the upper are exclusive, except for the upper limit 
        of the last bin which is inclusive (like numpy.histogram).

        If value is a single number single sv object is returned. Otherwise,
        if value is a list of numbers, a list of sv objects is returned.

        Arguments:
          - value: list of values, interpreted as value bins, or if 
          a single number it is a higher value limit, while 0 is the lower
          - name: (string) name of the attribute whose values are compared 
          with arg values
          - categories:

        Returns a list of (if arg value is a list), or one (if arg values is 
        a single number) obect(s) of this class
        """

        # get categories
        if categories is None:
            categories = self.keys()

        # check if one value or value bins
        if not isinstance(value, (list, numpy.ndarray)): 
            value = [0, value]
            one = True
        else:
            one = False

        # initialize resulting structure
        result = []
        n_bins = len(value) - 1
        for ind in range(n_bins):
            result.append(self.__class__())

        # split each category
        for categ in categories:

            # split
            curr_result = self[categ].split(bins=value, name=name)

            # assign to resulting structure
            for bin_ind in range(n_bins):
                result[bin_ind][categ] = curr_result[bin_ind]

        if one:
            return result[0]
        else:
            return result

    def splitByDistance(self, distance, name='distance_nm', categories=None):
        """
        Returns a list of Groups objects, where each object contains data
        for elements of observations whose distances (attribute specified by 
        name) fall into bins specified by arg distances.

        Lower distance bin limits are inclusive, while the upper are exclusive, 
        except for the upper limit of the last distance bin which is inclusive 
        (like numpy.histogram)

        If distance is a single number single sv object is retutred. Otherwise,
        if distance is a list of numbers, a list of sv objects is returned.

        Arguments:
          - distance: list of distances, interpreted as distance bins, or if 
          a single number it is a higher distance limit, while 0 is the lower
          - name: name of the distance attribute (default 'dist_nm')
          - categories:
        """
        
        if not isinstance(distance, (list, numpy.ndarray)):
            distance = [0, distance]

        return self.split(value=distance, name=name, categories=categories)

    def joinExperiments(
            self, name, mode='join', bins=None, fraction=None,
            fraction_name='fraction',
            groups=None, identifiers=None, removeEmpty=True):
        """
        Creates new Observations instance by joining data (specified by arg
        name) of all experiments belonging to one group. Essentially, groups
        comprising this instance (Groups) become observations of the resulting
        instance (Observations).

        Data of individual experiments (observations) can be joined in the 
        following ways (arg mode):
          - 'join': Data is pooled across experiments of the same group
          - 'mean': The mean values of data for all experiments are pooled
          accross experiments of the same group
          - 'mean_bin': For each experiment, data is binned (arg bins), the
          resulting histogram is normalized to 1 and data from bin number
          given by arg fraction is used. These values are pooled accross
          experiments of the same group and saved as property (arg)
          fraction_name. Args bins and fraction have to be specified.

        The joined data is ordered according to (the order of) arg identifiers.

        The resulting instance has the following properties: 
          - identifiers: same as group names of this instance, the order of
          identifiers is the same as the order of self.keys().
          - data name(s): the name of the data property stays the same
          - ids: set from 1 up (increment 1), correspond to each data point of
          the resulting instance 
          - idNames: unique string for each data value derived from identifier 
          and id for that value, see below.

        If data is scalar (one value per experiment), or the mode is 'mean' 
        experiment identifiers of this instance are saved as attribute 
        idNames of the new Observations. Alternatively, if data is indexed and
        mode is 'join' idNames of the new Observations for each data element is
        composed as identifier_id where identifier and id are specifying this
        data element. 

        Arg remove empty determines what to do if there is no data 
        in an experiment in case arg mode is 'mean' and the data is 
        indexed. If True, experiment that has no data is ignored, and the
        identifier of this experiment is not added to idNames. If False,
        numpy.NaN is added instead of the mean.

        Arguments:
          - name: name of the data attribute
          - mode: defines how experiment data are joined: 'join', 'mean' or
          'mean_bin'
          - bins: bins used for binning data, only if mode is 'mean_bin'
          - fraction: bin index, only if mode is 'mean_bin'
          - histogram_name:
          - fraction_name:
          - groups: (list) names of groups whose experiments are joined 
          - identifiers: list of experiment identifiers to be used here, if 
          None all are used. Non-existing identifiers are ignored. 
          - removeEmpty: Flag that determines what to do if there is no data 
          in an experiment. Used only if arg mode is 'mean' and the data is 
          indexed. 

        Returns: Observations instance
        """

        # check bin-related argumens
        if mode == 'mean_bin':
            if ((bins is None) or (fraction is None)):
                raise ValueError(
                    "Arguments bins and fraction have to be specified when "
                    + "arg mode is 'mean_bin'.")
        else:
            if ((bins is not None) or (fraction is not None)): 
                raise ValueError(
                    "Arguments bins and fraction should not be specified when "
                    + "arg mode (" + mode + ") is different from 'mean_bin'.")
        
        # initialize new Observations
        obs = Observations()
        obs.properties.add('identifiers')

        # check name
        # probably not needed (not in doc string but in test_groups) 01.2019
        if isinstance(name, str):
            name_list = [name]
        else:
            name_list = name

        # set data, ids and id_names
        for categ, group in self.items():

            # skip groups that are not listed 
            if (groups is not None) and (categ not in groups):
                continue

            # set identifiers
            if identifiers is None:
                loc_idents = group.identifiers
            else:
                loc_idents = identifiers

            # set id_names
            id_names = []
            for ident in loc_idents:

                # skip groups whose identifiers that are not listed
                if ident not in group.identifiers:
                    continue

                # set id names
                value = group.getValue(property=name_list[0], identifier=ident)
                if name_list[0] in group.indexed:

                    # indexed property
                    if mode == 'join':
                        curr_ids = group.getValue(property='ids', 
                                                  identifier=ident)
                        id_names.extend([ident + '_' + str(id_) 
                                         for id_ in curr_ids])
                    elif ((mode == 'mean') or (mode == 'mean_bin')):
                        if (len(value) > 0) or not removeEmpty:
                            id_names.append(ident)
                    else:
                        raise ValueError(
                            "Mode " + str(mode) + " not understood, Valid "
                            + "values are 'join', 'mean' and 'mean_bin'.")

                else:

                    # scalar property
                    id_names.append(ident)

            # add to the new group data
            obs.setValue(property='idNames', identifier=categ, value=id_names, 
                         indexed=True)

            # set all data of all experiments 
            for name in name_list:

                # set data
                data = []
                hist = None
                for ident in loc_idents:

                    # skip groups whose identifiers that are not listed
                    if ident not in group.identifiers:
                        continue

                    # set current data of the current experiment
                    value = group.getValue(property=name, identifier=ident)
                    old_indexed = name in group.indexed
                    if old_indexed:

                        # indexed property
                        if mode == 'join':
                            data.extend(value)
                        elif (mode == 'mean'):
                            if len(value) > 0:
                                data.append(numpy.mean(value))
                            elif not removeEmpty:
                                data.append(numpy.NaN)
                        elif (mode == 'mean_bin'):
                            if len(value) > 0:
                                hist, foo = numpy.histogram(value, bins)
                                frac_value = hist[fraction] / float(len(value))
                                #if hist is not None:
                                #    histogram = histogram + hist
                                #else:
                                #    histogram = hist
                                data.append(frac_value)
                            elif not removeEmpty:
                                data.append(numpy.NaN)

                    else:

                        # scalar property
                        data.append(value)

                # probably not needed, replace by indexed=True (01.2019)
                if name == name_list[0]:
                    indexed = True
                else:
                    indexed = ((name in group.indexed) 
                               == (name_list[0] in group.indexed))

                # add to the new group data
                if mode is not 'mean_bin':
                    obs.setValue(property=name, identifier=categ, 
                                 value=numpy.array(data), indexed=indexed)
                else:
                    #obs.setValue(
                    #    property=histogram_name, identifier=categ, 
                    #    value=histogram, indexed=True)
                    obs.setValue(
                        property=fraction_name, identifier=categ, 
                        value=numpy.array(data), indexed=indexed)
            # set ids
            if mode is not 'mean_bin':
                tmp_name = name_list[0]
            else:
                tmp_name = fraction_name
            data_0 = obs.getValue(property=tmp_name, identifier=categ)
            ids = range(1, len(data_0)+1)
            obs.setValue(property='ids', identifier=categ, value=ids, 
                         indexed=True)

        return obs

    @classmethod
    def joinExperimentsList(
            cls, list, listNames, name, groups=None, 
            identifiers=None, mode='join', removeEmpty=True):
        """
        Creates new Groups instance by joining data (specified by arg
        name) of all experiments belonging to one group. That is, for each
        Groups objects given in arg list, individual groups (Observations
        object) are joined to become individual observations. Each Groups
        object given becomes Observations object in the resulting Groups 
        object.

        Data of individual experiments (observations) are either joined
        together (mode 'join'), or an average value of each experiment is 
        used (mode 'mean') for the resulting instance.

        All Observations objects of the resulting Groups instance have the 
        following properties: 
          - identifiers: same as group names of Groups objects of arg list, 
          the order of identifiers is the same as the order of the
          corresponding groups_instance.keys().
          - data name: the name of the data property stays the same
          - ids: set from 1 up (increment 1), correspond to each data point of
          the resulting instance 
          - idNames: unique string for each data value derived from identifier 
          and id for that value, see below.
        Values of arg listNames become group names of the resulting object.

        If data is scalar (one value per experiment), or the mode is 'mean' 
        experiment identifiers are saved as attribute idNames of the new 
        Observations objects. Alternatively, if data is indexed and
        mode is 'join' idNames of the new Observations for each data element is
        composed as identifier_id where identifier and id are specifying this
        data element. 

        Arg remove empty determines what to do if there is no data 
        in an experiment in case arg mode is 'mean' and the data is 
        indexed. If True, experiment that has no data is ignored, and the
        identifier of this experiment is not added to idNames. If False,
        numpy.NaN is added instead of the mean.

        Arguments:
          - list: list of Groups objects
          - listNames: (list of strings) names of the Groups objects given
          in arg list
          - name: name of the data attribute
          - groups: (list) names of groups whose experiments are joined 
          - identifiers: list of experiment identifiers to be used here, if 
          None all are used. Non-existing identifiers are ignored. 
          - mode: 'join' to join or 'mean' to average data
          - removeEmpty: Flag that determines what to do if there is no data 
          in an experiment. Used only if arg mode is 'mean' and the data is 
          indexed. 

        Returns: Groups instance
        """

        # make new Groups object
        new_groups = cls()
        
        # loop over list items (Groups objects)
        for _groups, new_name in zip(list, listNames):

            # join to Observations
            obs = _groups.joinExperiments(
                name=name, groups=groups, identifiers=identifiers, mode=mode,
                removeEmpty=removeEmpty)

            # add the Observations to the now Groups
            new_groups[new_name] = obs

        return new_groups

    def transpose(self):
        """
        Rearranges experiments so that groups and observations are exchanged
        and returnes a new (rearranged) instance of this class.

        For example, if this instance contains:
          group a: exp 1, exp 5, exp 7
          group b: exp 1, exp 5, exp 7
        then the rearranged object is:
          group 1: exp a, exp b
          group 5: exp a, exp b
          group 7: exp a, exp b

        All groups have to have same experiment identifiers, otherwise a
        ValueError ir raised. They are also expected to have the same
        properties and indexed attributes.
        
        Properties reference and referenceGroup are interchanged for each 
        experiment. If an experiment of the original object doesn't have 
        referenceGroup the it is assumed that it is the (name of the) original 
        group and the reference of the same experiment in the transposed
        object is set accordingly.

        Does not modify any of the data.
        """

        # get old identifiers and check that they're the same for all groups
        old_identifiers = None
        for name, group in self.items():

            # get identifiers, properties and indexed
            if old_identifiers is None:
                old_identifiers = group.identifiers
                properties = group.properties
                indexed = group.indexed

            else:

                # check identifiers
                if set(old_identifiers) != set(group.identifiers):
                    raise ValueError(
                        "Can't transpose Groups object because individual " 
                        + "groups comprising this instance don't " 
                        + "have the same identifiers")

        # transpose
        new = self.__class__()
        for new_group_name in old_identifiers:

            # instantiate new group and add properties and indexed 
            new_group = Observations()
            new_group.properties = copy(properties)
            new_group.indexed = copy(indexed)

            # make a new group
            for new_identifier, old_group in self.items():

                # pick an experiment
                exp = old_group.getExperiment(new_group_name)
                exp.identifier = new_identifier
                new_group.addExperiment(experiment=exp)

                # interchange reference and referenceGroup
                if 'reference' in new_group.properties:
                    ref = new_group.getValue(identifier=new_identifier, 
                                             property='reference')
                    if 'referenceGroup' in new_group.properties:
                        ref_g = new_group.getValue(identifier=new_identifier, 
                                                   property='referenceGroup')
                    else:
                        ref_g = new_identifier
                    new_group.setValue(identifier=new_identifier, 
                                       property='referenceGroup', value=ref)
                    new_group.setValue(identifier=new_identifier, 
                                       property='reference', value=ref_g)

            # add the new group  
            new[new_group_name] = new_group

        return new

    def isTransposable(self):
        """
        Returns True if this object can be transposed (method transpose()),
        that is if all groups have the same experiment identifiers.
        """

        first_group = True
        for name, group in self.items():
            
            # first time around
            if first_group:
                old_idents = set(group.identifiers)
                first_group = False

            #  check
            idents = set(group.identifiers)
            if idents != old_idents:
                return False
            old_idents = idents

        return True


    #######################################################
    #
    # Statistics
    #
    #######################################################

    def joinAndStats(
        self, name, mode='join', bins=None, fraction=None,
        fraction_name='fraction', groups=None, 
        identifiers=None, test=None, reference=None, ddof=1, out=sys.stdout, 
        outNames=None, format_=None, title=None):
        """
        Does statistics on data (specified by arg name) for each group
        separately, and tests for statistical difference between groups and
        a specified reference group(s).

        Argument mode determines how the data is pooled across experiments.
        If mode is 'join', data of individual experiments (observations) are 
        joined (pooled)  together within a group to be used for further 
        analysis. If it is 'mean', the mean value for each experiment is 
        calculated and these means are used for further analysis.

        Argument bins determined how the above obtained data is further 
        processed. If arg bins is not specified, basic stats (mean, std, sem)
        are calculated for all groups and the data is statistically compared 
        among the groups. Alternatively, if arg bins is specified, histograms 
        of the data are calculated for all groups, normalized to 1 and 
        statistically compared between groups. 

        Modes 'join_bins' and 'byIndex' are described below. Specifically,
        the following types of analysis are implemented: 

          - mode='join', bins=None: Data is pooled across experiments of
          the same group, basic stats are calculated within groups and 
          statistically compared between groups.

          - mode='join', bins specified (not None): Data is pooled across 
          experiments of the same group, histograms (acording to arg bins)
          of the data values are calculated within group and statistically 
          compared among groups.

          - mode='mean', bins=None: Mean values are calculated for all
          experiments, basic stats are calculated for means within groups 
          and statistically compared between groups.

          - mode='mean', bins specified (not None): Mean values are 
          calculated for all experiment, histograms (acording to arg bins)
          of the means are calculated within groups and statistically 
          compared between groups.

          - mode='mean_bin', bins have to be specified (not None): 
          Histograms of the data values are calculated for each experiment 
          (acording to arg bins), normalized to 1 and values of the bin 
          specified by arg fraction are selected (saved as arg fraction_name). 
          Basic stats for this property are calculated and statistically 
          compared within and between groups. 

          - mode='byIndex', bins should not be specified: Basic stats 
          (mean, std, sem) are calculated for each index (position) 
          separately. Data has to be indexed, and all experiments within 
          one group have to have same ids.

        Returns Observations instance, where each experiment (observation) 
        of the returned instance correspond to a group of this instance. 

        In cases data distributions are statistically compared ('join', bins 
        None; 'mean', bins None; 'mean_bin'; 'byIndex'), the returned 
        instance has following attributes:
          - data: deepcopied data that is statistically analyzed
          - mean: mean
          - std: n-ddof degrees of freedom
          - n: number of individual data points
          - sem: standard error of means

        In cases histograms are statistically compared ('join' or 'mean', 
        bins specified), the returned instance has following attributes:
          - ids: left bin limits 
          - data: deepcopied data 
          - histogram: (ndarray of length 1 less than bins) histogram values 
          (indexed)
          - probability: (ndarray of length 1 less than bins) 
          histogram / sum_of_histogram (indexed)
          - n: number of individual data points
          - fraction: histogram[fraction] / n
          - histogram_data, probability_data, fraction_data: the same as the 
          corresponding attributes without '_data' suffix, except that these
          are lists that contain values for each group

        If arg test is specified, inference is calculated between groups 
        and the following attributes are set in the returned instance:
          - testValue: values of the test used
          - testSymbol: specifies the test ('t', 'h', 'u', ...)
          - confidence: probability that data from a group and the/its 
          reference come from the same population
        All of these attributes are lists with elements corresponding to 
        the original groups (experiments of the returned object), except 
        for testSymbol which is a string

        The results are written on a stream specifed by arg out

        Arguments:
          - name: name of the data attribute
          - bins: histogram bins, if specified histogram is calculated
          - fraction: (int) position of the histogram bin that is selected
          (starts from 0)
          - groups: list of the groop names to be analyzed, None for all groups
          - identifiers: list of experiment identifiers to be used here, if 
          None all are used. Non-existing identifiers are ignored. 
          - mode: 'join' to join, 'mean' to average data or 'byIndex'
          - reference: name of the reference group, a single name (string), 
          or a dictonary where each group name (keys) is associated with its 
          reference (values)         
          - test: statistical test used: 't', 'h', 'u', or any other stated
          in Observations.doInference() doc. If None no test is done. 
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name, None for no output
          - outNames: list of sttribute names to be printed, None for a default
          names list: ['mean', 'std', 'sem', 'n', 'testValue', 'confidence']
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first
        """
        
        # keep group order if given
        if groups == None:
            group_names = self.keys()
        else:
            group_names = groups

        if mode == 'byIndex':

            # mode byIndex

            # calculate 
            stats = Observations()
            for g_name in group_names:

                # skip non-specified groups
                if (groups is not None) and (g_name not in groups):
                    continue

                exp = self[g_name].doStatsByIndex(
                    name=name, identifiers=identifiers, 
                    identifier=g_name, ddof=ddof)
                stats.addExperiment(experiment=exp)

        elif ((mode == 'join') or (mode == 'mean')):

            # join and stats
            obser = self.joinExperiments(name=name, mode=mode, groups=groups,
                                         identifiers=identifiers)
            stats = obser.doStats(name=name, bins=bins, fraction=fraction, 
                                  identifiers=groups, new=True, ddof=ddof)

            # add fraction-like data properties for each observation separately
            if bins is not None:
                for g_name in group_names:
                    group = self[g_name]
                    stats_data = group.doStats(
                       name=name, bins=bins, fraction=fraction, 
                       identifiers=identifiers, new=True, ddof=ddof)

                    # in case this object was made before self._fract_data_names
                    # was added to __init__
                    try:
                        for (not_data, yes_data) in \
                            self._fract_data_names.items():
                            try:
                                current_value = getattr(stats_data, not_data)
                            except AttributeError:
                                continue
                            stats.setValue(
                                identifier=g_name, name=yes_data,
                                value=getattr(
                                    stats_data, not_data), indexed=False)
                    except AttributeError:
                        pass
                        
            # inference
            if test is not None:
                if bins is None:
                    infer_name = 'data'
                else:
                    infer_name = 'histogram'
                stats.doInference(test=test, name=infer_name,  
                                  identifiers=groups, reference=reference)

        elif mode == 'mean_bin':

            # join and stats
            obser = self.joinExperiments(
                name=name, mode=mode, bins=bins, fraction=fraction,
                fraction_name=fraction_name, groups=groups,
                identifiers=identifiers)
            stats = obser.doStats(
                name=fraction_name, identifiers=groups, new=True, ddof=ddof)

            # inference
            if test is not None:
                infer_name = 'data'
                stats.doInference(
                    test=test, name=infer_name,  
                    identifiers=groups, reference=reference)

        else:
            raise ValueError(
                "Argument mode " + str(mode) + " was not understood. "
                + "Acceptable values are: 'join', 'mean', 'mean_bin' and"
                + " 'byIndex'.") 
 
        # print
        stats.printStats(out=out, identifiers=groups, names=outNames, 
                         format_=format_, title=title)

        return stats

    @classmethod
    def joinAndStatsList(cls, list, listNames, name, mode='join', groups=None,
                         test=None, between='groups', reference=None, ddof=1):
        """
        Does basic statistics on data (specified by arg name) for each group
        separately, and tests for statistical difference between groups and
        a specified reference group(s).

        Essentially a combination of joinExperimentsList() that joins data
        belonging to a same group (over individual experiments) thus
        making a Groups object, and doStats() that calculates statstics on 
        this Groups object.

        Arg list contains Groups objects as items, and each of these can 
        have one or more groups. If arg between is 'list_items', all Groups 
        objects specified (arg list) have to have the same group names.

        Data of individual experiments (observations) are either joined
        together within a group (mode 'join'), or an average value of each 
        experiment is used (mode 'mean') for the resulting instance.
 
        If arg between is 'groups', statistical difference is calculated 
        between groups within a Groups object (item of arg list). In this case
        arg references can be one of the following:
          - single group name (string), so all groups of a Groups object have
          the same reference. Here the specified group name has to appear in
          all Groups objects.
          - dictonary where list item names are keys and reference group names
          are values. Again, all groups of a Groups object have the same
          reference, but Groups objects can have different group names.
          - dictionary where list item names are keys and values are again 
          dictionaries with all group names (belonging to the corresponding
          Groups objects) are keys and the corresponding reference group names
          are values. Here, each group can have a different reference.

        Alternatively, if arg between is 'list_items', the difference is 
        calculated between groups of different Groups objects that have the 
        same group names. In this case each Group object has to have the same 
        group names and arg reference can be one of the following:
           - single list item name (string), so all groups with the same group
           name have the same reference and all references belong to the same 
           Groups object
           - dictonary where igroup names are keys and list item names are 
          values. Again, all groups with same group name have the same
          reference, but reference groups do not have to belong to the
          same Groups object
          - dictionary where group names are keys and values are again
          dictionarys where all list item names are keys and the corresponding 
          refernce list item names are values.

        Returns a new instance of this class where individual groups correspond
        to Groups objects of arg list, and have group names from arg listNames.
        Identifiers of these Groups objects are the same as groups names of 
        arg list elements. Indifidual groups of the resulting Groups object
        (Observations instances) have following attributes (if test is None 
        inference is not calculated):
          - mean
          - std: n-ddof degrees of freedom
          - n: number of individual data points
          - sem: standard error of means
          - testValue: values of t, h or u depending on the test
          - testSymbol: 't', 'h', or 'u', depending on the test
          - confidence: probability that data from a group and the/its 
          reference come from the same population

        Arguments:
          - list: list of Groups objects
          - listNames: (list of strings) list item names 
          - name: name of the data attribute
          - groups: list of group names to be analyzed, None for all groups 
          - mode: 'join' to join or 'mean' to average data
          - test: statistical test used: 't', 'h', 'u', or any other stated
          in Observations.doInference() doc. If None no test is done. 
          - between: determines if statistical significance is calculated 
          between groups of the same Groups object (value 'groups') or between 
          groups having the same group name
          - reference: name of the reference group(s) (if arg between 
          is 'experiments'), or reference identifier(s) (arg between is
          'groups')
          - ddof: delta degrees of freedom, denominator for std is N-ddof
        """

        # join experiments to obtain single Groups object
        new_groups = cls.joinExperimentsList(list=list, listNames=listNames, 
                                             name=name, mode=mode)
        
        # adjust between to be correct for the joined data 
        if between == 'list_items':
            new_bet = 'groups'
        elif between == 'groups':
            new_bet = 'experiments'
        else:
            raise ValueError(
                "Argument between: " + between + " not understood. "
                + "Allowed values are 'list_items' and 'groups'.")

        # do stats and inference on the new Groups
        stats = new_groups.doStats(
            name=name, test=test, identifiers=groups, between=new_bet, 
            reference=reference, ddof=ddof, out=None)

        return stats

    def doStats(self, name, bins=None, fraction=1, 
                groups=None, identifiers=None, 
                test=None, between='experiments', reference=None, ddof=1, 
                out=sys.stdout, outNames=None, format_=None, title=None):
        """
        Does statistical analysis on data (specified by arg name) for each 
        experiment separately, and tests for statistical difference between
        the data and specified reference experiment(s) data.

        If arg bin is None, calculates basic statistics. The calculations are
        saved as following attributes:
          - data: deepcopied data 
          - mean: mean
          - std: n-ddof degrees of freedom,
          - n: number of individual data points
          - sem: standard error of means
          - ddof: delta degrees of freedom, denominator for std is N-ddof

        Alternatively, if arg bins is given calculates histogram of the data 
        according to bins ans saves it as the following attributes:
          - ids: left bin limits 
          - data: deepcopied data 
          - histogram: histogram (indexed)
          - probability: histogram / sum_of_histogram (indexed)
          - n: number of individual data points
          - fraction: histogram[fraction] / n
 
          If areg test is specified, inference is calculated and the results
          are saved as the follwoing attributes:
          - testValue: values of t, chi2, h or u depending on the test
          - testSymbol: 't', 'chi2', 'h', or 'u', depending on the test
          - confidence: probability that data from a group and the/its 
          reference come from the same population

        If arg between is 'experiments', statistical difference is calculated 
        between experiments within each group. In this case arg references
        can be one of the following:
          - single identifier (string), so all experiments of a group have the
          same reference. Here the specified identifier has to appear in all
          groups
          - dictonary where group names are keys and reference identifiers are 
          values. Again, all experiments of a group have the same reference, 
          but groups can have different identifiers.
          - dictionary where group names are keys and values are again 
          dictionaries with all identifiers (belonging to the corresponding
          group) are keys and the corresponding reference identifiers are 
          values. Here, each experiment can have a different reference.

        Alternatively, if arg between is 'groups', the difference is calculated
        between experiments of different groups that have the same identifiers.
        In this case each group has to have the same experiment identifiers
        and arg reference can be one of the following:
           - single group name (string), so all experiments with same 
           identifier have the same reference and all reference experiments
           belong to the same group
           - dictonary where identifiers are keys and group names are 
          values. Again, all experiments with same identifier have the same
          reference, but reference experiments do not have to belong to the
          same group
          - dictionary where identifiers are keys and values are again
          dictionarys where all group names are keys and the corresponding 
          refernce group names are values.

        Returns a new instance of this class that has the same groups and
        identifiers as this instance. Each group (Observations instance) has 
        the attributes listed above.

        Arguments:
          - name: name of the data attribute
          - bins: histogram bins, if specified histogram is calculated
          - fraction: (int) position of the histogram bin for which the
          - groups: list of group names, None for all groups 
          - identifiers: list of experiment identifiers, None for all 
          experiments of all groups.
          - between: determines if statistical significance is calculated 
          between experiments of the same group (value 'experiment') or between 
          experiments having the same identifyer (value 'groups')
          - reference: name of the reference group(s) (if arg between 
          is 'experiments'), or reference identifier(s) (arg between is
          'groups')
          - test: statistical test used: 't', 'h', 'u', or any other stated
          in Observations.doInference() doc. If None no test is done. 
          - ddof: delta degrees of freedom, denominator for std is N-ddof
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name
          - outNames: list of sttribute names to be printed, None for a default
          names list: ['mean', 'std', 'sem', 'n', 'testValue', 'confidence']
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first
        """

        # stats instance
        stats_groups = self.__class__()        

        if between == 'experiments':

            for g_name, group in self.items():

                # skip non-specified groups
                if (groups is not None) and (g_name not in groups):
                    continue

                # do stats and inference
                stats = group.doStats(
                    name=name, bins=bins, fraction=fraction,
                    new=True, identifiers=identifiers, ddof=ddof)

                # not good
                # add fraction-like data properties for each observation
                # separately
                #if bins is not None:
                #    stats_data = group.doStats(
                #        name=name, bins=bins, fraction=fraction, 
                #        identifiers=identifiers, new=True, ddof=ddof)
                #    for (not_data, yes_data) in self._fract_data_names.items():
                #        stats.setValue(
                #            identifier=g_name, name=yes_data,
                #            value=getattr(stats_data, not_data), indexed=False)

                # inference
                if test is not None:

                    # make reference and do inference
                    if isinstance(reference, str):
                        obs_ref = reference
                    elif isinstance(reference, dict):
                        obs_ref = reference[g_name]
                    else:
                        raise ValueError("Argument reference has to be "
                                         + "string or dictionary.")
                    if bins is None:
                        infer_name = 'data'
                    else:
                        infer_name = 'histogram'
                    stats.doInference(identifiers=identifiers, test=test, 
                                      name=infer_name, reference=obs_ref)
                    stats.setValue(property='referenceGroup', identifier=None,
                                   default=g_name)

                # add stats to stats_groups
                stats_groups[g_name] = stats

            # print stats
            stats_groups.printStats(
                out=out, names=outNames, groups=groups,
                identifiers=identifiers, format_=format_, title=title)

        elif between == 'groups':

            transp = self.transpose()
            t_stats = transp.doStats(
                name=name, bins=bins, fraction=fraction, test=test, 
                groups=identifiers, identifiers=groups, 
                between='experiments', reference=reference, ddof=ddof, out=out)
            stats_groups = t_stats.transpose()

        else:
            raise ValueError(
                "Argument between: " + between + " not understood. "
                + "Possible values are 'experiments' and 'groups'.")

        return stats_groups

    def countHistogram(
        self, name='ids', groups=None, identifiers=None, 
        test=None, reference=None, 
        out=sys.stdout, outNames=None, format_=None, title=None):
        """
        Analysis of histograms, where each histogram is obtained by counting 
        number of data points in all experiments that have the same identifier.

        The resulting object (of this class) has the same group structure as 
        this instance, and each group (Observation object) has the following
        data attributes:
          - count: number of data points
          - n: total number of data points for all experiments with the same
          identifier
          - fraction: count / n
 
        If areg test is specified, inference is calculated and the results
        are saved as the follwoing attributes (see Observations.doInference()):
          - reference: reference
          - testValue: values of t, chi2, h or u depending on the test
          - testSymbol: 't', 'chi2', 'h', or 'u', depending on the test
          - confidence: probability that data from a group and the/its 
          reference come from the same population
         
        For example, for an object where:

          group a, identifier 'i1', data = [1, 2, 3]
          group a, identifier 'i3', data = [1, 2, 3, 4]
          group b, identifier 'i1', data = [5]
          group b, identifier 'i3', data = [6, 7]
           
        the resulting object has:

          group a, identifier 'i1': count=3, n=4, fraction=3/4 
          group a, identifier 'i3': count=4, n=6, fraction=4/6 
          group b, identifier 'i1': count=1, n=4, fraction=1/4 
          group a, identifier 'i3': count=2, n=6, fraction=2/6 

        For significance tests histograms [3, 1] (i1) and [4, 2] (i3) are 
        compared.

        This method can be applied only on symmetrical objects.

        Also prints the calculated data.

        Arguments:
          - name: name of the data attribute (this data is counted)
          - groups: list of group names, None for all groups 
          - identifiers: list of experiment identifiers, None for all 
          experiments of all groups.
           - reference: name of the reference group(s) (if arg between 
          is 'experiments'), or reference identifier(s) (arg between is
          'groups')
          - test: statistical test used: 't', 'h', 'u', or any other stated
          in Observations.doInference() doc. If None no test is done. 
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name
          - outNames: list of sttribute names to be printed, None for a default
          names list: ['mean', 'std', 'sem', 'n', 'testValue', 'confidence']
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first
        """

        # make instance to hold statistics
        stats = self.__class__()
        total_n = {}
        
        # set data and count data point for each identifier (across groups)  
        for g_name, group in self.items():

            # skip non-specified groups
            if (groups is not None) and (g_name not in groups):
                continue

            # make new Observations object
            stats[g_name] = Observations()

            # data points for this group
            for ident in group.identifiers:

                # skip non-specified identifiers
                if (identifiers is not None) and (ident not in identifiers):
                    continue

                values = group.getValue(property=name, identifier=ident)
                data = len(values)
                stats[g_name].setValue(property='count', identifier=ident, 
                                       value=data)
                total_n[ident] = total_n.get(ident, 0) + data

        # set fraction and  number of data points for each identifier (across 
        # groups)  
        for g_name, group in stats.items():

            # skip non-specified groups
            if (groups is not None) and (g_name not in groups):
                continue

            # save number of data points
            for ident in group.identifiers:

                # skip non-specified identifiers
                if (identifiers is not None) and (ident not in identifiers):
                    continue
   
                # calculate fraction
                stats[g_name].setValue(property='n', 
                                       identifier=ident, value=total_n[ident])
                data = stats[g_name].getValue(property='count', 
                                              identifier=ident)
                fraction = data / float(total_n[ident])
                stats[g_name].setValue(property='fraction', identifier=ident, 
                                       value=fraction)
 
        # do inference
        if test is not None:

            # prepare data
            transposed = stats.transpose()
            obs = transposed.joinExperiments(name='count', mode='join')

            # find reference
            #if isinstance(reference, str):
            #    obs_ref = reference
            #elif isinstance(reference, dict):
            #    obs_ref = reference[g_name]
            #else:
            #    raise ValueError("Argument reference has to be "
            #                     + "string or dictionary.")

            # inference
            obs.doInference(name='count', identifiers=identifiers, test=test, 
                            reference=reference)
            obs.setValue(property='referenceGroup', identifier=None,
                         default=g_name)

        # copy inference values to stats object
        for g_name, group in stats.items():

            # skip non-specified groups
            if (groups is not None) and (g_name not in groups):
                continue

            # copy
            for ident in group.identifiers:

                # skip non-specified identifiers
                if (identifiers is not None) and (ident not in identifiers):
                    continue
   
                for name in ['confidence', 'reference', 'testValue', 
                             'testSymbol']:
                    value = obs.getValue(property=name, identifier=ident)
                    stats[g_name].setValue(property=name, identifier=ident,
                                           value=value)

        # print stats
        stats.printStats(
            out=out, names=outNames, groups=groups,
            identifiers=identifiers, format_=format_, title=title)

        return stats

    def doCorrelation(
        self, xName, yName, test=None, regress=False, reference=None, 
        mode=None, groups=None, identifiers=None, out=sys.stdout, 
        format_={}, title=''):
        """
        Tests if data specified by args xName and yName are correlated.

        If mode is None, data from each experiment is analyzed separately. 
        The returned object is an instance of this class and has the same
        group / experiment structure as this instance.

        Alternatively, if mode is 'join', data from all experiments of one group
        are taken together. In this case arg new is ignored (effectively set to
        True). In this case the returned object is an instance of Observations
        and its identifiers correspond to the group names of this object.

        If specified, arg identifiers determines the order of identifiers 
        in the resulting instance. If None, all existing identifiers are used.

        The results are saved as the following attributes:
          - xData, yData: deepcopied xName, yName data, only if new is True
          - n: number of individual data points
          - testValue: correlation test value
          - testSymbol: currently 'r' or 'tau', depending on the test
          - confidence: confidence
          - aRegress, bRegress: slope and intercept of the regression line 
          (if arg regress is True)

        Arguments:
          - xName, yName, property names of data 
          - test: correlation test, currently 'r' (or 'pearson'), 'tau' 
          (or 'kendall')
          - regress: flag indicating if regression (best fit) line is calculated
          - reference:
          - mode: None, or 'join'
          - groups: list of group names to use
          - identifiers: list of experiment identifiers for which the 
          correlation is calculated. Identifiers listed here that do not 
          exist among identifiers of this instance are ignored.  
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name
          - outNames: list of sttribute names to be printed, None for a default
          names list: ['mean', 'std', 'sem', 'n', 'testValue', 'confidence']
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first

        Returns object containing data and results.
        """

        # set groups
        if groups is None:
            group_names = self.keys()
        else:
            group_names = groups

        # calculate correlation 
        if mode is None:

            # make a new group
            inst = self.__class__()

            for group_nam in group_names:

                # discard non-existing groups
                if group_nam not in self.keys():
                    continue

                # do correlation
                corr = self[group_nam].doCorrelation(
                    xName=xName, yName=yName, test=test, regress=regress, 
                    reference=reference, mode=None, new=True, 
                    identifiers=identifiers, out=None)

                # add to the new object
                inst[group_nam] = corr

        elif mode == 'join':

            # make new Observations
            inst = Observations()

            for group_nam in group_names:
                
                # discard non-existing groups
                if group_nam not in self.keys():
                    continue

                # do correlation
                corr = self[group_nam].doCorrelation(
                    xName=xName, yName=yName, test=test, regress=regress, 
                    reference=reference, mode='join', new=True, 
                    identifiers=identifiers, out=None)

                # add to the new object
                inst.addExperiment(experiment=corr, identifier=group_nam)

        else:

            raise ValueError(
                "Argument mode: " + mode + " not understood. "
                + "Allowed values are None and 'join'.")

        # print 
        names = ['n']
        if test is not None:
            names.extend(['testValue', 'confidence'])
        if regress:
            names.extend(['aRegress', 'bRegress'])
        inst.printData(names=names, out=out, format_=format_, title=title)

        # return
        return inst

    def printStats(self, out=sys.stdout, names=None, identifiers=None, 
                   groups=None, format_={}, title=None,
                   other=None, otherNames=[]):

        """
        Prints basic statistics and inference analysis results contained
        in this instance.

        Data from another Groups object having the same group names and
        experiment identifiers can be printed along the stats of this
        object if the another group is specifed as arg other and the
        proeprty names of the other group are given in arg otherNames.

        Arguments:
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name
          - names: list of sttribute names to be printed, None for a default
          names list: ['mean', 'std', 'sem', 'n', 'fraction', 'testValue', 
          'confidence']
          - groups: list of group names, None for all groups 
          - identifiers: list of experiment identifiers, None for all 
          experiments of all groups.
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first
          - other: (Group) another object of the same group and identifier
          structure as this object
          - otherNames: list of property names of the other group
        """

        # names
        default_names = ['mean', 'std', 'sem', 'fraction', 'count', 'n', 
                         'testValue', 'confidence'] 
        if names is None:
            names = default_names

        # format
        loc_format = {
            'mean' : '    %5.2f ',
            'std' : '    %5.2f ',
            'sem' : '    %5.2f ',
            'fraction' : '    %5.3f ',
            'count' : '    %5d ',
            'n' : '    %5d ',
            'testValue' : '    %5.2f ',
            'confidence' : '   %7.4f'
            }
        if format_ is not None:
            loc_format.update(format_)

        # print
        self.printData(
            names=names, out=out, groups=groups, identifiers=identifiers, 
            format_=loc_format, title=title, other=other, otherNames=otherNames)

    def printData(self, names, out=sys.stdout, groups=None, identifiers=None, 
                  format_={}, title=None, other=None, otherNames=[]):
        """
        Prints data (arg names) of this instance.

        Data from another Groups object having the same group names and
        experiment identifiers can be printed along the data of this
        object if the another group is specifed as arg other and the
        proeprty names of the other group are given in arg otherNames.

        Arguments:
          - names: list of property names to be printed
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name
          - groups: list of group names, None for all groups 
          - identifiers: list of experiment identifiers, None for all 
          experiments of all groups.
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first
          - other: (Group) another object of the same group and identifier
          structure as this object
          - otherNames: list of property names of the other group
        """

        # default data format
        default_format = '    %5.2f '

        # set output 
        if out is None:
            return
        elif isinstance(out, str):
            out = open(out)

        # print title
        if title is not None:
            out.write(os.linesep + title + os.linesep)

        # set groups
        if groups is None:
            group_names = self.keys()
        else:
            group_names = groups

        # make group name format based on the longest one
        group_length = max([len(x) for x in group_names])
        group_length = max(group_length, len('Group'))
        group_format = '%' + str(group_length) + 's '

        # set all identifiers (used only to replace testValues)
        if identifiers is None:
            all_identifiers = []
            for gr_name in group_names:
                all_identifiers.extend(self[gr_name].identifiers)
        else:
            all_identifiers = identifiers

        # make identifier format based on the longest identifier
        ident_length = max([len(x) for x in all_identifiers])
        ident_length = max(ident_length, len('Identifier'))
        ident_format = '%' + str(ident_length) + 's '

        # pick one group to get properties
        for gr_name in group_names:
            one_group = self[gr_name]
            if len(one_group.identifiers) == 0:
                continue
            if other is not None:
                one_other_group = other[gr_name]
            break

        # set variable (attribute) names 
        names = [name for name in names if name in one_group.properties]
        if other is not None:
            other_names = [name for name in otherNames \
                               if name in one_other_group.properties]
        else:
            other_names = []
        if (other_names is None) or (len(other_names) == 0):
            other = None

        # replace 'testValues' by actual testSymbol in names
        head_names = []
        for nam in names:
            if (nam == 'testValue') and (len(all_identifiers) > 0):
                head_names.append(one_group.testSymbol[0])
            else:
                head_names.append(nam)
            
        # append head names from the other stats
        if other is not None:
            for nam in otherNames:
                if (nam == 'testValue'):
                    head_names.append(one_other_group.testSymbol[0])
                else:
                    head_names.append(nam)
                
        # print column (data) names
        var_head = (group_format + ident_format) % ('Group', 'Identifier')
        var_head = var_head + (('%9s ' * len(head_names)) % tuple(head_names))
        out.write(var_head + os.linesep)

        # make one row format string
        var_format = ''.join([format_.get(name, default_format) 
                              for name in names])
        if other is not None:
            other_var_format = ''.join([format_.get(name, default_format) 
                                        for name in other_names])
        else:
            other_var_format = ''
        row_format = group_format + ident_format + var_format \
            + other_var_format + os.linesep

        # print rows
        for gr_name in group_names:
            group = self[gr_name]

            # get identifiers
            curr_identifiers = group.identifiers
            if identifiers is not None:
                curr_identifiers = [ident for ident in identifiers 
                                    if ident in curr_identifiers]

            # loop over experiments in order
            for identif in curr_identifiers:

                # keep only experiments of the current group
                if identif not in group.identifiers:
                    continue

                # set values
                row_values = [
                    group.getValue(identifier=identif, property=nam) 
                    for nam in names]
                if other is not None:
                    other_row_values = [
                        other[gr_name].getValue(identifier=identif, 
                                                 property=nam) 
                        for nam in other_names]
                    row_values.extend(other_row_values)
                row_values = [gr_name, identif] + row_values

                # print
                out.write(row_format % tuple(row_values))


    ###############################################################
    #
    # Methods to calculate other properties
    #
    ##############################################################

    def getNPositive(self, name, n_name, categories=None):
        """
        Calculates number of positive elements of a given property for each
        observation (experiment) and saves the results as a new (non-indexed)
        property.

        Arguments
          - name: name of the property whose positive elements are counted
          - n_name: name of the property where the results are stored
          - categoriesL
          - categories: list of group names, in None all groups are used
        """
 
        if categories is None:
            categories = self.keys()

        # loop over groups and observations
        for categ in categories:
            for ident in self[categ].identifiers:

                values = self[categ].getValue(property=name, identifier=ident)
                n_pos = (values > 0).sum()
                self[categ].setValue(property=n_name, identifier=ident,
                                    value=n_pos, indexed=False)

    def getN(
        self, name, categories=None, inverse=False, fixed=None, layer=None, 
        layer_name='surface_nm', layer_index=1, layer_factor=1.e-6):
        """
        Counts number of elements of this instance, or calculates a related 
        property (depending on the arguments) for each observation and saves it
        as a separate (non-indexed) property named by arg name.

        The elements that are counted are the main elements of this class 
        (those specified by ids). For example they are vesicles if this class 
        is Vesicles, tethers or connectors if this class is Connections.

        If arg layers in None, only the number of elements is calculated.

        If arg layers is specified, number of elements per unit layer area
        is calculated. The unit layer area is calculated as:

          layer_name[layer_index] * layer_factor

        In a typical case layer area in nm (property 'surface_nm') of the
        layer 1 (layer_index=1). If layer_factor=1.e-6 (default), then this 
        method calculates number of elements per 1 um^2 of the layer surface.

        If arg inv is True, the inverse is calculated, that is the layer area 
        per one sv.

        If arg fixed is specified, te calculation proceed as above except that
        the number of elements (for each observation) is fixed to this number.
        For example, if fixed=1 and inverse=True, the area of the layer
        specifed by arg layer_index is calculated.

        Sets property named (arg) name.

        Arguments:
          - name: name of the newly calculated property
          - inverse: flag indicating if inverse of the property should be 
          calculated
          - layer: object containing layer info
          - layer_name: name of the layer property that contains the desired 
          layer data
          - layer_index: index of the layer whose surface is used
          - layer_factor: the surface is multiplied by this factor
        """

        # set categories
        if categories is None:
            categories = self.keys()

        # calculate and set as a new property
        for categ in categories:
            for ident in self[categ].identifiers:

                # get n 
                if fixed is None:
                    ids = self[categ].getValue(identifier=ident, property='ids')
                    n_sv = len(ids)
                else:
                    n_sv = fixed

                # get layer surface
                if layer is not None:
                    surface = layer[categ].getValue(
                        identifier=ident, property=layer_name, ids=layer_index)

                    # fix in case getValue returns an array
                    if (isinstance(surface, (list, numpy.ndarray)) 
                        and len(surface) == 1):
                        surface = surface[0]

                    surface = surface * layer_factor
                else:
                    surface = 1

                # get and set ratio
                if not inverse:
                    value = float(n_sv) / surface
                else:
                    value = surface / float(n_sv)
                self[categ].setValue(property=name, identifier=ident, 
                                         value=value, indexed=False)



    #######################################################
    #
    # Other methods 
    #
    #######################################################

    def apply(self, funct, args, kwargs={}, name=None, categories=None, 
              indexed=True):
        """
        Applies (arg) funct to indexed properties args where other arguments 
        are given in arg kwargs. 

        In other words, for each category, observation and for all observation 
        elements funct is applied to the corresponding values of the properties.

        Arguments to funct can be:
          - property names, specified in args (positional)
          - other values, specified in kwargs (keyword)

        Returns the results as a dictionary (representing groups) of lists 
        (observations) of ndarrays (observation elements) if name is None. 
        Otherwise a new indexed property is created to hold the result. This 
        new name can be the same as an already existing property, in which 
        case the original values are owerwriten. The name of the new property 
        is added to properties and indexed attributes.

        For example:

          def add(x, y): return x + y
          groups.apply(funct=add, args=['vector'], kwargs={'y', 5)

        will return property vector of instance groups increased by 5, while

          def add(x, y): return x + y
          groups.apply(
              funct=add, args=['vector'], kwargs={'y', 5}, name='new_vector')

        will save the new values as property new_vector.

        Arguments:
          - function: function
          - args: list of indexed property names that are used as
          arguments to funct
          - kwargs: (dict) other arguments passed to funct
          - name: name of the new property
          - categories: categories, all categoreis if None
          - indexed: specifies if the new property is indexed

        Sets indexed property name, or returns the result (dictionary of lists 
        of ndarrays).
        """

        # get categories
        if categories is None:
            categories = self.keys()

        # apply to each category
        for categ in categories:
            res_categ = self[categ].apply(
                funct=funct, args=args, kwargs=kwargs, name=name, 
                indexed=indexed)
            if name is None:
                try:
                    res[categ] = res_categ
                except NameError:
                    res = {categ : res_categ}

        # return
        if name is None:
            return res

    def max(self, name, categories=None):
        """
        Returns max value of the property given by arg name from all
        observations and all specified categories.

        Nan values are ignored. However, if nan is the only value, or if there 
        are no values numpy.nan is returned. This is consistent with 
        numpy.nanmin(). 

        Arguments:
          - name: property name
          - categories: list of categories

        Returns max value
        """

        # get categories
        if categories is None:
            categories = self.keys()

        # put all variables together
        all_ = [getattr(self[categ], name) for categ in categories]
        all_ = pyto.util.nested.flatten(all_)

        if len(all_) > 0:
            res = numpy.nanmax(all_)
        else:
            res = numpy.nan

        return res

    def min(self, name, categories=None):
        """
        Returns min value of the property given by arg name from all
        observations and all specified categories.

        Nan values are ignored. However, if nan is the only value, or if there 
        are no values numpy.nan is returned. This is consistent with 
        numpy.nanmin(). 

        Arguments:
          - name: property name
          - categories: list of categories

        Returns min value
        """

        # get categories
        if categories is None:
            categories = self.keys()

        # put all variables together
        all_ = [getattr(self[categ], name) for categ in categories]
        all_ = pyto.util.nested.flatten(all_)

        if len(all_) > 0:
            res = numpy.nanmin(all_)
        else:
            res = numpy.nan

        return res
