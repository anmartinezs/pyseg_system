"""
Defines class Observations that can hold data form one or more observations 
(experiments). 

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
#from past.utils import old_div
from past.builtins import basestring

__version__ = "$Revision$"


import sys
import os
import warnings
import logging
from copy import copy, deepcopy

import numpy
import scipy
try:
    import pandas as pd
except ImportError:
    pass # Python 2
import pyto
from ..util import nested
from .experiment import Experiment


class Observations(object):
    """
    Contains data for (a particular aspect of) one or more experiments and 
    methods useful in the analysis of this data. 

    The data for each experiment consist of one or more properties. Typically,
    some of these properties are vectorial, that is there as an array of values
    for each experiment, and thery are indexed by an index. The index itself is
    then one of the vectorial (here called undexed) properties. In addition,
    typically there are also scalar properties (one value for an experiment),
    here called non-indexed.

    The data is organized in the following way. Each data-related attribute 
    (or property) is a list where each element contains value(s) of that
    property pertaining to one observation. All of these attributes
    have to have the same order (as identifiers). 

    Furthermore, some of these attributes are 'indexed' (see attribute 
    indexed below), that is each element is another list or array. Elements 
    of these sublists (subarrays) are ordered in the same way as the elements
    of index attribute.

    Meta-data attributes (obligatory):
      - properties: set of properties (names), indexed and non-indexed
      - indexed: set of indexed properties (names)

    Observation related, non-indexed attributes:
      - identifiers: (list of strings) experiment identifiers, obligatory 
      - categories, treatments, features or related: (list of strings) 
      describing experiments, at least one has to exist

    Other attributes
      - index: (string) name of the index attributes (default 'ids')
      - ids (or other, see above): 
    """

    #######################################################
    #
    # Initialization
    #
    #######################################################

    def __init__(self):
        """
        Initialize attributes
        """
        self.categories = []
        self.identifiers = []
        self.index = 'ids'
        self.properties = set(['identifiers'])
        self.indexed = set([])

    def initializeProperties(self, names):
        """
        Adds elements of arg names to properties and initialize each property
        to []. 

        Can be used only on instances that have no identifiers.
        """

        # check
        if len(self.identifiers) > 0:
            raise ValueError("Method initializeProperties() can be invoked "
                             + " only on objects that have no identifiers.")

        # initialize
        if isinstance(names, basestring):
            names = [names]
        for nam in names:
            self.properties.add(nam)
            setattr(self, nam, [])

    def recast(self):
        """
        Returns a new instance of this class and sets all attributes of the new 
        object to the attributes of obj.

        Useful for objects of this class loaded from a pickle that were pickled
        before some of the current methods of this class were coded. In such a
        case the returned object would contain the same attributes as the
        unpickled object but it would accept recently coded methods. 
        """

        # make a new instance
        new = self.__class__()

        # set properties
        for name in self.properties:
            value = getattr(self, name)
            setattr(new, name, value)

        # set book-keeping attributes
        new.index = self.index
        new.indexed = self.indexed
        new.properties = self.properties

        return new


    ##########################################################
    #
    # Set/get methods
    #
    ##########################################################

    def addCatalog(self, catalog, default=None):
        """
        Adds data from a specified catalog to this instance. 
        
        If an experiment does not have a value for a preference that exists 
        for other experiment(s) the default value is used. However, if an
        identifier from catalog does not exist in the current instance, 
        values for that experiment are not added to the current instance.  

        Adds added property names to self.properties, but does not change 
        self.indexed.

        Arguments:
          - catalog: (Catalog) catalog 
          - default: default value
        """
        if catalog is None: return

        for property, values in list(catalog._db.items()):
            for identifier in self.identifiers:
                one_value = catalog._db[property].get(identifier, default)
                self.setValue(identifier=identifier, property=property,
                              value=one_value, default=default)
                self.properties.add(property)

    def getExperiment(self, identifier):
        """
        Extracts values from this instance that correspond to a single
        experiment, specified by its identifier. The extracted values are put
        into an Experiment object and returned

        In adition to data attributes (properties) the following attributes 
        are set in the resulting Experiment instance:
          - properties: copied form self.properties
          - indexed: copied form self.indexed 
          - identifier: identifier
        Attribute identifiers is not present in the resulting object

        Returns None if arg identifier is not in the list of the current
        identifiers (self.identifiers)

        Argument:
          - identifier: experiment identifier
        """

        # get index corresponding to the identifier
        index = self.getExperimentIndex(identifier=identifier)

        # instantiate Experiment object
        exp = Experiment()
        exp.indexed = copy(self.indexed)
        exp.properties = copy(self.properties) - set(['identifiers']) 
        exp.properties.add('identifier')
        exp.identifier = identifier

        # extract properties but not 'identifiers'
        for name in self.properties:
            if name != 'identifiers':
                value = getattr(self, name)[index]
                setattr(exp, name, value)

        return exp

    def getExperimentIndex(self, identifier):
        """
        Returns index correponding to the specifed experiment, or None if
        the identifier doesn't exist

        Argument:
          - identifier: experiment identifier
        """

        # get index corresponding to the identifier
        try:
            index = self.identifiers.index(identifier)
        except ValueError:
            return None

        # test
        if self.identifiers.count(identifier) > 1:
            raise ValueError("Identifier " + identifier + " occurs at more "
                             + " than one position in the identifiers list.")

        return index

    def addExperiment(self, experiment, identifier=None):
        """
        Adds experiment (deepcopied) to this instance.

        If the current object is empty (no items in self.identifiers list)
        properties and indexed attributes are copied from experiment, with
        the exception that 'identifier' is relaced by 'identifiers'.

        Otherwise, this instance and experiment have to have the same values
        for attributes properties and indexed. Also, experiment.identifier 
        should not be in self.identifiers before this method is called. This
        method appends experiment.identifier to self.identifiers. 

        Argument:
          - expreriment: instance of Experiment
          - identifier: identifier for the added eperiment, if None the value
          of experiment.identifier is used
        """
        
        # add properties and indexed if this object is empty
        if len(self.identifiers) == 0:
            properties = copy(experiment.properties)
            properties.difference_update(set(['identifier']))
            properties.update(set(['identifiers']))
            self.properties = properties
            self.indexed = copy(experiment.indexed)

        # set new identifier
        if identifier is None:
            identifier = experiment.identifier

        # check if new identifier
        if identifier in self.identifiers:
            raise ValueError("Can't add Experiment to this Observations "
                             + "because identifier " + identifier
                             + " already exists.")

        # check if properties and indexed the same
        if (((experiment.properties - set(['identifier'])) 
            != (self.properties - set(['identifiers']))) 
            or (experiment.indexed != self.indexed)):
            raise ValueError("Can't add Experiment to this Observations "
                             +"because properties or indexed are not the same.")

        # add
        for name in self.properties:
            if name == 'identifiers':
                self.identifiers.append(identifier)
            else:
                new_value = deepcopy(getattr(experiment, name))
                try:
                    old_values = getattr(self, name)
                except AttributeError:
                    old_values = []
                old_values.append(new_value)
                setattr(self, name, old_values)
        
    def getValue(self, identifier, property=None, name=None, ids=None):
        """
        Returns value of the specified property for the specified experiment.

        If arg ids is specified, returns the values corresponding to the
        specified ids. If ids is a list or ndarray returns an ndarray of values.
        Alternatively, if ids is a single int, returns only the corresponding
        value.

        Raises ValueError if specifed identifier doesn't exist.

        Arguments:
          - identifier: experiment identifyer
          - name: (string) property name, same as property 
          - property: (string) property name, same as name, kept for 
          backcompatibility
          - ids: (list, ndarray or int) one or more ids
        """

        # check arguments name and property
        if (name is not None) and (property is None):
            property = name
        elif (name is None) and (property is None):
            raise ValueError(
                "Either argument 'name' or 'property' have to be specified.")
        elif (name is not None) and (property is not None):
            raise ValueError(
                "Only one of the arguments 'name' or 'property' can be "
                + "specified.")
            
        # get observation values
        exp_index = self.getExperimentIndex(identifier=identifier)
        if exp_index is None:
            raise ValueError("Experiment identifier " + str(identifier) + 
                             " doesn't exist.") 
        value = getattr(self, property)[exp_index]

        # get values corresponding to ids
        if ids is not None:
            all_ids = self.ids[exp_index]
            if isinstance(ids, (list, numpy.ndarray)):
                sorted_ids = numpy.sort(all_ids)
                arg_ids = all_ids.argsort()
                id_pos = arg_ids[sorted_ids.searchsorted(ids)]
                result = value[id_pos]
            else:
                result = value[all_ids==ids][0]
        else:
            result = value

        return result

    def setValue(self, property=None, name=None, identifier=None, value=None, 
                 default=None, indexed=False, id_=None):
        """
        Sets the specified property for the specified experiment (identifier) 
        to the given value. 

        The value can correspond to one or more elements specified by id 
        (arg id_) or it can be an array that contain values for all ids of the
        specifed observation (experiment) (arg id_ is None). In case arg id_
        contains more than one element, arg value has to have the same number
        of elements.

        In case the property doesn't exist, a new one is created. The values
        of this property for other experiments is set to the default value.

        In case the experiment doesn't exist, a new entry is made for that
        experiment. The value of the specified property is set to the given
        value, while the values of all other properties corresponding to 
        the new identifier are set to the default value.

        However, if the experiment does not exist and arg id_ is not None
        ValueError is raised.

        If arg identifier is None, all existing experiments get the default
        value for the specified property. Arg value is in this case ignored.

        Adds the property (name) to self.properties. If arg indexed is True,
        the property (name) is also added to self.indexed.

        Uncommon use: If the property to be set for the specified identifier
        does not exist (i.e. the data structures are inconsistent) the 
        specified value is still set correctly. This can happen within
        pyto.io.MultiData.readPropertiesGen() loop for example, as in
        CleftLayers.read() (lines 154-156 in rev 697). 

        Arguments:
          - identifier: experiment identifier
          - name: (string) property name, same as property 
          - property: (string) property name, same as name, kept for 
          backcompatibility
          - value: single value or a ndarray of values for all ids
          - default: default value
          - indexed: flag indicating if the property is indexed
          - id_: (single number, list, tuple or numpy.ndarray) id of one 
          or more elements that needs to be set, or None for all ids
        """

        # check arguments name and property
        if (name is not None) and (property is None):
            property = name
        elif (name is None) and (property is None):
            raise ValueError(
                "Either argument 'name' or 'property' have to be specified.")
        elif (name is not None) and (property is not None):
            raise ValueError(
                "Only one of the arguments 'name' or 'property' can be "
                + "specified.")

        # if specific element, make sure indexed is True
        if (id_ is not None) and (not indexed):
            indexed = True
            
        # check if new property and new experiment
        if getattr(self, property, None) is None:
            new_property = True
        else:
            new_property = False
        if identifier in self.identifiers:
            new_experiment = False
        else:
            new_experiment = True

        # check if property ids exist
        try:
            self.ids
                #if not new_experiment:
            ids_exist = True
                #else:
                #    ids_exist = False
        except AttributeError:
            ids_exist = False

        # if new experiment can't set specific element of an indexed property
        if new_experiment and (id_ is not None):
            raise ValueError(
                "Can't set specific element of an indexed property"
                + " for a non-existing experiment.")

        # get current values of the property, or make defaults
        if not new_property:

            # property exists
            exp_values = getattr(self, property)
 
        else:
            
            # new property, set to defaults
            exp_values = [copy(default) for ind 
                          in range(len(self.identifiers))]

            # set all elements if indexed property
            if indexed and ids_exist:
                for exp_index in range(len(self.identifiers)):
                    
                    if self.ids[exp_index] is None:

                        # raise error if setting individual indexed element 
                        if id_ is not None:
                            raise ValueError(
                                "Can't set value for a single indexed property"
                                + " for a non-existing experiment.")

                        # set value
                        exp_values[exp_index] = value

                    else:
                        
                        # set value only if current ids is an array 
                        if isinstance(self.ids[exp_index], 
                                      (list, numpy.ndarray)):
                            exp_values[exp_index] = default
                            n_ids = len(self.ids[exp_index])
                            exp_list = [copy(default) for ind in range(n_ids)]
                            exp_values[exp_index] = numpy.array(exp_list)
                        else:
                            pass

            # bookkeeping
            self.properties.add(property)
            if indexed:
                self.indexed.add(property)

            # return if identifier is None
            if identifier is None:
                setattr(self, property, exp_values)
                return

        # if setting single element, make value array for all elements
        if id_ is not None:
            
            if new_property:
                self.setValue(identifier=None, name=property, default=default,
                              indexed=indexed)
            old_values = self.getValue(identifier=identifier, name=property)
            ids = self.getValue(identifier=identifier, name='ids')
            if isinstance(id_, (list, tuple, numpy.ndarray)):
                for one_id, one_val in zip(id_, value):
                    old_values[ids==one_id] = one_val
            else:
                old_values[ids==id_] = value
            value = old_values

        # find experiment index
        index = self.getExperimentIndex(identifier=identifier)

        # set new value(s)
        if not new_experiment:

            # identifier exists, set value
            try:
                exp_values[index] = value

            except IndexError:
                # identifier exists but property doesn't have a value for
                # this identifier
                missing = index - len(exp_values) + 1
                extend =  [copy(default) for ind in range(missing)]
                exp_values.extend(extend)
                exp_values[index] = value

            setattr(self, property, exp_values)

        else:

            # new identifier
            if id_ is not None:
                raise ValueError(
                    "Can't set element of indexed property " + property + 
                    " for a non-existing experiment " + identifier + ".")
            for cur_prop in self.properties:
                if cur_prop == 'identifiers':
                    self.identifiers.append(identifier)
                elif cur_prop == property:
                    exp_values.append(value)
                    setattr(self, property, exp_values)
                else:
                    cur_value = getattr(self, cur_prop)
                    cur_value.append(default)
                    setattr(self, cur_prop, cur_value)

    def get_indexed_data(self, identifiers=None, names=None, additional=[]):
        """
        Returns pandas.DataFrame that contains all indexed data.

        Columns correspond to properties. A row corresponds to a single element
        of one experiment and it is uniquely specified by identifier, index
        pair.

        If the current object does not contain any experiment identifier
        (self.identifiers), None is returned.

        Arguments:
          - identifiers: list of identifiers, if None self.identifiers are used 
          - names: list of indexed properties, if None self.indexed is used
          - additional: list of other properties

        Returns DataFrame, the columns are:
          - identifier
          - self.index (usually ids)
          - indexed properties, elements of self.indexed (sorted) or names, 
          except self.index 
          - all properties listed in arg additional
        """

        # sort out identifiers
        if identifiers is None:
            identifiers = self.identifiers
        if (identifiers is None) or (len(identifiers) == 0):
            return None

        # properties to includefigure out columns
        ids_name = self.index
        if names is None:
            names = sorted(self.indexed)
        try:
            names.remove('identifiers')
        except ValueError: pass
        try:
            names.remove(ids_name)
        except ValueError: pass
        columns_clean = names + additional
        columns = ['identifiers', ids_name] + columns_clean

        # get data for all experiments
        for ident in identifiers:
            if ident not in self.identifiers: continue
            
            data = {}
            data['identifiers'] = ident
            data[ids_name] = self.getValue(identifier=ident, name=ids_name)

            # get data for all indexed properties
            for name in columns_clean:
                data_value = self.getValue(identifier=ident, name=name)

                # deal with data >1 dimensional nd arrays
                if (isinstance(data_value, numpy.ndarray)
                    and len(data_value.shape) > 1):
                    data_value = [
                        data_value[ind] for ind
                        in list(range(data_value.shape[0]))]

                data[name] = data_value
                        
            # update data
            data_indexed_local = pd.DataFrame(data, columns=columns)
            try:
                data_indexed = data_indexed.append(
                    data_indexed_local, ignore_index=True)
            except (NameError, AttributeError):
                data_indexed = pd.DataFrame(data, columns=columns)

        return data_indexed

    indexed_data = property(
        get_indexed_data, doc="Indexed data in pandas.DataFrame")
                
    def get_scalar_data(self, identifiers=None, names=None):
        """
        Returns pandas.DataFrame that contains scalar (non-indexed) data.

        Columns correspond to properties. Rows correspond to individual 
        experiments (observations).

        Arguments:
          - identifiers: list of identifiers, if None self.identifiers is used
          - names: list of properties, if None all scalar properties
          are returned

        Returns dataframe having the following columns:
          - identifiers
          - other properties sorted by name, except identifiers
        """

        # sort out identifiers
        if identifiers is None:
            identifiers = self.identifiers
        if (identifiers is None) or (len(identifiers) == 0):
            return None

        # figure out columns
        if names is None:
            names = sorted(self.properties.difference(self.indexed))
        try: 
            names.remove('identifiers')
        except ValueError: pass
        columns_clean = names
        columns = ['identifiers'] + columns_clean

        # get data for all experiments
        data = {}
        for ident in identifiers:
            if ident not in self.identifiers: continue

            # get identifier
            try:
                data['identifiers'].append(ident)
            except (NameError, KeyError):
                data['identifiers'] = [ident]

            # get data for all properties
            for name in columns_clean:
                value = self.getValue(identifier=ident, name=name)
                try:
                    data[name].append(value)
                except (NameError, KeyError):
                    data[name] = [value]

        # set data
        data_scalar = pd.DataFrame(data, columns=columns)

        return data_scalar
        
    scalar_data = property(
        get_scalar_data, doc="Scalar data in pandas.DataFrame")
                

    #######################################################
    #
    # Methods that unite or separate observations
    #
    #######################################################

    def addData(self, source, names, identifiers=None, copy=True):
        """
        Adds properties listed in arg names of another Observations object
        (arg source) to this instance. 

        If arg names is a list added properties retain their names, and so 
        will overwrite the properties having same names of this instance (if 
        they exist). Otherwise, if names is a dictionary, the keys are the 
        property names in source object and values are the corresponding names 
        under which they are added to this object.

        All specified identifiers have to exist in both objects. If arg
        identifiers is None, identifiers of arg source have to exist in 
        this object.

        If arg copy is True, a copy of data is saved to the other object 
        (copy() method for numpy.ndarrays, deepcopy for the rest).

        Arguments:
          - source: (Observations) another object
          - names: list of property names of the source object, or a dictionary
          where keys are the property names in source and values are the
          new names
          - identifiers: list of experiment identifiers for which the data is
          copied. Identifiers listed here that do not exist among 
          identifiers of source are ignored.  
          - copy: Flag indicating if data is copied 
        """

        # set identifiers
        if identifiers is None:
            identifiers = source.identifiers

        # get and add data
        for nam in names:

            if isinstance(names, dict):
                new_nam = names[nam]
            elif isinstance(names, list):
                new_nam = nam

            for ident in identifiers:

                # copy data
                data = source.getValue(property=nam, identifier=ident)
                if copy:
                    if isinstance(data, numpy.ndarray):
                        data = data.copy()
                    else:
                        data = deepcopy(data)
                self.setValue(property=new_nam, identifier=ident, value=data)

            # just in case there's no identifiers
            self.properties.add(new_nam)
            if nam in source.indexed:
                self.indexed.add(new_nam)

    def join(self, obs):
        """
        Joins observations of this instance with those of arg obs. Only the 
        attributes listed in attribute self.properties are joined.

        No two identifiers can be the same. If that happens ValueError is
        raised.

        Arguments:
          - obs: Observations 
        """

        # check identifiers
        common = set(self.identifiers).intersection(set(obs.identifiers))
        if len(common) > 0:
            raise ValueError("Can't join observations becuase the following "
                             + " identifiers overlap: " + str(common))

        # in some cases identifiers were not in properties, so just in case
        properties = set(['identifiers'])
        properties.update(self.properties)

        # join all properties
        for name in properties:
            joined = getattr(self, name) + getattr(obs, name)
            setattr(self, name, joined)

    def joinExperiments(self, name, mode='join', identifiers=None):
        """
        Creates new Experiments instance by joining data (specified by arg
        name) of all experiments.

        Data of individual experiments (observations) are either joined
        together (mode 'join'), or an average value of each experiment is 
        used (mode 'mean') for the resulting instance.

        The joined data is ordered according to (the order of) arg identifiers.

        The resulting instance has the following properties: 
          - data name(s): the name of the data property stays the same
          - ids: set from 1 up (increment 1), correspond to each data point of
          the resulting instance 
          - idNames: unique string for each data value derived from identifier 
          and id for that value, see below.

        If data is scalar (one value per experiment), or the mode is 'mean' 
        experiment identifiers of this instance are saved as attribute 
        idNames of the resulting object. Alternatively, if data is indexed and
        mode is 'join' idNames of the resulting object for each data element is
        composed as identifier_id where identifier and id are specifying this
        data element. 

        Arguments:
          - name: name of the data attribute
          - mode: 'join' to join or 'mean' to average data
          - identifiers: list of experiment identifiers to be used here, if 
          None all are used. Non-existing identifiers are ignored. 

        Returns: Experiment instance
        """

        # wrap this instance in a Group
        from .groups import Groups
        groups = Groups()
        groups['_dummy'] = self

        # join the Group to get an Observations object
        joined_obs = groups.joinExperiments(name=name, mode=mode, 
                                            identifiers=identifiers)

        # extract Experiment object
        exp = joined_obs.getExperiment(identifier='_dummy')

        return exp 

    def remove(self, identifier):
        """
        Removes all data for the observation given by arg identifier.

        Argument:
          - identifier: observation identifier
        """

        # get experiment index
        index = self.getExperimentIndex(identifier=identifier)

        if index is None: return

        # remove from all properties
        for prop_name in self.properties:
            getattr(self, prop_name).pop(index)

        # make sure removed from 'categories' and 'identifiers'
        #if 'categories' not in self.properties:
        #    self.categories.pop(index)
        if 'identifiers' not in self.properties:
            self.identifiers.pop(index)

    def keep(self, identifiers):
        """
        Keeps only the experiments specified by arg identifiers and removes 
        data corresponding to all other experiments.

        Argument:
          - identifiers: list of experiment identifiers
        """

        # figure out identifiers to remove
        all_ = set(self.identifiers) 
        keep = set(identifiers)
        remove = all_.difference(keep)

        # remove experiments
        for ident in remove:
            self.remove(identifier=ident)

    def split(self, name, bins):
        """
        Extracts data from this instance according to the values of property
        given by arg name and arg bins, makes an instance out of each
        exctraction and returns them as list.

        If name is not an indexed property, individual observations (comprising
        this instance) are assigned to bins, by comparing the value of property 
        given by name of each observation with the bin limits specified in arg 
        bins. Lower limits are inclusive, while upper are exclusive except for
        the last bin. Consequently, if argument bins has n elements, they will
        form n-1 bins. 

        Otherwise, if name is an indexed property values of each element of 
        property specified by name of each observation is comparred with the
        bin limits. Elements of all other indexed properties are assigned to the
        same bins.

        Arguements:
          - name: name of the property whose value(s) is (are) comparred to bins
          - bins: bin limits (has to contain at least two elements)

        Returns: list of instances of this class.
        """

        # check if split by observations or by (indexed) elements of
        # observations 
        if name in self.indexed:
            indexed = True
        else:
            indexed = False

        # loop over bins
        binned = []
        for low, high in zip(bins[:-1], bins[1:]):
            
            # identify elements that belong to the current bin
            if high < bins[-1]:
                belong = [((value >= low) & (value < high)) \
                              for value in getattr(self, name)]
            else:
                belong = [((low <= value) & (value <= high)) \
                              for value in getattr(self, name)]
                
            # extract identified elements
            if indexed:
                new = self.extractIndexed(condition=belong)
            else:
                new = self.extract(condition=belong)

            # add them to the results
            binned.append(new)

        return binned

    def extract(self, condition=None, values=None, template=None, target=None):
        """
        Returns elements from (arg) target (observations) that have the same 
        positions as either the True elements of condition (if arg conditions 
        is specified), or the elements of (arg) values have in the (arg) 
        template (if arg condition is not specified). 

        Either condition or values has to be specified. In the later case 
        self.index (more precisely getrattr(self, name)) is used as a 
        template if template argument is None.

        If target is specified it has to be eather an array, or a nested
        array (list). 

        If target is None, all arrays of this instance that are indexed like 
        self.index (their names are in self.indexed) are used. A new instance 
        of this class is crated to hold these arrays. Values of attributes 
        categories, indentifiers index and indexed are copied form this 
        instance to the new instance. The new instance is returned.

        Structures of arguments condition, values, template and target have to 
        be the same (flat or nested arrays, same array lengths). In other words,
        these ale the same as data structures for properties of this class.
        The returned value has the same structure.

        Arguments:
          - condition: boolean values indicating which elements to extract
          - values: values of template
          - template: array (if None name of this array is given in self.index)
          - target: array to be extracted, or all arrays of this instance
          if None

        Returns: array of extracted values of target, or an instance of this
        class containing extracted values.
        """

        # set template
        if (condition is None) and (template is None):
            index_name = getattr(self, 'index')
            template = getattr(self, index_name)

        # extract
        if target is None:

            # extract all arrays (indexed variables)
            extracted = deepcopy(self)
            for name in self.indexed:
                targ = getattr(self, name)
                if condition is None:
                    value = self._valueExtractOne(values=values, 
                                            template=template, target=targ)
                else:
                    value = self._conditionExtractOne(condition=condition, 
                                                      target=targ)
                setattr(extracted, name, value)

            return extracted

        else:

            # extract one (possibly nested) array
            if condition is None:
                result = self._valueExtractOne(values=values, 
                                               template=template, target=target)
            else:
                result = self._conditionExtractOne(condition=condition, 
                                                   target=target)

            return result

    def _valueExtractOne(self, values, template, target, nested=None):
        """
        Extracts from one target array
        """
        
        # find out if values is nested
        if nested is None:
            nested = pyto.util.nested.is_nested(values)

        if nested:

            # multiple observations
            result = []
            for val, templ, targ in zip(values, template, target):
                one = self._valueExtractOne(values=val, template=templ, 
                                            target=targ, nested=False)
                result.append(one)
            return result

        else:

            # one observation
            all = numpy.equal.outer(values, template)
            condition = numpy.bitwise_or.reduce(all, axis=0)
            return target[condition]

    def _conditionExtractOne(self, condition, target, nested=None):
        """
        Extracts from one target array
        """
        
        # find out if values is nested
        if nested is None:
            nested = pyto.util.nested.is_nested(condition)

        if nested:

            # multiple observations
            result = []
            for cond, targ in zip(condition, target):
                one = self._conditionExtractOne(condition=cond, 
                                                target=targ, nested=False)
                result.append(one)
            return result

        else:

            # one observation
            return target.compress(condition, axis=0)

    def extractIndexed(self, condition, other=[]):
        """
        Extracts elements of individual observations of this instance, according
        to the arg condition and returns an object of this class containig 
        the extracted elements.

        The structure of arg condition has to correspond to the structure of 
        this instance, that is it has to be a list of the same number of
        elements as the number of observations, and each of these elements
        has to be a ndarray containing elements corresponding to the elements 
        of the indexed properties of this instance. 

        Elements of this instance which correspond to True elements in arg 
        condition are extracted.  

        Elements are extracted from all indexed properties (as specified in 
        self.indexed), while the other properties are copied in full.

        Arguments:
          - condition: condition
        """

        # coppy current object
        extracted = deepcopy(self)

        # extract indexed properties
        all_names = self.indexed.union(other)
        for name in all_names:

            # extract current property 
            extracted_values = []
            for val, cond in zip(getattr(self, name), condition):

                # extract elements of the current observation
                extr_val = val[cond]
                extracted_values.append(extr_val)

            # update 
            if name in other:
                extracted_values = deepcopy(extracted_values)
            setattr(extracted, name, extracted_values)

        return extracted

    def splitIndexed(self):
        """
        Splits this instance according to the indexed properties.

        Returns a list of instances of this class where each instance contains
        only one value of each of the indexed properties. Other (non-indexed)
        properties are copied from this to each of the resulting instances.

        If indexed properties of observations comprising this instance contain 
        different number of elements, only the common elements are returned.
        That is, the length of the returned list is the minimum number of
        elements that indexed properties have.

        Returns: list of instances of this class.
        """

        # get ids
        ids = getattr(self, self.index)

        # loop over indexed positions
        n_ids_min = min(len(obs_ids) for obs_ids in ids)
        result = []
        for ind in range(n_ids_min):

            # initial (all False) condition
            cond = [numpy.zeros(len(ids[obs_ind]), dtype='bool') \
                        for obs_ind in range(len(ids))]

            # set current condition
            for obs_ind in range(len(ids)):
                cond[obs_ind][ind] = True

            # extract values for the current index position
            result.append(self.extractIndexed(condition=cond))
            
        return result

    def transpose(self):
        """
        Rearanges indexed data so that observations and indices are exchanged

        ToDo
        """
        raise NotImplementedError("Sorry, not implemented yet")


    #######################################################
    #
    # Methods that generate new properties
    #
    #######################################################

    def apply(self, funct, args, kwargs={}, name=None, indexed=True):
        """
        Applies (arg) funct to properties args where other arguments are given
        in arg kwargs. 

        In other words, for each observation and for all observation elements 
        funct is applied to the corresponding values of the properties.

        Arguments to funct can be:
          - property names, specified in args (positional)
          - other values, specified in kwargs (keyword)

        Returns the results as a list (representing observations) of ndarrays 
        (observation elements) if name is None. Otherwise a new 
        property is created to hold the result. This new name can be the same
        as an already existing property, in which case the original values are
        owerwriten. The name of the new property is added to properties and
        indexed attributes.

        For example:

          def add(x, y): return x + y
          obs.apply(funct=add, args=['vector'], kwargs={'y', 5)

        will return property vector of instance obs increased by 5, while

          def add(x, y): return x + y
          obs.apply(
              funct=add, args=['vector'], kwargs={'y', 5}, name='new_vector')

        will save the new values as property new_vector.

        Arguments:
          - function: function
          - args: list of indexed property names that are used as
          arguments to funct
          - kwargs: (dict) other arguments passed to funct
          - name: name of the new property
          - indexed: specifies if the new property is indexed

        Sets indexed property name, or returns the result (list of ndarrays).
        """

        # get values of all properties needed for arguments and arrange them in
        # list of lists where the outer list is indexed by observations and the
        # inner lists contain values for all arguments
        prop_values = [[getattr(self, prop_name)[obs_ind] 
                        for prop_name in args] \
                           for obs_ind in range(len(getattr(self, args[0])))]
        
        # apply funct 
        res = [funct(*values, **kwargs) for values in prop_values]

        # set properties or return
        if name is None:
            return res
        else:
            setattr(self, name, res)
            self.properties.add(name)
            if indexed:
                self.indexed.add(name)

    def pixels2nm(self, name, conversion, power=1):
        """
        Converts all values of ndarray self.name from pixels to nm.

        Arguments:
          - name: name of the variable that need to be converted
          - conversion: dictionary of pixel sizes (in nm) for each identifier
          - power: used when the new property has dimensions pixels^power,
          can be positive or negative
        """
        value = getattr(self, name)
        #if isinstance(value, ndarray):
        in_nm = [in_pix * conversion[ident]**power for in_pix, ident \
                         in zip(value, self.identifiers)]
        
        return in_nm

    def rebin(self, name, bins):
        """
        Rebins histogram-like data specified by name according to the arg 
        bins and returns the new histogram.

        Binning is done according to the position (index) of elements (property
        values) within each onservation. For example:

          obs.volume[0] = [1, 2, 3, 4, 5, 6, 7]
          new_obs = obs.rebin(name='volume', bins=[0,2,4,6])
          -> new_obs.volume[0] = [3, 7, 11]

        When bins are not integers, trapesoidal rule is used to calculate
        rebinned data.

        The rebinned histogram has one element less than bins.

        Arguments:
          - name: (string) name of the attribute to be rebinned, where the
          attribute is expected to be a nested list
          - bins: (nested list) list of bins for each observation, in the
          same order as self.identifiers.
        """

        # prepare arrays
        old = getattr(self, name)
        new = []
        
        # calculate new histogram
        for old_one, bins_one in zip(old, bins):
            new_one = numpy.zeros(len(bins_one)-1)

            # bins and fractions
            floor = numpy.floor(bins_one).astype('int')
            ceil = numpy.ceil(bins_one).astype('int')
            mod = numpy.mod(bins_one, 1)

            # new histogram
            for ind in range(len(bins_one)-1):
                new_one[ind] += (1 - mod[ind]) * (ceil[ind] - floor[ind]) \
                    * old_one[floor[ind]] 
                new_one[ind] += mod[ind+1] * (ceil[ind+1] - floor[ind+1]) \
                    * old_one[floor[ind+1]]
                new_one[ind] += old_one[ceil[ind]:floor[ind+1]].sum()

            new.append(new_one)

        return new


    #######################################################
    #
    # Statistics
    #
    #######################################################

    def doStats(self, name, bins=None, fraction=None, identifiers=None, 
                new=True, ddof=1, remove_nan=True):
        """
        Does statistics on data specified by name for each experiment
        separately. 

        If arg bin is None, calculates basic statistics. The calculations are
        saved as following attributes:
          - data: deepcopied data 
          - mean: mean
          - std: n-ddof degrees of freedom,
          - n: number of individual data points
          - sem: standard error of means
          - ddof: delta degrees of freedom, denominator for std is N-ddof

        Alternatively, if arg bins is given calculates histogram of the data 
        according to bins and saves it as the following attributes:
          - ids: left bin limits 
          - data: deepcopied data 
          - histogram: (ndarray of length 1 less than bins) histogram values 
          (indexed)
          - probability: (ndarray of length 1 less than bins) 
          histogram / sum_of_histogram (indexed)
          - n: number of individual data points
          - fraction: histogram[fraction] / n

        If arg remove_nan is True, data points that are numpy.nan are removed.
 
        If arg new is True, a new instance of this class is created that 
        contain only the statistics attributes and attribute data (contains 
        data). If new is False statistics attributes are added to the current
        instance.

        If arg identifiers is specified and new is True, arg identifiers 
        determines the order of identifiers in the resulting instance. If None,
        all existing identifiers are used.

        If new is True, and the resulting object has no identifiers, all
        data properties are initialized to [].

        Arguments:
          - name: data attribute name, can't be any of statistics results 
          listed above
          - bins: histogram bins, if specified histogram is calculated
          - fraction: (int) position of the histogram bin for which the
          fraction of total histogram values is calculated.
          - identifiers: list of experiment identifiers for which the stats
          are calculated. Identifiers listed here that do not exist among 
          identifiers of this instance are ignored.  
          - new: flag indicating if new instance is created
          - ddof: delta degrees of freedom, denominator for std is N-ddof

        Returns new instance is new is True
        """

        # data names
        if bins is None:
            data_names = ['data', 'mean', 'std', 'n', 'sem']
        else:
            data_names = ['data', 'n', 'histogram', 'probability', 'fraction']

        # make new instance if needed
        if new is True:
            observ = self.__class__()
        else:
            observ = self
            if name in data_names:
                raise ValueError(
                    "Property name " + name + " conflicts with the names of "
                    + "calculated statistical values. Set argument new to True "
                    + "to avoid this problem.")

        if identifiers is None:
            identifiers = self.identifiers
        ident_exists = False
        for ident in identifiers:

            # restrict to specified identifiers
            if ident not in self.identifiers:
                continue
            ident_exists = True

            # get and set data
            data = self.getValue(identifier=ident, property=name)
            if remove_nan:
                data = numpy.compress(numpy.isnan(data)==False, data)
            if new:
                observ.setValue(identifier=ident, property='data', value=data)

            # calculate
            if bins is None:

                # basic stats
                mean = numpy.mean(data)
                observ.setValue(property='mean', identifier=ident, value=mean)
                n = len(data)
                observ.setValue(property='n', identifier=ident, value=n)
                if (n - ddof) < 0:
                    std = numpy.nan
                else:
                    # don't show warning if too few data elements
                    if len(data) > ddof:
                        std = numpy.std(data, ddof=ddof)
                    else:
                        std = numpy.NaN
                    # nicer but doesn't work
                    #try:
                    #    std = numpy.std(data, ddof=ddof)
                    #except RuntimeWarning:
                    #    if len(data) > ddof:
                    #        raise
                    #    else:
                    #        pass                        
                        
                observ.setValue(property='std', identifier=ident, value=std)
                sem = std / numpy.sqrt(n)
                observ.setValue(property='sem', identifier=ident, value=sem)

            else:

                # make histogram from data values
                histo_ids = numpy.asarray(bins)[:-1]
                observ.setValue(property='ids', identifier=ident, 
                                value=histo_ids, indexed=True)
                his, foo = numpy.histogram(data, bins=bins)
                observ.setValue(property='histogram', identifier=ident, 
                                value=his, indexed=True)
                n = len(data)
                observ.setValue(property='n', identifier=ident, value=n)
                prob = his / float(his.sum())
                observ.setValue(property='probability', identifier=ident, 
                                value=prob, indexed=True)
                if fraction is not None:
                    fract = prob[fraction]
                    observ.setValue(property='fraction', identifier=ident, 
                                    value=fract)

            # no identifiers, just initialize
        if (not ident_exists) and new:
            observ.initializeProperties(names=data_names)

        # return if new object
        if new:
            return observ

    def doStatsByIndex(self, name, identifiers=None, identifier='', ddof=1):
        """
        Calculates basic statistics (mean, std, sem, n) for each data point
        accross the specified experiments.

        In case experiments don't have the same number of data points the min
        number of points (across experiments) is used. Furthermore,
        the data points are grouped according their position within data
        arrays (and not according to actual ids).        

        Arguments:
          - name: name of the property to be analyzed (has to be indexed)
          - identifiers: list of experiment identifiers, None for all 
          experiments
          - identifier: identifier of the resulting Experiment object
          - ddof: delta degrees of freedom, denominator for std is N-ddof
      
        Returns: (Experiment) object containing the calculated stats having
        the following properties:
          - data: 2D ndarray containing all data, where the first index 
          (axis 0) denotes different experiments  
          - mean, std, sem, n: statistical properties
        """

        # new object to hold data
        exp = Experiment()

        # set identifiers
        if identifiers is None:
            identifiers = self.identifiers
        else:
            identifiers = [ident for ident in identifiers 
                           if ident in self.identifiers]
        ident_exists = False

        # find min data length
        min_len = min(len(self.getValue(identifier=ident, property=name)) 
                      for ident in identifiers)

        # put all data in a list
        all_data_seq = []
        for ident in identifiers:

            # restrict to specified identifiers
            if ident not in self.identifiers:
                continue
            ident_exists = True

            # add data
            single_data = self.getValue(identifier=ident, property=name)
            all_data_seq.append(single_data[:min_len])

            # add ids
            exp.ids = copy(self.getValue(identifier=ident, property='ids'))

        # calculate stats and save
        data = numpy.vstack(all_data_seq)
        exp.data = data
        exp.mean = data.mean(axis=0)
        exp.std = data.std(axis=0, ddof=ddof)
        exp.n = data.shape[0]
        exp.sem = exp.std / numpy.sqrt(exp.n)
        exp.identifier = identifier

        # set metadata
        exp.properties = set(['ids', 'identifier', 'mean', 'std', 
                              'sem', 'n', 'data'])
        exp.indexed = set(['ids', 'mean', 'std', 'sem'])

        return exp

    def doInference(self, test, reference, name='data', identifiers=None):
        """
        Tests if data for different experiments are statistically different
        from a given reference experiment(s) data.

        Data for each of the experiments comprising this object are compared
        with the reference experiment(s). There can be one or more  reference 
        experiments.

        In case of paired t-test (test='t_rel') numpy.NaN entries and the
        coreresponding reference data (or the other way around) are ignored.

        Arguments:
          - test: statistical test used: 't', 'h', 'u', or any other stated
          in doInference() doc.
          - reference: experiment identifier for the reference experiment, 
          a dictonary where each experiment identifier (keys) is associated
          with its reference (values), or a list of reference identifiers
          corresponding (in the same order) to the actual identifiers.
          - name: data attribute name
          - identifiers: list of experiment identifiers for which the stats
          are calculated. Identifiers listed here that do not exist among 
          identifiers of this instance are ignored.  

        Sets attributes:
          - testValue: values of t, h or u depending on the test
          - testSymbol: 't', 'h', or 'u', depending on the test
          - confidence: probability that data from an experiment and the/its 
          reference come from the same population
        All attributes are lists with elements corresponding to experiments,
        except for testSymbol which is a string
        """

        # get data
        #values = getattr(self, name)

        # parse test ant check 
        test_method, test_symbol = self.__class__.parseTest(test=test)
        if ((test_symbol == 'chi2') and (name != 'histogram') 
            and (name != 'count')):
            logging.debug(
                "Applying chi2 test directly to data (" + name + "). This "
                + "might be ok, but normally chi2 is applied on histograms.")

        # test all experiments
        self.testValue = []
        self.confidence = []
        self.testSymbol = []
        self.properties.update(['testValue', 'confidence', 'testSymbol'])
        for ident in self.identifiers:
 
            # restrict to specified identifiers
            if (identifiers is not None) and (ident not in identifiers):
                continue

            # get data
            data = self.getValue(identifier=ident, property=name)

            # set reference
            if isinstance(reference, dict):
                ref_data = self.getValue(identifier=reference[ident], 
                                         property=name) 
                self.setValue(identifier=ident, property='reference', 
                              value=reference[ident])
            elif isinstance(reference, list):
                index = self.getExperimentIndex(identifier=ident)
                ref_data = self.getValue(identifier=reference[index], 
                                         property=name) 
                self.setValue(identifier=ident, property='reference', 
                              value=reference[index])
            else:
                ref_data = self.getValue(identifier=reference, property=name)
                self.setValue(identifier=ident, property='reference', 
                              value=reference)

            # test this experiment
            if (len(data) == 0) or (len(ref_data) == 0):
 
                # no data
                test_value = numpy.NaN
                confid = numpy.NaN 

            elif test == 't_rel':

                # paired t test, remove nans
                pair_data = []
                pair_ref = []
                for x, y in zip(data, ref_data):
                    if not numpy.isnan(x) and not numpy.isnan(y):
                        pair_data.append(x)
                        pair_ref.append(y)
                test_value, confid = test_method(pair_data, pair_ref)

            else:
 
                # other tests
                try:
                    # generates RuntimeWarning if too few data
                    # better not catch
                    test_value, confid = test_method(data, ref_data)
                except ValueError:
                    # deals with Kruskal
                    if (data == ref_data).all():
                        test_value = 0.
                        confid = 1.
 
            # update variables
            self.testValue.append(test_value)
            self.confidence.append(confid)
            self.testSymbol.append(test_symbol)

    def doCorrelation(self, xName, yName, test=None, regress=False, 
                      reference=None, mode=None, new=True, identifiers=None, 
                      out=sys.stdout, format_={}, title=''):
        """
        Tests if data specified by args xName and yName are correlated.

        If mode is None, data from each experiment is analyzed separately. 
        Otherwise, if it's 'join', data from all experiments are taken
        together. In this case arg new is ignored (effectively set to True).

        If new is True, a new object is created to hold the data and results.

        If arg identifiers is specified and new is True, arg identifiers 
        determines the order of identifiers in the resulting instance. If None,
        all existing identifiers are used.

        If new is True, and the resulting object has no identifiers, all
        data properties are initialized to [].

        The results are saved as the following attributes:
          - xData, yData: deepcopied xName, yName data, only if new is True
          - n: number of individual data points
          - testValue: correlation test value
          - testSymbol: currently 'r' or 'tau', depending on the test
          - confidence: confidence
          - aRegress, bRegress: slope and intercept of the regression line 
          (if arg regress is True)

        Arguments:
          - xName, yName: property names of data 
          - test: correlation test, currently 'r' (or 'pearson'), 'tau' 
          (or 'kendall')
          - regress: flag indicating if regression (best fit) line is calculated
          - mode: None, or 'join'
          - new: flag indicating if new instance is created and returned
          - identifiers: list of experiment identifiers for which the 
          correlation is calculated. Identifiers listed here that do not 
          exist among identifiers of this instance are ignored.  
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first

        Returns object containing data and results (only if new is True,
        or join is 'join'). The object is Observations if join is None,
        or Experiment if join is None.
        """

        # parse test
        if test is not None:
            test_method, test_symbol = self.__class__.parseTest(test=test)

        # set argument defaults
        if identifiers is None:
            identifiers = self.identifiers

        ident_exists = False
        if mode is None:

            # make new instance if needed
            if new:
                corr = self.__class__()
            else:
                corr = self

            for ident in identifiers:

                if ident not in self.identifiers:
                    continue
                ident_exists = True

                # get data
                x_data = self.getValue(identifier=ident, property=xName)
                y_data = self.getValue(identifier=ident, property=yName)

                # test and set related properties
                if test is not None:
                    test_value, confid = test_method(x_data, y_data)
                    corr.setValue(property='testValue', identifier=ident, 
                                  value=test_value) 
                    corr.setValue(property='confidence', identifier=ident, 
                                  value=confid) 
                    corr.setValue(property='testSymbol', identifier=ident, 
                                  value=test_symbol)

                # regression
                if regress:
                    reg = scipy.stats.linregress(x_data, y_data) 
                    a_reg, b_reg, r_reg, p_reg, err_reg = reg
                    corr.setValue(property='aRegress', identifier=ident, 
                                  value=numpy.array(a_reg))
                    corr.setValue(property='bRegress', identifier=ident, 
                                  value=numpy.array(b_reg))

                # set other properties
                corr.setValue(property='n', identifier=ident, 
                                value=len(x_data))
                if new:
                    corr.setValue(property='xData', identifier=ident, 
                                    value=x_data) 
                    corr.setValue(property='yData', identifier=ident, 
                                    value=y_data) 

            # no identifiers, just initialize
            if (not ident_exists) and new:
                if test is not None:
                    corr.initializeProperties(
                        names=['testValue', 'confidence', 'testSymbol', 'n',
                               'xData', 'yData'])
                else:
                    corr.initializeProperties(names=['n', 'xData', 'yData'])
                if regress:
                    corr.properties.add('aRegress')
                    corr.properties.add('bRegress')

        elif mode == 'join':

            # first join and then correlate
            exp = self.joinExperiments(name=[xName, yName], mode='join', 
                                       identifiers=identifiers)
            corr = exp.doCorrelation(
                xName=xName, yName=yName, test=test, regress=regress, 
                reference=reference, out=None)

        # print
        names = ['n']
        if test is not None:
            names.extend(['testValue', 'confidence'])
        if regress:
            names.extend(['aRegress', 'bRegress'])
        corr.printData(names=names, out=out, format_=format_, title=title)

        # return
        if new or (mode == 'join'):
            return corr

    @classmethod
    def parseTest(cls, test):
        """
        Returns (method, symbol) for the statistical test corresponding to
        the arg test.

        Valid test values, and the corresponding tests and symbols are:
          - 't_ind' or 't': scipy.stats.ttest_ind, 't' (independent samples)
          - 't_rel': scipy.stats.ttest_rel, 't_rel' (related or paired t-test)
          - 'kruskal', 'kruskal-wallis', or 'h': scipy.stats.kruskal, 'h'
          - 'mannwhitney' or 'u': scipy.stats.mannwhitneyu, 'u'
          - 'pearson' or 'r': scipy.stats.pearsonr, 'r'
          - 'kendall' or 'tau' : scipy.stats.kendalltau, 'tau'
          - 'chi2' or 'chisquare': pyto.util.scipy_plus.chisquare_2, 'chi2'
        """

        # tests ans symbols  
        test_dict = {
            't_ind' : (scipy.stats.ttest_ind, 't'),
            't' : (scipy.stats.ttest_ind, 't'),
            't_rel' : (scipy.stats.ttest_rel, 't_rel'),
            'kruskal' : (scipy.stats.kruskal, 'h'),
            'kruskal-wallis' : (scipy.stats.kruskal, 'h'),
            'h' : (scipy.stats.kruskal, 'h'),
            'mannwhitney' : (scipy.stats.mannwhitneyu, 'u'),
            'u' : (scipy.stats.mannwhitneyu, 'u'),
            'r' : (scipy.stats.pearsonr, 'r'),
            'pearson' : (scipy.stats.pearsonr, 'r'),
            'tau' : (scipy.stats.kendalltau, 'tau'),
            'kendall' : (scipy.stats.kendalltau, 'tau'),
            'chisquare' : (pyto.util.scipy_plus.chisquare_2, 'chi2'),
            'chi2' : (pyto.util.scipy_plus.chisquare_2, 'chi2') }

        try:
            value = test_dict[test]
        except KeyError:
            raise ValueError(
                'Test ' + str(test) + " not understood. Defined tests are: "
                + "'t' (or 't_ind'), 'kruskal' ('kruskal-wallis' or 'h'), and "
                + "'mannwhitneyu' (or 'u').")

        return value

    #######################################################
    #
    # Output
    #
    #######################################################

    def printStats(self, out=sys.stdout, names=None, identifiers=None, 
                   format_={}, title=None):
        """
        Prints basic statistics and inference analysis results contained
        in this instance.

        Arguments:
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name, None for no output
          - names: list of attribute names to be printed, None for a default
          names list: ['mean', 'std', 'sem', 'n', 'testValue', 'confidence']
          - identifiers: list of experiment identifiers, None for all 
          experiments
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first
        """

        # default names
        default_names = ['mean', 'std', 'sem', 'fraction', 'n', 'testValue', 
                         'confidence']
        if names is None:
            names = default_names

        # set format
        loc_format = {
            'mean' : '    %5.2f ',
            'std' : '    %5.2f ',
            'sem' : '    %5.2f ',
            'n' : '    %5d ',
            'fraction' : '    %5.2f ',
            'testValue' : '    %5.2f ',
            'confidence' : '   %7.4f'
            }
        if format_ is not None:
            loc_format.update(format_)
 
        # print
        self.printData(names=names, out=out, identifiers=identifiers, 
                       format_=loc_format, title=title)

    def printData(self, names, out=sys.stdout, identifiers=None, 
                  format_={}, title=None):
        """
        Prints data of this instance.

        Arguments:
          - names: list of property (attribute) names to be printed
          - out: output stream, sys.stdout (default) for standard out, if
          string it's understood as a file name, None for no output
          names list: ['mean', 'std', 'sem', 'n', 'testValue', 'confidence']
          - identifiers: list of experiment identifiers, None for all 
          experiments
          - format_: dictionary containing data formats (keyes are attribute
          names) and values are formating strings. If None default names are
          used.
          - title: string that's printed first
        """

        # default data format
        default_format = '    %5.2f '

        # set output 
        if out is None:
            return
        elif isinstance(out, basestring):
            out = open(out)

        # print title
        if title is not None:
            out.write(os.linesep + title + os.linesep)

        # set identifiers
        if identifiers is None:
            identifiers = self.identifiers

        # set names
        names = [name for name in names if name in self.properties]

        # make identifier format based on the longest identifier
        ident_length = max([len(x) for x in identifiers])
        ident_length = max(ident_length, 10)
        ident_format = '%' + str(ident_length) + 's '

        # replace 'testValues' by actual testSymbol in names
        head_names = []
        for nam in names:
            if (nam == 'testValue') and (len(identifiers) > 0):
                head_names.append(self.testSymbol[0])
            else:
                head_names.append(nam)
            
        # print column (data) names
        var_head = ident_format % 'Identifier'
        var_head = var_head + (('%9s ' * len(head_names)) % tuple(head_names))
        out.write(var_head + os.linesep)

        # make one row format string
        var_format = ''.join([format_.get(name, default_format) 
                              for name in names])
        row_format = ident_format + var_format + os.linesep

        # print rows
        for identif in identifiers:
            row_values = [self.getValue(identifier=identif, property=nam) 
                          for nam in names]
            row_values = [identif] + row_values
            out.write(row_format % tuple(row_values))

