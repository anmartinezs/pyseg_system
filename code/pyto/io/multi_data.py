"""
Contains (abstract) class MultiData for input of data stored in multiple files. 
Each file contains results (data) of one observation (experiment).

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: multi_data.py 1461 2017-10-12 10:10:49Z vladan $
"""

__version__ = "$Revision: 1461 $"


import pickle
import warnings
import logging
import numpy

import pyto.util.attributes
#from pyto.analysis.observations import Observations


class MultiData(object):
    """
    Only subclasses of this class can be instantiated.

    The main purpose of subclasses of this class is to read data from one or
    more experiments (the exact subclass depends of the format of the saved 
    data), and organize the data into a structre that hold data from several
    experiments (..analysis.Observation class). 

    ToDo: see about ordering by index
    """

    ###############################################################
    #
    # Initialization
    #
    ##############################################################

    def __init__(self, files):
        """
        Initializes files attribute.

        Argument files has to be a dictionary of dictionaries, where ouside
        keys are group (category) names, inside keys experiment identifiers and 
        inside values file names. For example:

        files = {'group_a' : {'exp_1' : file_1,
                              'exp_2' : file_2,
                              ...             },
                 'group_b' : {'exp_5' : file_5,
                              ...             },
                 ...                           }
        """
        self.files = files
        self.properties = None

    ###############################################################
    #
    # Iterators
    #
    ##############################################################

    def categories(self):
        """
        Generator that yields category names.
        """
        for categ in self.files:
            yield categ

    def identifiers(self, category=None):
        """
        Generators that yeilds observation identifiers of the specified 
        category(ies), or identifiers of all categories if category is None.

        A category that does not exist (in self.files) is ignored.

        Argument:
          - category: (str) category, or a list of categories

        Yields: (str) identifiers
        """

        if category is None:
            
            # identifiers of all categories
            for categ in self.categories():
                for identif in self.identifiers(category=categ):
                    yield identif

        elif isinstance(category, list):

            # all identifiers of multiple categories
            for categ in category:
                if self.files.get(categ) is None:
                    continue
                for identif in self.files.get(categ):
                    if identif is not None:
                        yield identif

        else:

            # all identifiers of a specified category
            if self.files.get(category) is not None:
                for identif in self.files.get(category):
                    if identif is not None:
                        yield identif

    def data(self, category=None, identifier=None):
        """
        Generator that yields data (objects) for specified category(ies), 
        and/or specified identifier(s).

        If category is None all categories are used. Also, if identifier is
        None all identifiers for the cpecified category(ies) are used. 

        If more than one category is specified (argument category is None or
        a list) argument identifier should be None. Alternatively, if one
        category is specified, one or more (including all) identifiers can be
        specified

        In case multiple categories or identifiers are specified (as a list of
        strings) the order of yielded data has to have the same order as
        the specified list.

        Argument:
          - category: None for all categories, single category or a list of
          categories
          - identifier: None for all identifiers, single identifier, or a list
          of identifiers

        Yields: object, category, identifier
        """

        # chose category iterator
        if category is None:
            categ_iter = self.categories()
        elif isinstance(category, list):
            categ_iter = category
        else:
            categ_iter = [category]

        # iterate over categories
        for categ in categ_iter:

            # chose identifier iterator
            if identifier is None:
                ident_iter = self.identifiers(category=categ)
            elif isinstance(identifier, list):
                ident_iter = identifier
            else:
                ident_iter = [identifier]

            # iterate over identifiers and yield
            for ident in ident_iter:
                obj = self.getSingle(category=categ, identifier=ident)
                yield obj, categ, ident


    ###############################################################
    #
    # Input
    #
    ##############################################################

    def readProperties(self, properties, index='ids', category=None,
                       identifier=None, multi=None, compactify=True, deep='_'):
        """
        Reads data for all observations specified by category and identifier
        arguments, extracts specified properties and returns an instance of 
        ..analysis.Observations containig the extracted properties.

        If a property is not found, its value in the resulting Observations
        instance is set to None.

        If category is None all categories are used. Also, if identifier is
        None all identifiers for the cpecified category(ies) are used. 

        If more than one category is specified (argument category is None or
        a list) argument identifier should be None. Alternatively, if one
        category is specified, one or more (including all) identifiers can be
        specified

        In case multiple categories or identifiers are specified (as a list of
        strings) the order of yielded data has to have the same order as
        the specified list.

        The argument properties is a list that holds attribute names of 
        objects containing data. If a property does not contain '.', its values
        (for all observations) are saved in this instance as an attribute of 
        the same name.

        In case a property name contain one or more '.'s (that is the property
        is an attribute of an attribute ...) the values of that property (for
        all observations) are saved in this instance as an attribute whose 
        name is derived from the property name and arg deep. If deep is '_',
        the '.'s in the property name are replaced by '_'s. Alternatively, if
        deep is 'last', the part aver the last '.' is used for the name of the
        corresponding attribute of this instance. Actually, arg deep can have 
        the same values as arg 'mode' of pyto.util.attributes.get_deep_name().

        If arg index is specified it is also read together with other 
        properties.   

        If arg multi is an object, the properties read are assigned to
        attributes of multi. If it is None, a new instance of 
        pyto.analysis.Observations is created to hold these properties.

        If one of the observations could not be read all its properties
        are set to None.

        Note: can be called only on objects of subclasses of MultiData that 
        implement getSingle() method.

        Arguments:
          - properties: list of properties (strings) that are read 
          - category: (None, single item or a list) observation category(ies)
          - identifier: (None, single item or a list) observation identifier(s)
          - multi: instance that will hold the resulting data
          - index: name of the indexing attribute
          - compactify: convert all property arrays to compact form
          - deep: the mode of converting names of properties

        Sets:
          - self.properties: set of property names 

        Returns (..analysis.Observations) object containing requested  
        properties form all requested observations (experiments).

        Notes: 
          - depreciated, use readPropertiesGen() instead
          - still used by analysis.Vesicles
        """

        # initialize result object if needed
        if multi is None:
            from pyto.analysis.observations import Observations as Observations
            multi = Observations()

        # initialize self.properties
        self.properties = set(['categories', 'identifiers'])

        # add attribute ids if not there already
        if (index is not None) and (index not in properties):
            properties.insert(0, index)

        # check properties
        if ('categories' in properties) or ('identifiers' in properties):
            raise ValueError, "Sorry, 'categories' and 'identifiers' are " \
                "reserved names and can't be present in properties argument."

        # get objects and extract properties
        for obj, category, name in self.data(category=category, 
                                             identifier=identifier):

            # append current values of category and image name
            multi.categories.append(category)
            multi.identifiers.append(name)
            #multi.properties.add(self.properties)

            # append (or create) current values to all properties
            for attr in properties:

                # get multi.attr
                deep_name = pyto.util.attributes.get_deep_name(attr, mode=deep)
                self.properties.add(deep_name)
                multi.properties.add(deep_name)
                try:
                    old_values = getattr(multi, deep_name)
                except AttributeError:
                    old_values  = []

                # get obj.attr
                if obj is None:
                    new_value = None
                else:
                    try:
                        new_value = pyto.util.attributes.getattr_deep(obj, attr)
                        if (not self.compact) and compactify \
                                and (attr != index):
                            new_value = new_value[getattr(obj, index)]
                    except AttributeError:
                        logging.warning("Object " + name \
                                      + " does not have attribute " + attr) 
                        new_value = None
                    except TypeError:
                        logging.warning("Attribute " + attr + ' of object ' \
                                            + name + 'is unsubscribable.') 

                # append obj.attr to multi.attr
                old_values.append(new_value)
                pyto.util.attributes.setattr_deep(multi, attr, old_values,
                                                  mode=deep)

        # Note: The above part should be deleted and this uncommented,
        # when all subclasses of analysis.Groups are brought to the 
        # form appropriate for readPropertiesGen()
        #for multi, obj, category, identifier in self.readPropertiesGen(
        #    properties=properties, category=category, identifier=identifier,
        #    multi=multi, index=index, compactify=compactify, deep=deep):
        #    pass

        return multi

    def readPropertiesGen(self, properties, index='ids', indexed=[], 
                          category=None, identifier=None, multi=None, 
                          compactify=True, deep='_'):

        """
        Reads data for all observations specified by category and identifier
        arguments, extracts specified properties and returns an instance of 
        ..analysis.Observations containig the extracted properties.

        The same as readProperties() except that this is a generator that
        yields object (..analysis.Observations) holding requested data from all 
        observations (experiments) read so far and the object containing 
        data for the current observation.

        Note: Currently (r690) this method is not completely equivalent to
        readProperties(). When all subclasses of ..analysis.Groups are 
        modified to agree with this method, readProperties() should be 
        modified to be based on this method.

        If a property is not found, its value in the resulting Observations
        instance is set to None.

        If category is None all categories are used. Also, if identifier is
        None all identifiers for the cpecified category(ies) are used. 

        If more than one category is specified (argument category is None or
        a list) argument identifier should be None. Alternatively, if one
        category is specified, one or more (including all) identifiers can be
        specified

        In case multiple categories or identifiers are specified (as a list of
        strings) the order of yielded data has to have the same order as
        the specified list.

        The argument properties is a list that holds attribute names of 
        objects containing data. If a property does not contain '.', its values
        (for all observations) are saved in this instance as an attribute of 
        the same name.

        In case a property name contain one or more '.'s (that is the property
        is an attribute of an attribute ...) the values of that property (for
        all observations) are saved in this instance as an attribute whose 
        name is derived from the property name and arg deep. If deep is '_',
        the '.'s in the property name are replaced by '_'s. Alternatively, if
        deep is 'last', the part aver the last '.' is used for the name of the
        corresponding attribute of this instance. Actually, arg deep can have 
        the same values as arg 'mode' of pyto.util.attributes.get_deep_name().

        If arg index is specified it is also read together with other 
        properties.   

        If arg multi is an object, the properties read are assigned to
        attributes of multi. If it is None, a new instance of 
        pyto.analysis.Observations is created to hold these properties.

        If arg properties is a list, it has to contain a list of properties
        that are read. These properties are saved under property names derived 
        from arg properties (needs arg deep).

        If arg deep is '_', dots in name are replaced by underscores, and 
        if it is 'last', only the part after the rightmost dot is used as name.

        Alternatively, if arg properties is a dict, the keys are names of 
        properties that are read. The values are the names under which these
        properties are saved.

        Note: can be called only on objects of subclasses of MultiData that 
        implement getSingle() method.

        Arguments:
          - properties: list of names of properties that are read, or a dict
          where keys are names of properties that are read and values are
          the new property names used  
          - index: name of the indexing property 
          - indexed: list of indexed properties 
          - category: (None, single item or a list) observation category(ies)
          - identifier: (None, single item or a list) observation identifier(s)
          - multi: instance that will hold the resulting data
          - compactify: convert all property arrays to compact form
          - deep: the mode of converting names of properties

        Yields multi, object, category, identifier:
          - multi: (..analysis.Observations) holds requested properties
          - obj: contains data of the curent observation (experiment)
          - category: category
          - identifier: experiment identifier
        """

        # initialize result object if needed
        if multi is None:
            from pyto.analysis.observations import Observations as Observations
            multi = Observations()

        # initialize self.properties
        self.properties = set(['categories', 'identifiers'])

        # add attribute ids if not there already
        if isinstance(properties, dict):
            if (index is not None) and (index not in properties):
                properties[index] = 'ids'
        else:
            properties = list(properties)
            if (index is not None) and (index not in properties):
                properties.insert(0, index)

        # check properties
        if ('categories' in properties) or ('identifiers' in properties):
            raise ValueError, "Sorry, 'categories' and 'identifiers' are " \
                "reserved names and can't be present in properties argument."

        # get objects and extract properties
        for obj, category, name in self.data(category=category, 
                                             identifier=identifier):

            # append current values of category and image name
            multi.categories.append(category)
            multi.identifiers.append(name)

            # get ids
            ids = pyto.util.attributes.getattr_deep(obj, index)

            # find values of all properties and add them to the result object
            for attr in properties:

                # get new property name
                if isinstance(properties, dict):
                    deep_name = properties[attr]
                else:
                    deep_name = pyto.util.attributes.get_deep_name(
                        attr, mode=deep)
                self.properties.add(deep_name)
                multi.properties.add(deep_name)

                # get old value
                try:
                    old_values = getattr(multi, deep_name)
                except AttributeError:
                    old_values  = []

                # get property of the current observation 
                if obj is None:
                    new_value = None
                else:
                    try:
                        new_value = pyto.util.attributes.getattr_deep(obj, attr)
                        if (attr in indexed) and (attr != index) \
                                and (not self.compact) and compactify:
                            new_value = new_value[ids] 
                    except AttributeError:
                        logging.info("Object " + name \
                                         + " does not have attribute " + attr) 
                        new_value = None
                    except TypeError:
                        logging.warning("Attribute " + attr + ' of object ' \
                                            + name + ' is unsubscribable.') 

                # append property of the current observation to the group
                old_values.append(new_value)
                if isinstance(properties, dict):
                    setattr(multi, properties[attr], old_values)
                else:
                    pyto.util.attributes.setattr_deep(multi, attr, old_values,
                                                      mode=deep)

            # yield
            yield multi, obj, category, name
