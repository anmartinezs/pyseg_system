"""
Defines class Catalog that contains meta data for each experiment. The meta 
data includes filenames where the data obtained by image analysis is stored 
as well as other info such as instrumentation settings and (visual) 
characterization of the experimental data.


# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from past.builtins import basestring
from builtins import zip
from builtins import str
from builtins import object

__version__ = "$Revision$"

import warnings
import logging
from copy import copy, deepcopy
import re
import os.path
import sys
try:
    # depreciated in Python 3.7
    import imp
except ModuleNotFoundError:
    warnings.warn(
        "Module imp not loaded, loading importlib instead "
        + "Run test/test_catalog.py to check the importlib related code.")  
    import importlib

import numpy
import scipy

       
class Catalog(object):
    """
    Typical usage involves reading meta-data form catalog(s) and making groups
    based on a feature. Attributes of this instance then contain the meta-data
    organized in groups. For example:

      cat = Catalog(catalog=catalog_names)
      cat.makeGroups(feature=group_defining_property)
      cat.property_name ...

    Important attributes:
      - self._db: (set by read()), holds all meta-data read from catalogs in 
      the following form:
        {property_a : {exp_1 : value_a1, exp_2 : value_a2, ...},
         property_b : {exp_1 : value_b1, ... },
         ...
        }
      where property_* are names of the meta-data variables and exp_n
      are unique experiment names (idetifiers)
      - self.property (where property can be any name not starting 
      with '_', these properties are set by makeGroups()) contains meta-data for
      the specified property from all experiments, organized by groups:
        {group_x : {exp_1 : value_1, exp_2 : value_2, ...},
         group_y : {exp_5 : value_5, exp_6 : value_6, ...},
        }
    """

    ###############################################################
    #
    # Initialization
    #
    ##############################################################

    def __init__(self, catalog=None, dir='.', type='distributed',
                 identifiers=None):
        """
        If arg catalog is specified, reads catalog info (see self.read())

        Arguments:
          - catalog: (string or a list of strings) file name(s) of the (all)
          catalog(s), or regular expression(s) matching all catalogs
          - dir: directory where catalogs reside
          - type: catalog type (currently 'distributed')
          - identifiers: list of experiment identifiers to be used, identifiers
          that are listed here but not found in the catalog are ignored
        """

        # set db
        if catalog is not None:
            self.read(
                catalog=catalog, dir=dir, type=type, identifiers=identifiers)
        else:
            self._db = {}


    #######################################################
    #
    # Methods for establishing database
    #
    #######################################################

    def read(self, catalog, dir='.', type='distributed', identifiers=None,
             extensions=['.pkl', '.dat']):
        """
        Reads all properties from all specified catalogs.

        Currently implemented only for 'distributed' catalog type. In this 
        case, each catalog contains meta-data for one experiment. All catalogs
        have to reside in the dictionary specified by arg 'dir' (default:
        current diectory).

        Catalog can be one or a list of catalog names, where each name is 
        a regular expression. All files in the specified directory that
        are matched by any of the catalog names is considered to be a catalog
        and it is read. Regular expression match is done in the search mode,
        that is a regular expression needs to match the beginning of a catalog
        file name.

        Properties are read only for experiment identifiers specified in arg 
        identifiers. Identifiers that are listed in this arg but not found in 
        the catalog(s) are ignored. Alternatively, if identifiers is None,
        properties of all experiemnts found in the catalog(s) are read.

        Any property value that is a string and ends with one of the elements
        of arg extensions is considered to be a file path and it is converted
        to absolute path. 

        Arguments:
          - catalog: (string or a list of strings) file name(s) of the (all)
          catalog(s), or regular expression(s) matching all catalogs
          - dir: directory where catalogs reside
          - type: catalog type (currently 'distributed')
          - identifiers: list of experiment identifiers to be used
          - extensions: list of file extensions
        """

        if type == 'single':

            raise ValueError("Sorry, type 'single' is not implemented. ",
                             "Available type: 'distributed'.")
            # import single file that contains dictionaries oof all names
            # not good
            #if dir is not None:
            #    catalog = os.path.join(dir, catalog)
            #mod_file, mod_dir, mod_desc = imp.find_module(catalog)
            #result = imp.load_module(catalog, mod_file, mod_dir, mod_desc)

        elif type == 'distributed':

            self._readDistributed(catalog=catalog, dir=dir, 
                                 identifiers=identifiers, extensions=extensions)

    def _readDistributed(self, catalog, dir='.', extensions=['.pkl', '.dat'],
                         identifiers=None):
        """
        Reads all properties from all specified catalogs for 'distributed'
        catalog type.
        """
        
        # initialize
        db = {}
        if not isinstance(catalog, list):
            catalog = [catalog]

        # read properties for all files that match any of the catalogs 
        for file_ in os.listdir(dir):
            for cat in catalog:
                  
                # keep only py files that match current catalog
                if not re.match(cat, file_):
                    continue
                base, ext = os.path.splitext(file_)
                if ext != '.py':
                    continue

                # derefernce links and remove .py to get module name
                cat_path = os.path.normpath(os.path.join(dir, file_))
                cat_path = os.path.realpath(cat_path)
                cat_dir, cat_base = os.path.split(cat_path)
                cat_base_main, cat_ext = os.path.splitext(cat_base)
                cat_mod = os.path.join(cat_dir, cat_base_main)

                # read current catalog module
                if 'imp' in sys.modules:
                    # depreciated in python 3.8
                    try:
                        mod_file, mod_dir, mod_desc = imp.find_module(cat_mod)
                    except ImportError:
                        # happens when run from a command line, don't know why
                        mod_file, mod_dir, mod_desc = imp.find_module(
                            cat_base_main, [cat_dir])
                    try:
                        module = imp.load_module(
                            base, mod_file, mod_dir, mod_desc)
                    finally:
                        mod_file.close()

                else:
                    # python 3
                    spec = importlib.util.spec_from_file_location(
                        cat_base_main, cat_mod)
                    module = spec.loader.load_module(spec.name)
                    
                # get identifier and skip non-specified identifiers
                identifier = module.identifier
                if ((identifiers is not None) 
                    and (identifier not in identifiers)):
                    continue

                # read other variables and put in the database
                for name, value in list(module.__dict__.items()):
                    # skip internal and identifier
                    if (name.startswith('_')) or (name == 'identifier'):
                        continue

                    # add (dereferenced, abs) dir to file names
                    if isinstance(value, basestring):
                        other, ext = os.path.splitext(value)
                        if ext in extensions:
                            value = os.path.join(cat_dir, value)
                            value = os.path.normpath(value)

                    # add variable 
                    if db.get(name) is None:
                        db[name] = {}
                    db[name][identifier] = value

        self._db = db

    def makeGroups(
            self, feature='category', db=None, include=None, 
            singleGroupName='foo', singleFeature='dummy',
            exclude=None, includeCateg=None, excludeCateg=None):
        """
        Classifies meta-data into groups according to a specified property
        (arg feature). Each value of the specified property defines a
        group.
        
        Sets an attribute for each property (meta-data variable name) existing
        in the database. These properties have the following form:

          {group_x : {exp_1 : value_1, exp_2 : value_2, ...},
           group_y : {exp_5 : value_5, exp_6 : value_6, ...},
           ...
          }
        where group_* are the values of the specifed feature (property used
        for making groups), exp_n are the experiment names (identifiers) and
        value_n are the property values. 

        In order to put everything in one group specify feature=None. In that 
        case the group name will be arg singleGroupName and the corresponding 
        property arg singleFeature. If the specified arg singleFeature
        already exists as a property, ValueError is raised. This is 
        incompatible with specifying arg db.

        This method does not change the database. Each new invocation of this
        method may add to the properties set by the previous invocation.

        Arguments:
          - feature: name of the property used for the classification
          - db: database containing all meta-data (default self._db)
          - include: list of experiments (identifiers) to include (default
          all experiments)
          - exclude: list of experiments (identifiers) to exclude (default
          none)
          - includeCateg: list of arg feature values (categories) to include
          (default all categories)
          - excludeCateg: list of arg feature values (categories) to exclude
          (default none)
        """

        # add a property to make only one group
        if feature is None:
            all_identifiers = self.getIdentifiers(property=None)
            dummy_values = dict(list(zip(
                all_identifiers, len(all_identifiers) * [singleGroupName])))
            self.add(name=singleFeature, values=dummy_values, overwrite=False)
            local_feature = singleFeature
        else:
            local_feature = feature

        # parse arguments
        if db is None:
            db = self._db

        # figure out if identifiers specified
        find_identifiers = False
        if include is None:
            find_identifiers = True

        # loop over all properties
        for name in db:

            # find identifiers to include
            if find_identifiers:
                idents = db[name]
            else:
                idents = include

            # loop over all identifiers (for this property)
            for identifier in idents:

                # check if need to exclude this identifier
                if (exclude is not None) and (identifier in exclude):
                    continue

                # find category and find out if it should be included
                categ = db[local_feature][identifier]
                if (includeCateg is not None) and (categ not in includeCateg):
                    continue
                if (excludeCateg is not None) and (categ in excludeCateg):
                    continue

                # add current property for current identifier to category
                try:
                    property = getattr(self, name)
                except AttributeError:
                    property = {}
                try:
                    property[categ]
                except KeyError:
                    property[categ] = {}
                try:
                    property[categ][identifier] = db[name][identifier]
                    setattr(self, name, property)
                except KeyError:
                    logging.info("Experiment " + identifier + " doesn't have "
                                 + "property " + name + ".")
                    
    def getIdentifiers(self, property='category'):
        """
        Returns a list of experiment identifiers from the database. 

        If property is None all identifiers are returned. Alternatively, it 
        property is given returns the identifiers existing for that property
        only.

        Argument:
          - property: (str) property name
        """
        if property is None:
            identifiers = set()
            for property in self._db:
                identifiers = identifiers.union(list(self._db[property].keys()))
            identifiers = list(identifiers)
        else:
            identifiers = list(self._db[property].keys())
        return identifiers

    def getProperties(self):
        """
        Returns a list of all properties (names) existing in the database.
        """
        props = list(self._db.keys())
        return props

    def add(self, name, values, overwrite=False):
        """
        Adds property name to this instance. The identifiers and the 
        corresponding property values are given by arg values. 

        In case the current instance already contains the property name 
        that has one of the identifiers specified in values, the behavior
        depends on arg overwrite. It it's True, the added values replace
        the existing ones. If it's False, ValueError is raised.

        Modifies self._db.

        Arguments:
          - name: property name
          - values: property values in the following form:
            {identifier_1 : value_1, identifier_2 : value_2, ...} 
          - owerwrite: flag indicating whether an existing property should
          be overwritten in case property / identifiier exists already
        """

        # check overwrite
        if not overwrite:
            if name in self.getProperties():
                common_idents = set(self._db[name]).intersection(values)
                if len( common_idents ) > 0:
                    raise ValueError(
                        "Attempt to owerwrite property " + name +
                        " for identifiers: " + str(common_idents))

        # add property
        if name in self.getProperties():
            self._db[name].update(values) 
        else:
            self._db[name] = values 

    def pool(self, categories, name):
        """
        Unites specified categories (groups) into a new category. 

        As a resuilt, for all properties of this instance
        
            self.property_x[name]

        will contain combined data of proeperty_x of all groups specified
        in arg. categories.

        The new category shares the data with the original categories. The
        other categories are not modified.

        Arguments:
          - categories: (list) categories (groups) to be united
          - name: new category (group) name (key)
        """

        # combine values for all properties
        for prop_name in self.getProperties():

            # get current property
            prop = self.__getattribute__(prop_name)

            # combine values of the current property for all categoreis 
            combined_value = {}
            for old_group in categories:
                combined_value.update(prop[old_group])

            # add combined values to the current property
            prop[name] = combined_value
