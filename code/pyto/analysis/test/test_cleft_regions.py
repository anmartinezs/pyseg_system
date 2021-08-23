"""

Tests module analysis.cleft_regions and analysis.groups, which currently 
(r689) covers most of the functionality of other classes that inherits
from analysis.Groups.

ToDo:
  - fix bug in setup
  - See if some recent changes from test_cleft_layers need to be merged here
  and remove test_cleft_layers 
 
# Author: Vladan Lucic
# $Id: test_cleft_layers.py 823 2011-01-27 16:46:40Z vladan $
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
#from builtins import str
from builtins import range
#from past.utils import old_div

__version__ = "$Revision: 823 $"

from copy import copy, deepcopy
import pickle
import os.path
import unittest

import numpy
import numpy.testing as np_test 
import scipy

import pyto
from pyto.scene.cleft_regions import CleftRegions as SceneCleftRegions
import pyto.scene.test.common as scene_cmn
from pyto.analysis.catalog import Catalog
from pyto.analysis.cleft_regions import CleftRegions
from pyto.analysis.test import common
#from pyto.core.image import Image
#from pyto.segmentation.segment import Segment
#from pyto.segmentation.cleft import Cleft


class TestCleftRegions(np_test.TestCase):
    """
    Makes CleftRegions objects, pickles them and then tests reading and 
    analysis of these pickles.
    """

    def setUp(self):
        """
        Makes CleftLayer objects and pickles them for testing
        """
        
        # keep paths of all files generated here
        self.tmp_paths = []

        # make cleft layer pickles 
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        path = os.path.join(curr_dir, 'catalogs_b/results_1.pkl')
        common.make_and_pickle(file_=path)
        self.tmp_paths.append(path)
        path = os.path.join(curr_dir, 'catalogs_b/results_2.pkl')
        common.make_and_pickle(file_=path, data=1)
        self.tmp_paths.append(path)
        path = os.path.join(curr_dir, 'catalogs_a/results_3.pkl')
        common.make_and_pickle(file_=path, data=2)
        self.tmp_paths.append(path)
        path = os.path.join(curr_dir, 'catalogs_a/catalogs_c/results_4.pkl')
        common.make_and_pickle(file_=path, data=3)
        self.tmp_paths.append(path)

        # make cleft column pickles 
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        path = os.path.join(curr_dir, 'catalogs_b/cleft_columns_1.pkl')
        common.make_and_pickle(file_=path, mode='columns')
        self.tmp_paths.append(path)
        path = os.path.join(curr_dir, 'catalogs_b/cleft_columns_2.pkl')
        common.make_and_pickle(file_=path, mode='columns', data=1)
        self.tmp_paths.append(path)
        path = os.path.join(curr_dir, 'catalogs_a/cleft_columns_3.pkl')
        common.make_and_pickle(file_=path, mode='columns', data=2)
        self.tmp_paths.append(path)
        path = os.path.join(curr_dir, 
                            'catalogs_a/catalogs_c/cleft_columns_4.pkl')
        common.make_and_pickle(file_=path, mode='columns', data=3)
        self.tmp_paths.append(path)

        # desired for layers
        self.indexed = set(['ids', 'mean', 'std', 'min', 'max', 'volume',
                            'volume_nm', 'surface_nm', 'normalMean'])
        self.properties = self.indexed.union(
            ['width', 'width_nm', 'phiDeg', 'thetaDeg', 'minCleftDensityId', 
             'minCleftDensityPosition', 'relativeMinCleftDensity', 'cleftIds',
             'boundIds', 'bound1Ids', 'bound2Ids', 'identifiers',
             'boundThick', 'boundThick'])
        self.properties.update(['angleToYDeg'])
        self.properties.update(['results_file', 'results_obj', 'category', 
                                'feature', 'pixel_size', 'cleft_columns_obj'])
        self.ids = numpy.arange(1, 10, dtype=int)
        self.width = common.cleft_layers_width
        self.mean_0 = common.cleft_layers_density_mean
        self.volume = common.cleft_layers_density_volume

        # columns
        self.indexed_columns = set(
            ['ids', 'mean', 'std', 'min', 'max', 'volume', 'volume_nm'])
        self.properties_columns = self.indexed_columns.union(
            ['identifiers', 'results_file', 'results_obj', 'category', 
             'feature', 'pixel_size', 'cleft_columns_obj'])
        self.ids_columns = numpy.arange(1, 3, dtype=int)
        self.mean_0_columns = common.cleft_columns_density_mean
        self.volume_columns = common.cleft_columns_volume
        
        # layers on columns
        self.properties_layers_on_columns = self.properties.difference(
            set(['thetaDeg', 'phiDeg', 'angleToYDeg', 'width', 'width_nm',
                 'bound1Ids', 'bound2Ids', 'boundIds']))

    def test_read(self):
        """
        Tests read(mode=None)
        """

        # make catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        cat_dir = os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog()
        catal.read(dir=cat_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        catal.makeGroups(feature='category')

        # read all
        cleft_lay = CleftRegions.read(files=catal.results_obj, catalog=catal,
                                     mode='layers')
        
        # test meta-properties
        for category in ['first', 'second']:
            np_test.assert_equal(cleft_lay[category].indexed, self.indexed)
            np_test.assert_equal(cleft_lay[category].properties, 
                                 self.properties)

        # test catalog properties
        np_test.assert_equal(set(cleft_lay.first.identifiers),
                             set(['exp_2', 'exp_3']))
        np_test.assert_equal(set(cleft_lay.second.identifiers),
                             set(['exp_1', 'exp_4']))

        # test category
        for category in ['first', 'second']:
            np_test.assert_equal(cleft_lay[category].ids, [self.ids, self.ids])
            np_test.assert_equal(cleft_lay[category].width, 
                                 [self.width, self.width])

        # identifiers
        np_test.assert_equal(set(cleft_lay.first.identifiers),
                             set(['exp_2', 'exp_3']))
        np_test.assert_equal(set(cleft_lay.second.identifiers),
                             set(['exp_1', 'exp_4']))

        # data
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_1', property='mean'),
            self.mean_0)
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_2', property='mean'),
            self.mean_0 + 1)
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_3', property='mean'),
            self.mean_0 + 2)
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_4', property='mean'),
            self.mean_0 + 3)

        # width and volume in nm
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_1', property='width_nm'),
            self.width * 3.1)
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_2', property='width_nm'),
            self.width * 3.2)
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_3', property='width_nm'),
            self.width * 3.3)
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_4', property='width_nm'),
            self.width * 3.4)
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_1', property='volume_nm'),
            self.volume * 3.1**3)
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_2', property='volume_nm'),
            self.volume * 3.2**3)
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_3', property='volume_nm'),
            self.volume * 3.3**3)
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_4', property='volume_nm'),
            self.volume * 3.4**3)

    def test_read_order(self):
        """
        Tests read(order)
        """

        # make catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        cat_dir = os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=cat_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        catal.makeGroups(feature='category')

        # make order and read 
        order = {'first' : ['exp_3', 'exp_2'], 'second' : ['exp_4']}
        cleft_lay = CleftRegions.read(
            files=catal.results_obj, catalog=catal, mode='layers', order=order)

        # test catalog properties
        np_test.assert_equal(cleft_lay.first.identifiers, ['exp_3', 'exp_2'])
        np_test.assert_equal(cleft_lay.second.identifiers, ['exp_4'])

        # data
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_2', property='mean'),
            self.mean_0 + 1)
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_3', property='mean'),
            self.mean_0 + 2)
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_4', property='mean'),
            self.mean_0 + 3)

        # make order and read 
        order = {'first' : ['exp_3', 'exp_5'], 'second' : ['exp_4', 'exp_1']}
        print("Now a warning should appear:")
        cleft_lay = CleftRegions.read(
            files=catal.results_obj, catalog=catal, mode='layers', order=order)

        # test catalog properties
        np_test.assert_equal(cleft_lay.first.identifiers, ['exp_3'])
        np_test.assert_equal(cleft_lay.second.identifiers, ['exp_4', 'exp_1'])

    def test_read_layers(self):
        """
        Tests read(mode='layers')
        """

        # make catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        cat_dir = os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=cat_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        catal.makeGroups(feature='category')

        # read all
        cleft_lay = CleftRegions.read(files=catal.results_obj, catalog=catal,
                                     mode='layers')

        # test meta-properties
        for category in ['first', 'second']:
            np_test.assert_equal(cleft_lay[category].indexed, self.indexed)
            np_test.assert_equal(cleft_lay[category].properties, 
                                 self.properties.union(set(['normalMean'])))

        # test catalog properties
        np_test.assert_equal(set(cleft_lay.first.identifiers),
                             set(['exp_2', 'exp_3']))
        np_test.assert_equal(set(cleft_lay.second.identifiers),
                             set(['exp_1', 'exp_4']))

        # test category
        for category in ['first', 'second']:
            np_test.assert_equal(cleft_lay[category].ids, [self.ids, self.ids])
            np_test.assert_equal(cleft_lay[category].width, 
                                 [self.width, self.width])

        # identifiers
        np_test.assert_equal(set(cleft_lay.first.identifiers),
                             set(['exp_2', 'exp_3']))
        np_test.assert_equal(set(cleft_lay.second.identifiers),
                             set(['exp_1', 'exp_4']))

        # data
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_1', property='mean'),
            self.mean_0)
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_2', property='mean'),
            self.mean_0 + 1)
        np_test.assert_equal(
            cleft_lay.first.getValue(identifier='exp_3', property='mean'),
            self.mean_0 + 2)
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_4', property='mean'),
            self.mean_0 + 3)

        # ids
        np_test.assert_equal(cleft_lay.first.cleftIds, 
                             [list(range(3,8)), list(range(3,8))])
        np_test.assert_equal(cleft_lay.first.bound1Ids, [[1,2], [1,2]])
        np_test.assert_equal(cleft_lay.first.bound2Ids, [[8,9], [8,9]])
        np_test.assert_equal(cleft_lay.first.boundIds, [[1,2,8,9], [1,2,8,9]])

    def test_read_columns(self):
        """
        Tests read(mode='columns')
        """

        # make catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        cat_dir = os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=cat_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        catal.makeGroups(feature='category')

        # read all
        cleft_col = CleftRegions.read(files=catal.cleft_columns_obj, 
                                      catalog=catal, mode='columns')

        # test meta-properties
        for category in ['first', 'second']:
            np_test.assert_equal(cleft_col[category].indexed, 
                                 self.indexed_columns)
            np_test.assert_equal(cleft_col[category].properties, 
                                 self.properties_columns)

        # read columns with layers as reference
        cleft_lay = CleftRegions.read(files=catal.results_obj, catalog=catal,
                                     mode='layers')
        cleft_col = CleftRegions.read(
            files=catal.cleft_columns_obj, catalog=catal,
            mode='columns', reference=cleft_lay)

        # test meta-properties
        for category in ['first', 'second']:
            np_test.assert_equal(
                cleft_col[category].indexed, 
                self.indexed_columns.union(set(['normalMean'])))
            np_test.assert_equal(
                cleft_col[category].properties, 
                self.properties_columns.union(set(['normalMean'])))

        # test catalog properties
        np_test.assert_equal(set(cleft_col.first.identifiers),
                             set(['exp_2', 'exp_3']))
        np_test.assert_equal(set(cleft_col.second.identifiers),
                             set(['exp_1', 'exp_4']))

        # test category
        for category in ['first', 'second']:
            np_test.assert_equal(cleft_col[category].ids, 
                                 [self.ids_columns, self.ids_columns])

        # identifiers
        np_test.assert_equal(set(cleft_col.first.identifiers),
                             set(['exp_2', 'exp_3']))
        np_test.assert_equal(set(cleft_col.second.identifiers),
                             set(['exp_1', 'exp_4']))

        # data
        np_test.assert_equal(
            cleft_col.second.getValue(identifier='exp_1', property='mean'),
            self.mean_0_columns)
        np_test.assert_equal(
            cleft_col.first.getValue(identifier='exp_2', property='mean'),
            self.mean_0_columns + 1)
        np_test.assert_equal(
            cleft_col.first.getValue(identifier='exp_3', property='mean'),
            self.mean_0_columns + 2)
        np_test.assert_equal(
            cleft_col.second.getValue(identifier='exp_4', property='mean'),
            self.mean_0_columns + 3)
        np_test.assert_equal(
            cleft_col.first.getValue(identifier='exp_2', property='volume'),
            self.volume_columns)

    def test_read_layers_cleft(self):
        """
        Tests read(mode='layers_cleft')
        """

        # make catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        cat_dir = os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=cat_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        catal.makeGroups(feature='category')

        # read all
        cleft_lay = CleftRegions.read(files=catal.results_obj, catalog=catal,
                                     mode='layers_cleft')

        # test meta-properties
        for category in ['first', 'second']:
            np_test.assert_equal(cleft_lay[category].indexed, 
                                 self.indexed.union(['normalVolume']))
            np_test.assert_equal(cleft_lay[category].properties, 
                                 self.properties.union(set(['normalVolume'])))

        # test normalized mean
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(identifier='exp_1', 
                                      property='normalMean'),
            [0, 5, -2, -1, 0, 1, 2, 7, 1])

    def test_read_layers_on_columns(self):
        """
        Tests read(mode='layers_on_columns')
        """

        # make catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        cat_dir = os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=cat_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        catal.makeGroups(feature='category')

        # read all
        cleft_lay = CleftRegions.read(
            files=catal.results_obj, catalog=catal, mode='layers')
        cleft_layoncol = CleftRegions.read(
            files=catal.results_obj, catalog=catal, mode='layers_on_columns',
            reference=cleft_lay)

        # test meta-properties
        for category in ['first', 'second']:
            np_test.assert_equal(cleft_layoncol[category].indexed, self.indexed)
            np_test.assert_equal(
                cleft_layoncol[category].properties, 
                self.properties_layers_on_columns)

        # ids
        np_test.assert_equal(cleft_layoncol.first.cleftIds, 
                             [list(range(3,8)), list(range(3,8))])

        # data
        np_test.assert_equal(
            cleft_lay.second.getValue(identifier='exp_1', property='mean'),
            self.mean_0)

    def testNormalizeByMean(self):
        """
        Tests normalizeByMean() for layers, layers on columns and columns
        """

        # make catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        cat_dir = os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=cat_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        catal.makeGroups(feature='category')

        # make cleft layers
        cleft_lay = CleftRegions.read(files=catal.results_obj, catalog=catal,
                                      mode='layers')

        # absolute, specified ids
        cleft_lay.normalizeByMean(name='mean', normalName='normalized',
                                  ids=[3,4,5], mode='absolute')
        np_test.assert_equal('normalized' in cleft_lay.first.properties, True)
        np_test.assert_equal('normalized' in cleft_lay.second.properties, True)
        np_test.assert_equal('normMean' in cleft_lay.first.properties, False)
        np_test.assert_equal('normMean' in cleft_lay.second.properties, False)
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(identifier='exp_1', 
                                      property='normalized'),
            [1, 6, -1, 0, 1, 2, 3, 8, 2])
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(identifier='exp_4', 
                                      property='normalized'),
            [1, 6, -1, 0, 1, 2, 3, 8, 2])

        # absolute, specified ids, reference
        cleft_lay.normalizeByMean(
            name='mean', normalName='normalized',
            ids=[3,4,5], mode='absolute', reference=cleft_lay)
        np_test.assert_equal('normalized' in cleft_lay.first.properties, True)
        np_test.assert_equal('normalized' in cleft_lay.second.properties, True)
        np_test.assert_equal('normMean' in cleft_lay.first.properties, False)
        np_test.assert_equal('normMean' in cleft_lay.second.properties, False)
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(identifier='exp_1', 
                                      property='normalized'),
            [1, 6, -1, 0, 1, 2, 3, 8, 2])
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(identifier='exp_4', 
                                      property='normalized'),
            [1, 6, -1, 0, 1, 2, 3, 8, 2])

        # relative, all ids
        cleft_lay.normalizeByMean(name='mean', normalName='normalized',
                                  mode='relative')
        np_test.assert_equal('normalized' in cleft_lay.first.properties, True)
        np_test.assert_equal('normalized' in cleft_lay.second.properties, True)
        np_test.assert_equal('normMean' in cleft_lay.first.properties, False)
        np_test.assert_equal('normMean' in cleft_lay.second.properties, False)
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(
                identifier='exp_1', property='normalized'),
            (self.mean_0 - self.mean_0.mean()) / self.mean_0.mean())
        np_test.assert_almost_equal(
            cleft_lay.first.getValue(
                identifier='exp_2', property='normalized'),
            (self.mean_0 - self.mean_0.mean()) / (self.mean_0.mean() + 1))

        # relative, all ids, reference
        cleft_lay.normalizeByMean(name='mean', normalName='normalized',
                                  mode='relative', reference=cleft_lay)
        np_test.assert_equal('normalized' in cleft_lay.first.properties, True)
        np_test.assert_equal('normalized' in cleft_lay.second.properties, True)
        np_test.assert_equal('normMean' in cleft_lay.first.properties, False)
        np_test.assert_equal('normMean' in cleft_lay.second.properties, False)
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(
                identifier='exp_1', property='normalized'),
            (self.mean_0 - self.mean_0.mean()) / self.mean_0.mean())
        np_test.assert_almost_equal(
            cleft_lay.first.getValue(
                identifier='exp_2', property='normalized'),
            (self.mean_0 - self.mean_0.mean()) / (self.mean_0.mean() + 1))

        # 0to1, regions
        cleft_lay.normalizeByMean(name='mean', normalName='normal0to1',
                                  region=['bound', 'cleft'], mode='0to1')
        np_test.assert_equal('normal0to1' in cleft_lay.first.properties, True)
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(
                identifier='exp_1', property='normal0to1'),
            (common.cleft_layers_density_mean - 33/4.) / (5 - 33/4.))
        np_test.assert_almost_equal(
            cleft_lay.first.getValue(
                identifier='exp_2', property='normal0to1'),
            (common.cleft_layers_density_mean - 33/4.) / (5 - 33/4.))

        # 0to1, regions, with reference
        cleft_lay.normalizeByMean(
            name='mean', normalName='normal0to1', region=['bound', 'cleft'], 
            mode='0to1', reference=cleft_lay)
        np_test.assert_equal('normal0to1' in cleft_lay.first.properties, True)
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(
                identifier='exp_1', property='normal0to1'),
            (common.cleft_layers_density_mean - 33/4.) / (5 - 33/4.))
        np_test.assert_almost_equal(
            cleft_lay.first.getValue(
                identifier='exp_2', property='normal0to1'),
            (common.cleft_layers_density_mean - 33/4.) / (5 - 33/4.))

        # 0to1, ids
        cleft_lay.normalizeByMean(name='mean', normalName='normal0to1',
                                  ids=[[1,2,8,9], [3,4,5,6,7]], mode='0to1')
        np_test.assert_equal('normal0to1' in cleft_lay.first.properties, True)
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(
                identifier='exp_1', property='normal0to1'),
            (common.cleft_layers_density_mean - 33/4.) / (5 - 33/4.))

        # 0to1 layers on columns, simple
        cleft_layoncol = CleftRegions.read(
            files=catal.results_obj, catalog=catal, mode='layers_on_columns',
            reference=cleft_lay)
        np_test.assert_almost_equal(
            cleft_layoncol.second.getValue(identifier='exp_1', 
                                      property='normalMean'),
            (common.cleft_layers_density_mean - 33/4.) / (5 - 33/4.))


        # make another layers with simple values
        cleft_lay_2 = CleftRegions.read(files=catal.results_obj, catalog=catal,
                                      mode='layers')
        cleft_lay_2.second.setValue(
            identifier='exp_1', property='mean', 
            value=numpy.array([0., 0, 10, 10, 10, 10, 10, 0, 0.]))
        cleft_lay_2.first.setValue(
            identifier='exp_2', property='mean', 
            value=numpy.array([0., 0, 10, 10, 10, 10, 10, 0, 0.]))
        cleft_lay_2.second.setValue(
            identifier='exp_4', property='mean', 
            value=numpy.array([0., 0, 10, 10, 10, 10, 10, 0, 0.]))

        # 0to1, layers on columns, with reference 0
        cleft_layoncol.normalizeByMean(
            name='mean', normalName='nor', ids=[[1,2,8,9], [3,4,5,6,7]], 
            mode='0to1', reference=[cleft_lay_2, None])
        np_test.assert_almost_equal(
            cleft_layoncol.second.getValue(
                identifier='exp_1', property='nor'),
            (common.cleft_layers_density_mean - 0.) / (5.))
        np_test.assert_almost_equal(
            cleft_layoncol.first.getValue(
                identifier='exp_2', property='nor'),
            (common.cleft_layers_density_mean + 1 - 0.) / (6.))

        # 0to1, layers on columns, with references 0 and 1
        cleft_layoncol.normalizeByMean(
            name='mean', normalName='nor', ids=[[1,2,8,9], [3,4,5,6,7]], 
            mode='0to1', reference=[cleft_lay_2, cleft_lay_2])
        np_test.assert_almost_equal(
            cleft_layoncol.second.getValue(identifier='exp_1', 
                                      property='nor'),
            (common.cleft_layers_density_mean - 0.) / (10.))
      
        # 0to1, layers on columns, with references 0 and 1, shortcut
        cleft_layoncol.normalizeByMean(
            name='mean', normalName='nor', ids=[[1,2,8,9], [3,4,5,6,7]], 
            mode='0to1', reference=cleft_lay_2)
        np_test.assert_almost_equal(
            cleft_layoncol.second.getValue(identifier='exp_1', 
                                      property='nor'),
            (common.cleft_layers_density_mean - 0.) / (10.))
      
        # absolute, columns, with references 0 and 1
        cleft_col = CleftRegions.read(
            files=catal.cleft_columns_obj, catalog=catal,
            mode='columns', reference=cleft_lay_2)
        cleft_col.normalizeByMean(
            name='mean', normalName='nor', ids=[1,2,8,9], 
            mode='absolute', reference=cleft_lay_2)
        np_test.assert_almost_equal(
            cleft_col.second.getValue(
                identifier='exp_1', property='nor'),
            common.cleft_columns_density_mean)
 
       # 0to1, columns, with references 0 and 1
        cleft_col = CleftRegions.read(
            files=catal.cleft_columns_obj, catalog=catal,
            mode='columns', reference=cleft_lay_2)
        cleft_col.normalizeByMean(
            name='mean', normalName='nor', ids=[[1,2,8,9], [3,4,5,6,7]], 
            mode='0to1', reference=[cleft_lay_2, cleft_lay_2])
        np_test.assert_almost_equal(
            cleft_col.second.getValue(identifier='exp_1', 
                                      property='nor'),
            (common.cleft_columns_density_mean - 0.) / (10.))
        np_test.assert_almost_equal(
            cleft_col.second.getValue(identifier='exp_4', 
                                      property='nor'),
            (common.cleft_columns_density_mean + 3 - 0.) / (10.))
      
    def testGetRelative(self):
        """
        Tests getRelative()
        """

        # make catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        cat_dir = os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=cat_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        catal.makeGroups(feature='category')

        # read layers
        cleft_lay = CleftRegions.read(
            files=catal.results_obj, catalog=catal, mode='layers')
       
        # test regions
        cleft_lay.getRelative(fraction=0.5, new='half', name='mean',
                              region=['bound', 'cleft'])
        np_test.assert_almost_equal( 
            cleft_lay.second.getValue(identifier='exp_1', property='half'), 
            (33/4. + 5) / 2)

        cleft_lay.getRelative(fraction=0.3, new='point_three', name='mean',
                              region=['bound', 'cleft'])
        np_test.assert_almost_equal( 
            cleft_lay.first.getValue(identifier='exp_2', 
                                      property='point_three'), 
            37/4. + (6 - 37/4.) * 0.3)

        # test regions with volume
        cleft_lay.getRelative(fraction=0.5, new='half', name='mean',
                              region=['bound', 'cleft'], weight='volume')
        np_test.assert_almost_equal( 
            cleft_lay.second.getValue(identifier='exp_1', property='half'), 
            (33/4. + 5) / 2)

        cleft_lay.second.setValue(identifier='exp_1', property='volumeX',
                                  value=numpy.array([6,6,6,6,6,10,10,6,6]))
        cleft_lay.second.setValue(identifier='exp_4', property='volumeX',
                                  value=numpy.array([6,6,6,6,6,10,10,6,6]))
        cleft_lay.getRelative(
            fraction=0.5, new='half', name='mean', region=['bound', 'cleft'], 
            weight='volumeX', categories=['second'])
        np_test.assert_almost_equal( 
            cleft_lay.second.getValue(identifier='exp_1', property='half'), 
            (33/4. + (130 + 12*6)/38.) / 2)

        # test ids
        cleft_lay.getRelative(fraction=0.5, new='half',
                              ids=[[1,2,8,9], [3,4,5,6,7]])
        np_test.assert_almost_equal( 
            cleft_lay.second.getValue(identifier='exp_1', property='half'), 
            (33/4. + 5) / 2)

    def testGetBoundarySurfaces(self):
        """
        Tests getBoundarySurfaces()
        """

        # make catalog
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        cat_dir = os.path.join(curr_dir, common.rel_catalogs_dir)
        catal = Catalog() 
        catal.read(dir=cat_dir, catalog=r'catalog_[0-9]*\.', 
                   type='distributed')
        catal.makeGroups(feature='category')

        # read all
        cleft_lay = CleftRegions.read(files=catal.results_obj, catalog=catal,
                                     mode='layers')

        #  test simple
        cleft_lay.getBoundarySurfaces(names=['surfaceBound1', 'surfaceBound2'],
                                      surface='volume')
        np_test.assert_almost_equal(
            cleft_lay.first.getValue(identifier='exp_2', 
                                     property='surfaceBound1'), 6)
        np_test.assert_almost_equal(
            cleft_lay.first.getValue(identifier='exp_2', 
                                     property='surfaceBound2'), 6)
        np_test.assert_almost_equal(
            cleft_lay.first.getValue(identifier='exp_3', 
                                     property='surfaceBound1'), 6)
        np_test.assert_almost_equal(
            cleft_lay.second.getValue(identifier='exp_1', 
                                     property='surfaceBound1'), 6)

        #  test factor = 10
        cleft_lay.getBoundarySurfaces(names=['surfaceBound1', 'surfaceBound2'],
                                      factor=10, surface='volume')
        np_test.assert_almost_equal(
            cleft_lay.first.getValue(identifier='exp_3', 
                                     property='surfaceBound2'), 60)

    def tearDown(self):
        """
        Remove files generated here
        """
        for path in self.tmp_paths:
            try:
                os.remove(path)
                #pass
            except OSError:
                print("Tests fine but could not remove " + str(path))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCleftRegions)
    unittest.TextTestRunner(verbosity=2).run(suite)
