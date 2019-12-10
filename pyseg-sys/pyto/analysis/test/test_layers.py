"""

Tests module analysis.layers.

# Author: Vladan Lucic
# $Id: test_layers.py 1151 2015-05-26 08:52:30Z vladan $
"""

__version__ = "$Revision: 1151 $"

from copy import copy, deepcopy
import pickle
import os.path
import sys
import unittest

import numpy
import numpy.testing as np_test 
import scipy

import pyto
from pyto.analysis.layers import Layers


class TestLayers(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """

        # set layer files and adjust paths
        dir_, base = os.path.split(__file__)
        layer_files = {
            'rim_wt' : {'77_4': 'segmentations/layers_77-4.dat',
                        '78_3': 'segmentations/layers_78-3.dat'},
            'rim_altered' : {'75_4' : 'segmentations/layers_75-4.dat'}}
        for categ in layer_files:
            for ident, name in layer_files[categ].items():
               layer_files[categ][ident] = os.path.join(dir_, name)
        self.layer_files = layer_files

        # set pixel size
        self.pixel_size = {
            'rim_wt' : {'77_4' : 2.644, '78_3' : 2.644},
            'rim_altered' : {'75_4' : 2.644}}

        # set catalog
        catalog = pyto.analysis.Catalog()
        catalog._db = {
            'category' : {'77_4' : 'rim_wt', '78_3' : 'rim_wt', 
                          '75_4' : 'rim_altered'},
            'tether_files' : {'77_4': 'segmentations/layers_77-4.dat',
                              '78_3': 'segmentations/layers_78-3.dat',
                              '75_4' : 'segmentations/layers_75-4.dat'},
            'pixel_size' : {'77_4' : 2.644, '78_3' : 2.644, '75_4' : 2.644},
            'operator' : {'77_4' : 'emerson', '78_3' : 'lake', 
                          '75_4' : 'palmer'}
            }
        for ident, name in catalog._db['tether_files'].items():
               catalog._db['tether_files'][ident] = os.path.join(dir_, name)
        catalog.makeGroups()
        self.catalog = catalog
        
    def testReadNoCatalogs(self):
        """
        Tests read() with catalogs
        """

        # read
        layer = Layers.read(files=self.layer_files, 
                            pixel=self.pixel_size)
        pixel_size = self.pixel_size['rim_wt']['77_4']        

        # test general
        np_test.assert_equal(layer.rim_wt.identifiers, ['77_4', '78_3'])
        np_test.assert_equal(layer.rim_altered.identifiers, ['75_4'])

        # test 77_4
        np_test.assert_equal(
            layer.rim_wt.getValue(identifier='77_4', property='ids'), 
            range(1, 171))
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='77_4', property='volume', ids=17),
            11371, decimal=2)
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='77_4', property='volume_nm', 
                                  ids=17),
            11371 * pixel_size**3, decimal=0)
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='77_4', property='surface_nm', 
                                  ids=17),
            11371 * pixel_size**2, decimal=0)

        # test 78_3
        np_test.assert_equal(
            layer.rim_wt.getValue(identifier='78_3', property='ids'), 
            range(1,161))
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='78_3', property='occupancy', 
                               ids=17), 
            0.01847, decimal=5)
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='78_3', property='occupied', 
                               ids=17), 
            200.0, decimal=1)

        # test 75_4
        np_test.assert_equal(
            layer.rim_altered.getValue(identifier='75_4', property='ids'), 
            range(1, 151))
        np_test.assert_almost_equal(
            layer.rim_altered.getValue(identifier='75_4', property='distance', 
                                    ids=27),
            27, decimal=1)
        np_test.assert_almost_equal(
            layer.rim_altered.getValue(identifier='75_4', 
                                       property='distance_nm', ids=27),
            71.4, decimal=1)

    def testReadCatalogs(self):
        """
        Tests read() with catalogs
        """

        # read
        layer = Layers.read(files=self.layer_files, catalog=self.catalog)
        pixel_size = self.catalog.pixel_size['rim_wt']['77_4']

        # test general
        np_test.assert_equal(layer.rim_wt.identifiers, ['77_4', '78_3'])
        np_test.assert_equal(layer.rim_altered.identifiers, ['75_4'])

        # test 77_4
        np_test.assert_equal(
            layer.rim_wt.getValue(identifier='77_4', property='ids'), 
            range(1, 171))
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='77_4', property='volume', ids=17),
            11371, decimal=2)
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='77_4', property='volume_nm', 
                                  ids=17),
            11371 * pixel_size**3, decimal=0)
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='77_4', property='surface_nm', 
                                  ids=17),
            11371 * pixel_size**2, decimal=0)

        # test 78_3
        np_test.assert_equal(
            layer.rim_wt.getValue(identifier='78_3', property='ids'), 
            range(1,161))
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='78_3', property='occupancy', 
                               ids=17), 
            0.01847, decimal=5)
        np_test.assert_almost_equal(
            layer.rim_wt.getValue(identifier='78_3', property='occupied', 
                               ids=17), 
            200.0, decimal=1)

        # test 75_4
        np_test.assert_equal(
            layer.rim_altered.getValue(identifier='75_4', property='ids'), 
            range(1, 151))
        np_test.assert_almost_equal(
            layer.rim_altered.getValue(identifier='75_4', property='distance', 
                                    ids=27),
            27, decimal=1)
        np_test.assert_almost_equal(
            layer.rim_altered.getValue(identifier='75_4', 
                                       property='distance_nm', ids=27),
            71.4, decimal=1)

    def testReadOrder(self):
        """
        Tests read() with specified order 
        """

        # make order and read
        order = {'rim_wt' : ['78_3', '77_4'], 'rim_altered' : ['75_4']}
        layer = Layers.read(files=self.layer_files, catalog=self.catalog,
                            order=order)

        # test general
        np_test.assert_equal(layer.rim_wt.identifiers, ['78_3', '77_4'])
        np_test.assert_equal(layer.rim_altered.identifiers, ['75_4'])

    def testRebin(self):
        """
        Tests rebin()
        """

        # read
        layer = Layers.read(files=self.layer_files, catalog=self.catalog)
        pixel_size = self.catalog.pixel_size['rim_wt']['77_4']

        # without pixel_size
        layer_bin = layer.rebin(bins=[0,2,4,6])
        np_test.assert_equal(
            layer_bin.rim_wt.getValue(identifier='77_4', property='volume'), 
            [7557 + 6205, 5345 + 6647, 6989 + 6549])
        np_test.assert_equal(
            layer_bin.rim_wt.getValue(identifier='77_4', property='volume_nm'), 
            numpy.array([7557+6205, 5345+6647, 6989+6549]) * pixel_size**3)
        np_test.assert_equal(
            layer_bin.rim_wt.operator,
            layer.rim_wt.operator)

        # pixel_size
        pixel = {
            'rim_wt' : {'77_4' : 0.5, '78_3' : 0.5},
            'rim_altered' : {'75_4' : 0.5}}
        layer_bin = layer.rebin(bins=[0,2,4], pixel=pixel)
        np_test.assert_equal(
            layer_bin.rim_wt.getValue(identifier='77_4', property='volume'), 
            [7557 + 6205 + 5345 + 6647, 6989 + 6549 + 6940 + 7106])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLayers)
    unittest.TextTestRunner(verbosity=2).run(suite)
