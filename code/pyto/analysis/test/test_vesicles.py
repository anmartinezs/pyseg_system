"""

Tests module analysis.vesicles.

Note: The pickles used here (segmentations/*.pkl) are real, except that
large (image) arrays that are not needed in tests were removed in
order to fit the GutHub size limit. 

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"

from copy import copy, deepcopy
import pickle
import os.path
import sys
import unittest

import numpy
import numpy.testing as np_test 
import scipy

import pyto
from pyto.analysis.vesicles import Vesicles
from pyto.analysis.clusters import Clusters
from pyto.analysis.observations import Observations


class TestVesicles(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """

        # set sv files and adjust paths
        dir_, base = os.path.split(__file__)
        sv_files = {
            'rim_wt' : {'77_4': 'segmentations/sv_77-4_vesicles.pkl',
                        '78_3': 'segmentations/sv_78-3_vesicles.pkl'},
            'rim_altered' : {'75_4' : 'segmentations/sv_75-4_vesicles.pkl'}}
        for categ in sv_files:
            for ident, name in list(sv_files[categ].items()):
               sv_files[categ][ident] = os.path.join(dir_, name)
        self.sv_files = sv_files

        # set membrane files and adjust paths
        mem_files = {          
            'rim_wt' : {'77_4': 'segmentations/sv_77-4_mem.pkl',
                        '78_3': 'segmentations/sv_78-3_mem.pkl'},
            'rim_altered' : {'75_4' : 'segmentations/sv_75-4_mem.pkl'}}
        for categ in mem_files:
            for ident, name in list(mem_files[categ].items()):
               mem_files[categ][ident] = os.path.join(dir_, name)
        self.mem_files = mem_files

        # set lumen files and adjust paths
        lum_files = {
            'rim_wt' : {'77_4': 'segmentations/sv_77-4_lum.pkl',
                        '78_3': 'segmentations/sv_78-3_lum.pkl'},
            'rim_altered' : {'75_4' : 'segmentations/sv_75-4_lum.pkl'}}
        for categ in lum_files:
            for ident, name in list(lum_files[categ].items()):
               lum_files[categ][ident] = os.path.join(dir_, name)
        self.lum_files = lum_files

        # set catalog for tethers and connectors
        dir_, base = os.path.split(__file__)
        catalog = pyto.analysis.Catalog()
        catalog._db = {
            'category' : {'77_4' : 'rim_wt', '78_3' : 'rim_wt', 
                          '75_4' : 'rim_altered'},
            'tether_files' : {'77_4': 'segmentations/conn_77-4_new_AZ.pkl',
                              '78_3': 'segmentations/conn_78-3_new_AZ.pkl',
                              '75_4' : 'segmentations/conn_75-4_new_AZ.pkl'},
            'connector_files' : {'77_4': 'segmentations/conn_77-4_new_rest.pkl',
                              '78_3': 'segmentations/conn_78-3_new_rest.pkl',
                              '75_4' : 'segmentations/conn_75-4_new_rest.pkl'},
            'layer_files' : {'77_4': 'segmentations/layers_77-4.dat',
                              '78_3': 'segmentations/layers_78-3.dat',
                              '75_4' : 'segmentations/layers_75-4.dat'},
            'pixel_size' : {'77_4' : 2.644, '78_3' : 2.644, '75_4' : 2.644},
            'operator' : {'77_4' : 'emerson', '78_3' : 'lake', 
                          '75_4' : 'palmer'}
            }
        for ident, name in list(catalog._db['tether_files'].items()):
               catalog._db['tether_files'][ident] = os.path.join(dir_, name)
        for ident, name in list(catalog._db['connector_files'].items()):
               catalog._db['connector_files'][ident] = os.path.join(dir_, name)
        for ident, name in list(catalog._db['layer_files'].items()):
               catalog._db['layer_files'][ident] = os.path.join(dir_, name)
        catalog.makeGroups()
        self.catalog = catalog

        # read tethers and connectors 
        from pyto.analysis.connections import Connections
        self.tether = Connections.read(files=catalog.tether_files, 
                                       mode='connectors', catalog=catalog)
        self.connector = Connections.read(
            files=catalog.connector_files, mode='connectors', catalog=catalog,
            order=self.tether)

        # read layers
        from pyto.analysis.layers import Layers
        self.layer = Layers.read(
            files=catalog.layer_files, catalog=catalog, order=self.tether)

        # read vesicles
        self.sv = Vesicles.read(files=self.sv_files, catalog=self.catalog,
                                membrane=self.mem_files, lumen=self.mem_files)

    def testReadNoCatalogs(self):
        """
        Tests read() without catalogs
        """

        # read
        sv = Vesicles.read(files=self.sv_files, pixel=self.catalog.pixel_size, 
                           membrane=self.mem_files, lumen=self.lum_files)

        # test general
        np_test.assert_equal(sv.rim_wt.identifiers, ['77_4', '78_3'])
        np_test.assert_equal(sv.rim_altered.identifiers, ['75_4'])

        # test 77_4
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='77_4', property='ids'), 
            list(range(2, 144)))
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='density', ids=7),
            -0.02, decimal=2)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='membrane_density', 
                               ids=7),
            -0.04, decimal=2)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='lumen_density', 
                               ids=7),
            0.00, decimal=2)

        # test 78_3
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='78_3', property='ids'), 
            list(range(2,81)))
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='78_3', property='meanDistance', 
                               ids=7), 
            50.4, decimal=1)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='78_3', property='minDistance', 
                               ids=7),
            44.0, decimal=1)

        # test 75_4
        np_test.assert_equal(
            sv.rim_altered.getValue(identifier='75_4', property='ids'), 
            list(range(2, 133)))
        np_test.assert_almost_equal(
            sv.rim_altered.getValue(identifier='75_4', property='radius', 
                                    ids=7),
            6.3, decimal=1)
        np_test.assert_almost_equal(
            sv.rim_altered.getValue(identifier='75_4', property='radius_nm', 
                                    ids=7),
            16.5, decimal=1)

    def testReadCatalogs(self):
        """
        Tests read() with catalogs
        """

        # read
        sv = Vesicles.read(files=self.sv_files, catalog=self.catalog, 
                           membrane=self.mem_files, lumen=self.lum_files)

        # test general
        np_test.assert_equal(sv.rim_wt.identifiers, ['77_4', '78_3'])
        np_test.assert_equal(sv.rim_altered.identifiers, ['75_4'])

        # test 77_4
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='77_4', property='ids'), 
            list(range(2, 144)))
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='density', ids=7),
            -0.02, decimal=2)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='membrane_density', 
                               ids=7),
            -0.04, decimal=2)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='lumen_density', 
                               ids=7),
            0.00, decimal=2)

        # test 78_3
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='78_3', property='ids'), 
            list(range(2,81)))
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='78_3', property='meanDistance', 
                               ids=7), 
            50.4, decimal=1)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='78_3', property='minDistance', 
                               ids=7),
            44.0, decimal=1)

        # test 75_4
        np_test.assert_equal(
            sv.rim_altered.getValue(identifier='75_4', property='ids'), 
            list(range(2, 133)))
        np_test.assert_almost_equal(
            sv.rim_altered.getValue(identifier='75_4', property='radius', 
                                    ids=7),
            6.3, decimal=1)
        np_test.assert_almost_equal(
            sv.rim_altered.getValue(identifier='75_4', property='radius_nm', 
                                    ids=7),
            16.5, decimal=1)

    def testRead_order(self):
        """
        Tests read() with specified order
        """

        # make order and read
        order = {'rim_wt' : ['78_3', '77_4'], 'rim_altered' : ['75_4']}
        sv = Vesicles.read(
            files=self.sv_files, catalog=self.catalog, order=order,
            membrane=self.mem_files, lumen=self.lum_files)
 
        # test general
        np_test.assert_equal(sv.rim_wt.identifiers, ['78_3', '77_4'])
        np_test.assert_equal(sv.rim_altered.identifiers, ['75_4'])

    def testAddLinked(self):
        """
        Tests addLinked() and getNLinked()
        """

        # read svs
        sv = Vesicles.read(files=self.sv_files, catalog=self.catalog, 
                           membrane=self.mem_files, lumen=self.lum_files)

        # read connectors
        conn_files = {
            'rim_wt' : {'77_4': 'segmentations/conn_77-4_new_rest.pkl',
                        '78_3': 'segmentations/conn_78-3_new_rest.pkl'},
            'rim_altered' : {'75_4' : 'segmentations/conn_75-4_new_rest.pkl'}}

        # calculate linked and N linked
        sv.addLinked(files=self.catalog.connector_files)
        sv.getNLinked()

        # test 77_4
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='77_4', property='linked', ids=3),
            [4, 30])
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='77_4', property='n_linked', ids=3), 
            2)
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='77_4', property='linked', ids=5),
            [4, 22, 88])
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='77_4', property='n_linked', ids=5), 
            3)

    def testGetMeanConnectionLength(self):
        """
        Tests getMeanConnectionLength()
        """

        # read svs
        sv = Vesicles.read(files=self.sv_files, catalog=self.catalog,
                           membrane=self.mem_files, lumen=self.mem_files)

        # calculate mean tether lengths
        sv.getMeanConnectionLength(conn=self.tether, name='mean_tether_nm')

        # test 77_4
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='mean_tether_nm', 
                               ids=3),
            5.54 * self.catalog.pixel_size['rim_wt']['77_4'], 
            decimal=1)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='mean_tether_nm', 
                               ids=5),
            4.88 * self.catalog.pixel_size['rim_wt']['77_4'], 
            decimal=1)

    def testGetNVesicles(self):
        """
        Tests getNVesicles()
        """

        # read svs
        sv = Vesicles.read(files=self.sv_files, catalog=self.catalog,
                           membrane=self.mem_files, lumen=self.mem_files)
        pixel_size = self.catalog.pixel_size['rim_wt']['77_4']

        # n vesicles 
        sv.getNVesicles(name='n_ves')
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='77_4', property='n_ves'), 142)
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='78_3', property='n_ves'), 79)
        np_test.assert_equal(
            sv.rim_altered.getValue(identifier='75_4', property='n_ves'), 131)

        # n vesicles per unit layer area
        sv.getNVesicles(name='n_area', layer=self.layer)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='n_area'), 
            142 / (7557 * pixel_size**2 * 1e-6), 
            decimal=1)

        # n vesicles per unit layer area, layer_factor=0.001
        sv.getNVesicles(name='n_area', layer=self.layer, layer_factor=0.001)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='n_area'), 
            142 / (7557 * pixel_size**2 * 0.001), 
            decimal=4)
 
       # unit layer area in nm^2
        sv.getNVesicles(name='area_nm', layer=self.layer, inverse=True, 
                        fixed=1, layer_factor=1.)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='area_nm'), 
            7557 * pixel_size**2, 
            decimal=1)

       # unit layer area in nm^3
        sv.getNVesicles(name='area_um', layer=self.layer, inverse=True, 
                        fixed=1, layer_factor=1.e-6)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='area_um'), 
            7557 * pixel_size**2 * 1.e-6, 
            decimal=1)

    def testGetN(self):
        """
        Tests getN()
        """

        # read svs
        sv = Vesicles.read(files=self.sv_files, catalog=self.catalog,
                           membrane=self.mem_files, lumen=self.mem_files)
        pixel_size = self.catalog.pixel_size['rim_wt']['77_4']

        # n vesicles 
        sv.getNVesicles(name='n_ves')
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='77_4', property='n_ves'), 142)
        np_test.assert_equal(
            sv.rim_wt.getValue(identifier='78_3', property='n_ves'), 79)
        np_test.assert_equal(
            sv.rim_altered.getValue(identifier='75_4', property='n_ves'), 131)

        # n vesicles per unit layer area
        sv.getNVesicles(name='n_area', layer=self.layer)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='n_area'), 
            142 / (7557 * pixel_size**2 * 1e-6), 
            decimal=1)

        # n vesicles per unit layer area, layer_factor=0.001
        sv.getNVesicles(name='n_area', layer=self.layer, layer_factor=0.001)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='n_area'), 
            142 / (7557 * pixel_size**2 * 0.001), 
            decimal=4)
 
       # unit layer area in nm^2
        sv.getNVesicles(name='area_nm', layer=self.layer, inverse=True, 
                        fixed=1, layer_factor=1.)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='area_nm'), 
            7557 * pixel_size**2, 
            decimal=1)

       # unit layer area in nm^3
        sv.getNVesicles(name='area_um', layer=self.layer, inverse=True, 
                        fixed=1, layer_factor=1.e-6)
        np_test.assert_almost_equal(
            sv.rim_wt.getValue(identifier='77_4', property='area_um'), 
            7557 * pixel_size**2 * 1.e-6, 
            decimal=1)

    def testGetNTethers(self):
        """
        Tests getNTethers()
        """

        # get N tethers
        self.sv.getNTethers(tether=self.tether)

        # test 78_3
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='78_3', name='n_tether', ids=2),
            3)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='78_3', name='n_tether', ids=31),
            4)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='78_3', name='n_tether', ids=35),
            1)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='78_3', name='n_tether', ids=3),
            0)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='78_3', name='n_tether', ids=4),
            0)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='78_3', name='n_tether', ids=80),
            0)

        # test 77_4
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=3),
            1)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=4),
            1)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=5),
            3)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=6),
            3)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=22),
            4)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=30),
            2)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=45),
            2)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=88),
            1)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=13),
            0)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=29),
            0)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='n_tether', ids=89),
            0)
       
        # test 75_4
        np_test.assert_equal(
            self.sv.rim_altered.getValue(identifier='75_4', 
                                         name='n_tether', ids=21), 2)
        np_test.assert_equal(
            self.sv.rim_altered.getValue(identifier='75_4', 
                                         name='n_tether', ids=73), 1)
        np_test.assert_equal(
            self.sv.rim_altered.getValue(identifier='75_4', 
                                         name='n_tether', ids=103), 1)
        np_test.assert_equal(
            self.sv.rim_altered.getValue(identifier='75_4', 
                                         name='n_tether', ids=20), 0)
        np_test.assert_equal(
            self.sv.rim_altered.getValue(identifier='75_4', 
                                         name='n_tether', ids=22), 0)
        np_test.assert_equal(
            self.sv.rim_altered.getValue(identifier='75_4', 
                                         name='n_tether', ids=102), 0)
        
    def testExtractTethered(self):
        """
        Tests extractTethered()
        """

        # get N tethers
        self.sv.getNTethers(tether=self.tether)

        # extract tethered and non-tethered
        teth_sv, non_teth_sv = self.sv.extractTethered(other=True)

        # test 78_3
        desired_teth = [2, 31, 35]
        np_test.assert_equal(
            teth_sv.rim_wt.getValue(identifier='78_3', name='ids'), 
            desired_teth)
        desired_non_teth = list(set(range(2,81)).difference(desired_teth))
        np_test.assert_equal(
            non_teth_sv.rim_wt.getValue(identifier='78_3', name='ids'),
            desired_non_teth)

        # test 77_4
        desired_teth = [3, 4, 5, 6, 22, 30, 45, 88]
        np_test.assert_equal(
            teth_sv.rim_wt.getValue(identifier='77_4', name='ids'), 
            desired_teth)
        desired_non_teth = list(set(range(2,144)).difference(desired_teth))
        np_test.assert_equal(
            non_teth_sv.rim_wt.getValue(identifier='77_4', name='ids'),
            desired_non_teth)

        # test 75_4
        desired_teth = [21, 73, 103]
        np_test.assert_equal(
            teth_sv.rim_altered.getValue(identifier='75_4', name='ids'), 
            desired_teth)
        desired_non_teth = list(set(range(2,133)).difference(desired_teth))
        np_test.assert_equal(
            non_teth_sv.rim_altered.getValue(identifier='75_4', name='ids'),
            desired_non_teth)

    def testGetConnectivityDistance(self):
        """
        Tests getConnectionDistance()
        """

        # extract tethered and non-tethered
        self.sv.getNTethers(tether=self.tether)
        teth_sv, non_teth_sv = self.sv.extractTethered(other=True)

        # linked not set, test error
        #np_test.assert_raises(
        #    ValueError, self.sv.getConnectivityDistance, 
        #    {'initial':teth_sv})

        # calculate linked and N linked
        self.sv.addLinked(files=self.catalog.connector_files)

        # calculate distance
        self.sv.getConnectivityDistance(initial=teth_sv, distance=1)
 
        # test 78_3
        np_test.assert_equal('conn_distance' in self.sv.rim_wt.properties, True)
        np_test.assert_equal('conn_distance' in self.sv.rim_wt.indexed, True)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='78_3', name='conn_distance',
                                    ids=[2, 31, 35]),
            [1,1,1])
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='78_3', name='conn_distance',
                                    ids=[3, 32, 36]),
            [-1,-1,-1])
        
        # test 77_4
        np_test.assert_equal('conn_distance' in self.sv.rim_wt.properties, True)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=[3, 4, 5, 6, 22, 30, 45, 88]),
            8*[1])
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=[32, 31, 34]),
            3*[2])
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=[41, 35]),
            2*[3])
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=[71, 105, 71]),
            3*[4])
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=[114, 23, 67]),
            3*[5])
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=[10, 111]),
            2*[6])
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=[107]),
            [7])
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=[26]),
            [8])
        np_test.assert_equal(
            max(self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=list(range(2,144)))), 
            8)
        np_test.assert_equal(
            self.sv.rim_wt.getValue(identifier='77_4', name='conn_distance',
                                    ids=[8, 55, 87, 11, 44, 12, 54]),
            7*[-1])
      
        # test 75_4
        np_test.assert_equal(
            'conn_distance' in self.sv.rim_altered.properties, True)
        np_test.assert_equal(
            self.sv.rim_altered.getValue(
                identifier='75_4', name='conn_distance', ids=[21, 73, 103]),
            3*[1])
        np_test.assert_equal(
            self.sv.rim_altered.getValue(
                identifier='75_4', name='conn_distance', ids=[58, 108]),
            2*[2])
        np_test.assert_equal(
            self.sv.rim_altered.getValue(
                identifier='75_4', name='conn_distance', ids=[107, 2]),
            2*[3])
        np_test.assert_equal(
            self.sv.rim_altered.getValue(
                identifier='75_4', name='conn_distance', ids=[10]),
            [4])
        np_test.assert_equal(
            max(self.sv.rim_altered.getValue(
                identifier='75_4', name='conn_distance', ids=list(range(2,133)))), 
            10)
        np_test.assert_equal(
            self.sv.rim_altered.getValue(
                identifier='75_4', name='conn_distance', ids=[4, 9]),
            2*[-1])

        # different initial distance and default
        self.sv.getConnectivityDistance(
            initial=teth_sv, distance=10, default=-2, name='conn_distance2')
        np_test.assert_equal(
            'conn_distance2' in self.sv.rim_altered.properties, True)
        np_test.assert_equal(
            'conn_distance2' in self.sv.rim_altered.indexed, True)
        np_test.assert_equal(
            self.sv.rim_altered.getValue(
                identifier='75_4', name='conn_distance2', ids=[21, 73, 103]),
            3*[10])
        np_test.assert_equal(
            self.sv.rim_altered.getValue(
                identifier='75_4', name='conn_distance2', ids=[58, 108]),
            2*[11])
        np_test.assert_equal(
            self.sv.rim_altered.getValue(
                identifier='75_4', name='conn_distance2', ids=[4, 9]),
            2*[-2])

    def testGetNearestNeighbor(self):
        """
        Tests getNearestNeighbor()
        """

        # setup
        sv = Vesicles()
        sv['group_a'] = Observations()
        clust = Clusters()
        clust['group_a'] = Observations()
        ids_1 = [1, 4, 5, 7]
        sv['group_a'].setValue(
            identifier='exp_1', name='ids', indexed=True, value=ids_1)
        clust['group_a'].setValue(
            identifier='exp_1', name='bound_ids', indexed=False, value=ids_1)
        clust['group_a'].setValue(
            identifier='exp_1', name='bound_dist', indexed=False, 
            value=[3., 2, 4, 3, 1, 2.5])
        clust['group_a'].setValue(
            identifier='exp_1', name='bound_dist_nm', indexed=False, 
             value=[41., 51, 71, 54, 74, 75])
        ids_2 = [1, 8, 5, 7, 4]
        sv['group_a'].setValue(
            identifier='exp_2', name='ids', indexed=True, value=ids_2)
        clust['group_a'].setValue(
            identifier='exp_2', name='bound_ids', indexed=False, 
            value=[4, 8, 3, 5])
        clust['group_a'].setValue(
            identifier='exp_2', name='bound_dist', indexed=False, 
            value=[48., 34, 45, 38, 58, 35])
        clust['group_a'].setValue(
            identifier='exp_2', name='bound_dist_nm', indexed=False, 
            value=[84., 43, 54, 83, 85, 53])

        # in pixels
        sv.getNearestNeighbor(cluster=clust, dist_name='bound_dist')
        near_dist = sv['group_a'].getValue(
            identifier='exp_1', name='nearest_distance')
        np_test.assert_equal(near_dist, [2, 1, 2, 1.])
        near_ids = sv['group_a'].getValue(
            identifier='exp_1', name='nearest_ids')
        np_test.assert_equal(near_ids, [5, 7, 1, 4])

        # different sv ids and clust bound_ids
        sv.getNearestNeighbor(cluster=clust, dist_name='bound_dist')
        near_dist = sv['group_a'].getValue(
            identifier='exp_2', name='nearest_distance')
        np_test.assert_equal(near_dist, [-1, 48, 45, -1, 45])
        near_ids = sv['group_a'].getValue(
            identifier='exp_2', name='nearest_ids')
        np_test.assert_equal(near_ids, [-1, 4, 4, -1, 5])

        # in pixels and in nm
        clust['group_a'].setValue(
            identifier='exp_1', name='bound_dist_nm', indexed=True, 
            value=2*numpy.array([3., 2, 4, 3, 1, 2.5]))
        sv.getNearestNeighbor(
            cluster=clust, dist_name='bound_dist', name='foo', default=-2)
        near_dist = sv['group_a'].getValue(
            identifier='exp_1', name='foo_distance')
        np_test.assert_equal(near_dist, [2, 1, 2, 1.])
        near_ids = sv['group_a'].getValue(
            identifier='exp_1', name='foo_ids')
        np_test.assert_equal(near_ids, [5, 7, 1, 4])
        near_dist_nm = sv['group_a'].getValue(
            identifier='exp_1', name='foo_distance_nm')
        np_test.assert_equal(near_dist_nm, 2*numpy.array([2, 1, 2, 1.]))
        near_dist = sv['group_a'].getValue(
            identifier='exp_2', name='foo_distance')
        np_test.assert_equal(near_dist, [-2, 48, 45, -2, 45])
        near_dist = sv['group_a'].getValue(
            identifier='exp_2', name='foo_distance_nm')
        np_test.assert_equal(near_dist, [-2, 84, 54, -2, 54])
        near_ids = sv['group_a'].getValue(
            identifier='exp_2', name='foo_ids')
        np_test.assert_equal(near_ids, [-2, 4, 4, -2, 5])

        # consistency check
        sv['group_b'] = Observations()
        sv['group_b'].setValue(
            identifier='exp_1', name='ids', indexed=True, value=ids_1)
        clust['group_b'] = Observations()
        clust['group_b'].setValue(
            identifier='exp_1', name='bound_ids', indexed=False, value=ids_1)
        clust['group_b'].setValue(
            identifier='exp_1', name='bound_dist', indexed=False, 
            value=[3., 2, 4, 3])
        # this way because assert_raises doesn't work for some reason
        caught = False
        try:
            sv.getNearestNeighbor(cluster=clust, dist_name='bound_dist')
        except ValueError:
            caught = True
        np_test.assert_equal(caught, True)
        #np_test.assert_raises(
        #    ValueError,
        #    sv.getNearestNeighbor,
        #    {'cluster':clust, 'dist_name':'bound_dist'})


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVesicles)
    unittest.TextTestRunner(verbosity=2).run(suite)
