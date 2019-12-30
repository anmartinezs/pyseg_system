"""

Tests module analysis.connections.

# Author: Vladan Lucic
# $Id: test_connections.py 1151 2015-05-26 08:52:30Z vladan $
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
from pyto.analysis.connections import Connections


class TestConnections(np_test.TestCase):
    """
    Tests Connections.
    """

    def setUp(self):
        """
        """

        # local path
        dir_, base = os.path.split(__file__)

        # set catalog
        catalog = pyto.analysis.Catalog()
        catalog._db = {
            'category' : {'77_4' : 'rim_wt', '78_3' : 'rim_wt', 
                          '75_4' : 'rim_altered'},
            'tether_files' : {'77_4': 'segmentations/conn_77-4_new_AZ.pkl',
                              '78_3': 'segmentations/conn_78-3_new_AZ.pkl',
                              '75_4' : 'segmentations/conn_75-4_new_AZ.pkl'},
            'sv_files' : {'77_4': 'segmentations/sv_77-4_vesicles.pkl',
                          '78_3': 'segmentations/sv_78-3_vesicles.pkl',
                          '75_4' : 'segmentations/sv_75-4_vesicles.pkl'},
            'pixel_size' : {'77_4' : 2.644, '78_3' : 2.644, '75_4' : 2.644},
            'operator' : {'77_4' : 'emerson', '78_3' : 'lake', 
                          '75_4' : 'palmer'}
            }
        for ident, name in catalog._db['tether_files'].items():
               catalog._db['tether_files'][ident] = os.path.join(dir_, name)
        for ident, name in catalog._db['sv_files'].items():
               catalog._db['sv_files'][ident] = os.path.join(dir_, name)
        catalog.makeGroups()
        self.catalog = catalog
        
        # read tethers and sv
        self.tether = Connections.read(files=catalog.tether_files, 
                                       mode='connectors', catalog=catalog)
        from pyto.analysis.vesicles import Vesicles
        self.sv = Vesicles.read(files=catalog.sv_files, catalog=catalog)

        # set segmentation files and adjust paths
        tether_files = {
            'rim_wt' : {'77_4': 'segmentations/conn_77-4_AZ_all.pkl',
                        '78_3': 'segmentations/conn_78-3_AZ_all.pkl'},
            'rim_altered' : {'75_4' : 'segmentations/conn_75-4_AZ_all.pkl'}}
        for categ in tether_files:
            for ident, name in tether_files[categ].items():
               tether_files[categ][ident] = os.path.join(dir_, name)
        self.tether_files = tether_files

        # set pixel size
        self.pixel_size = {
            'rim_wt' : {'77_4' : 2.644, '78_3' : 2.644},
            'rim_altered' : {'75_4' : 2.644}}

    def testReadConnectors(self):
        """
        Tests read() with mode='connectors'
        """

        # read
        tether = Connections.read(
            files=self.catalog.tether_files, 
            mode='connectors', catalog=self.catalog)

        # test general
        np_test.assert_equal(tether.rim_wt.identifiers, ['77_4', '78_3'])
        np_test.assert_equal(tether.rim_altered.identifiers, ['75_4'])

        # test 77_4
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='ids'), 
            [11, 13, 37, 46, 72, 79, 83, 130, 141, 146, 156, 168, 224,
             238, 244, 292, 333])
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='surface'), 
            [18, 79, 19, 44, 30, 57, 97, 293,  12, 33, 73, 47, 173,
             30, 8, 69, 24])
        np_test.assert_almost_equal(
            tether.rim_wt.getValue(identifier='77_4', property='distance'), 
            [1.2, 2.4, 1.9, 3.2, 3.6, 3.06, 4.3, 2.7, 1.26, 2.6, 5.5, 3.3,
             2.4, 2.5, 2.0, 1.7, 2.9],
            decimal=1)
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='boundaries',
                                   ids=[224]),
            [[1,3]])
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='boundaries',
                                   ids=224),
            [1,3])
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='boundaries',
                                   ids=[156, 72]),
            [[1,45], [1,88]])
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='operator'), 
            'emerson')

        # test 78_3
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='78_3', property='ids'), 
            [ 31,  37,  44,  79, 139, 145, 159, 164])
        np_test.assert_almost_equal(
            tether.rim_wt.getValue(identifier='78_3', property='length'), 
            [3., 5.0, 4.1, 17.4, 11.7, 8.2, 5.9, 6.5],
            decimal=1)
        np_test.assert_almost_equal(
            tether.rim_wt.getValue(identifier='78_3', property='distance_nm'),
            [13.8, 9.6, 6.6, 18.1, 15.5, 22.8, 11.9, 11.7],
            decimal=1)
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='78_3', property='boundaries'),
            [[1,  2], [ 1,  2], [ 1, 35], [ 1, 31], [ 1,  2], [ 1, 31], 
             [ 1, 31], [ 1, 31]])

        # test 75_4
        np_test.assert_equal(
            tether.rim_altered.getValue(identifier='75_4', property='ids'), 
            [124, 268, 343, 408])
        np_test.assert_equal(
            tether.rim_altered.getValue(identifier='75_4', property='volume'),
            [124, 262, 4092 , 470]) 
        np_test.assert_almost_equal(
            tether.rim_altered.getValue(identifier='75_4', 
                                   property='boundaryDistance'),
            [ 4.24264069,  4.24264069,  3.        ,  7.68114575])

    def testReadConnectors_order(self):
        """
        Tests read() with specified order and mode='connectors'
        """

        # make order and read
        order = {'rim_wt' : ['78_3', '77_4'], 'rim_altered' : ['75_4']}
        tether = Connections.read(
            files=self.catalog.tether_files, order=order,
            mode='connectors', catalog=self.catalog)

        # test general
        np_test.assert_equal(tether.rim_wt.identifiers, ['78_3', '77_4'])
        np_test.assert_equal(tether.rim_altered.identifiers, ['75_4'])

    def testRead(self):
        """
        Tests read() with mode='sv_old'

        ToDo
        """

        # read
        tether = Connections.read(files=self.tether_files, mode='sv_old', 
                                 pixel=self.pixel_size)

        # test general
        np_test.assert_equal(tether.rim_wt.identifiers, ['77_4', '78_3'])
        np_test.assert_equal(tether.rim_altered.identifiers, ['75_4'])

        # test 77_4
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='ids'), 
            [11, 13, 37, 46, 72, 79, 83, 141, 146, 156, 168, 
             238, 244, 292, 333])
        #np_test.assert_equal(
        #    tether.rim_wt.getValue(identifier='77_4', property='surface'), 
        #    [18, 79, 19, 44, 30, 57, 97,  12, 33, 73, 47, 
        #     30, 8, 69, 24])
        np_test.assert_almost_equal(
            tether.rim_wt.getValue(identifier='77_4', property='distance'), 
            [1.2, 2.4, 1.9, 3.2, 3.6, 3.06, 4.3, 1.26, 2.6, 5.5, 3.3,
             2.5, 2.0, 1.7, 2.9],
            decimal=1)
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='boundaries',
                                   ids=238),
            [1,6])
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='boundaries',
                                   ids=[156, 72])[0],
            numpy.array([1,45]))
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='77_4', property='boundaries',
                                   ids=[156, 72])[1],
            [1,88])

        # test 78_3
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='78_3', property='ids'), 
            [ 31,  37,  159, 164])
        np_test.assert_almost_equal(
            tether.rim_wt.getValue(identifier='78_3', property='length'), 
            [3., 5.0,  5.9, 6.5],
            decimal=1)
        np_test.assert_almost_equal(
            tether.rim_wt.getValue(identifier='78_3', property='distance_nm'),
            [13.8, 9.6, 11.9, 11.7],
            decimal=1)
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='78_3', property='boundaries',
                                   ids=31),
            [1,  2])
        np_test.assert_equal(
            tether.rim_wt.getValue(identifier='78_3', property='boundaries',
                                   ids=[164])[0],
            numpy.array([ 1, 31]))

        # test 75_4
        np_test.assert_equal(
            tether.rim_altered.getValue(identifier='75_4', property='ids'), 
            [124])
        #np_test.assert_equal(
        #    tether.rim_altered.getValue(identifier='75_4', property='volume'),
        #    [124]) 
        np_test.assert_almost_equal(
            tether.rim_altered.getValue(identifier='75_4', 
                                   property='boundaryDistance'),
            [ 4.24264069])

    def testExtractByVesicles(self):
        """
        Tests extractByVesicles()
        """

        # split svs and extract tethers
        self.sv.getNTethers(tether=self.tether)
        sv_1, sv_2 = self.sv.split(name='n_tether', value=[1,3,300])
        tether_1 = self.tether.extractByVesicles(vesicles=sv_1, other=False)
        tether_2 = self.tether.extractByVesicles(vesicles=sv_2, other=False)

        # test 
        for g_name in tether_1:
            for ident in tether_1[g_name].identifiers:

                # get tether boundaries
                bounds = self.tether[g_name].getValue(name='boundaries', 
                                                      identifier=ident)
                #print "bounds tether: ", ident, bounds 
                bounds = set(pyto.util.nested.flatten(bounds))
                bounds.remove(1)
                bounds_1 = tether_1[g_name].getValue(name='boundaries', 
                                                     identifier=ident)
                #print "bounds tether_1: ", ident, bounds_1 
                bounds_1 = set(pyto.util.nested.flatten(bounds_1))
                try:
                    bounds_1.remove(1)
                except KeyError:
                    pass
                bounds_2 = tether_2[g_name].getValue(name='boundaries', 
                                                     identifier=ident)
                #print "bounds tether_2: ", ident, bounds_2 
                bounds_2 = set(pyto.util.nested.flatten(bounds_2))
                try:
                    bounds_2.remove(1)
                except KeyError:
                    pass

                # get sv ids
                sv_ids = self.sv[g_name].getValue(name='ids', identifier=ident)
                sv_ids_1 = sv_1[g_name].getValue(name='ids', identifier=ident)
                sv_ids_2 = sv_2[g_name].getValue(name='ids', identifier=ident)

                # test if extracted tether boundaries correspond to sv ids
                #print "bounds tether_1: ", ident, bounds_1 
                #print "bounds tether_2: ", ident, bounds_2 
                #print "sv_ids_2: ", ident, sv_ids_2
                np_test.assert_equal(
                        numpy.in1d(list(bounds_1), sv_ids_1).all(), True)
                np_test.assert_equal(
                        numpy.intersect1d(list(bounds_1), sv_ids_2), [])
                np_test.assert_equal(
                        numpy.in1d(list(bounds_2), sv_ids_2).all(), True)
                np_test.assert_equal(
                        numpy.intersect1d(list(bounds_2), sv_ids_1), [])

                # test if other tether properties are kept after extraction
                teth_ids = self.tether[g_name].getValue(
                    name='ids',identifier=ident)
                teth_ids_1 = tether_1[g_name].getValue(
                    name='ids', identifier=ident)
                teth_ids_2 = tether_2[g_name].getValue(
                    name='ids', identifier=ident)
                length_1 = tether_1[g_name].getValue(
                    name='length', identifier=ident)
                length_2 = tether_2[g_name].getValue(
                    name='length', identifier=ident)
                length = self.tether[g_name].getValue(
                    name='length', identifier=ident, ids=teth_ids_1)
                np_test.assert_almost_equal(length_1, length)
                length = self.tether[g_name].getValue(
                    name='length', identifier=ident, ids=teth_ids_2)
                np_test.assert_almost_equal(length_2, length)
               
        # exact tests 77_4
        bounds = tether_1['rim_wt'].getValue(
            name='boundaries', identifier='77_4')
        bounds = set(pyto.util.nested.flatten(bounds))
        np_test.assert_equal(bounds, set([1, 3, 4, 30, 45, 88]))
        bounds = tether_2['rim_wt'].getValue(
            name='boundaries', identifier='77_4')
        bounds = set(pyto.util.nested.flatten(bounds))
        np_test.assert_equal(bounds, set([1, 5, 6, 22]))

        # exact tests 78_3
        bounds = tether_1['rim_wt'].getValue(
            name='boundaries', identifier='78_3')
        bounds = set(pyto.util.nested.flatten(bounds))
        np_test.assert_equal(bounds, set([1, 35]))
        bounds = tether_2['rim_wt'].getValue(
            name='boundaries', identifier='78_3')
        bounds = set(pyto.util.nested.flatten(bounds))
        np_test.assert_equal(bounds, set([1, 2, 31]))

        # exact tests 75_4
        bounds = tether_1['rim_altered'].getValue(
            name='boundaries', identifier='75_4')
        bounds = set(pyto.util.nested.flatten(bounds))
        np_test.assert_equal(bounds, set([1, 21, 73, 103]))
        bounds = tether_2['rim_altered'].getValue(
            name='boundaries', identifier='75_4')
        bounds = set(pyto.util.nested.flatten(bounds))
        np_test.assert_equal(bounds, set([]))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConnections)
    unittest.TextTestRunner(verbosity=2).run(suite)
