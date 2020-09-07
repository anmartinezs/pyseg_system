"""

Tests module em_lm_correlation 

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.scene.em_lm_correlation import EmLmCorrelation

class TestEmLmCorrelation(np_test.TestCase):
    """
    """

    def setUp(self):
        pass

    def testEstablishMoveSearch(self):
        """
        Tests establish(mode='move_search')
        """

        # simple 
        lm_markers = numpy.array([[1, 2], [6, 2], [6, 5], [1, 5]])
        overview_markers = numpy.array([[0,0], [10,0], [10,6], [0,6]])
        overview_detail = numpy.array([[3,4], [6,4], [5,2]])
        search_detail = numpy.array([[2.5, 4], [4, 4], [3.5, 3]])
        c1 = EmLmCorrelation(mode='move search', 
                   lmMarkers=lm_markers, overviewMarkers=overview_markers,
                   overviewDetail=overview_detail, searchDetail=search_detail)
        c1.establish()
        np_test.assert_almost_equal(c1.lm2overview.d, [-2, -4])
        np_test.assert_almost_equal(c1.lm2overview.scale, [2, 2])
        np_test.assert_almost_equal(c1.overview2search.d, [1, 2])
        np_test.assert_almost_equal(c1.overview2search.scale, [0.5, 0.5])
        np_test.assert_almost_equal(c1.lm2search.scale, [1, 1])
        np_test.assert_almost_equal(c1.lm2search.d, [0, 0])
        np_test.assert_almost_equal(c1.lm2search.gl, [[1, 0], [0, 1]])

        # medium
        lm_markers = numpy.array(
            [[ 2.08493649,  0.95096189],
             [ 6.41506351,  3.45096189],
             [ 4.91506351,  6.04903811],
             [ 0.58493649,  3.54903811]])
        overview_markers = numpy.array([[0,0], [10,0], [10,6], [0,6]])
        overview_detail = numpy.array(
            [[ 3.76794919,  4.8660254 ],
             [ 6.3660254 ,  3.3660254 ],
             [ 4.5       ,  2.1339746 ]])
        search_detail = numpy.array([[2.5, 4], [4, 4], [3.5, 3]])
        c2 = EmLmCorrelation(mode='move search', 
                   lmMarkers=lm_markers, overviewMarkers=overview_markers,
                   overviewDetail=overview_detail, searchDetail=search_detail)
        c2.establish()
        np_test.assert_almost_equal(c2.lm2search.d, [0, 0])
        np_test.assert_almost_equal(c2.lm2search.gl, [[1, 0], [0, 1]])
        np_test.assert_almost_equal(c2.lm2overview.phi, -numpy.pi/6)
        np_test.assert_almost_equal(c2.overview2search.phi, numpy.pi/6)
        spots = numpy.array([[1,2], [3,4], [5,-2]])
        corr_spots = c2.lm2search.transform(spots)
        np_test.assert_almost_equal(corr_spots, spots)

        # medium with parity
        lm_markers = numpy.array(
            [[ 0.95096189, 2.08493649],
             [ 3.45096189, 6.41506351],
             [ 6.04903811, 4.91506351],
             [ 3.54903811, 0.58493649]])
        c3 = EmLmCorrelation(mode='move search', 
                   lmMarkers=lm_markers, overviewMarkers=overview_markers,
                   overviewDetail=overview_detail, searchDetail=search_detail)
        c3.establish()
        np_test.assert_almost_equal(c3.lm2overview.phi, numpy.pi/3)
        np_test.assert_equal(c3.lm2overview.parity, -1)
        np_test.assert_almost_equal(c3.overview2search.phi, numpy.pi/6)
        np_test.assert_almost_equal(c3.overview2search.d, 
                                    [2.08493649, 0.95096189])
        np_test.assert_almost_equal(c3.lm2search.phi, numpy.pi/2) 
        np_test.assert_almost_equal(c3.lm2search.parity, -1)
        np_test.assert_almost_equal(c3.lm2search.scale, 1)
        np_test.assert_almost_equal(c3.lm2search.d, [0, 0])
        np_test.assert_almost_equal(c3.lm2search.gl, [[0, 1], [1, 0]])
        spots = numpy.array([[1,2], [3,4], [5,-2]])
        corr_spots = c3.lm2search.transform(spots)
        desired = numpy.array([[2,1], [4,3], [-2,5]])
        np_test.assert_almost_equal(corr_spots, desired)

    def testEstablishMoveSearchSeparate(self):
        """
        Tests establish(mode='move_search') with separate lm to overview gl 
        and d   
        """

        # simple 
        lm_markers = numpy.array([[1, 2], [6, 2], [6, 5], [1, 5]])
        lm_markers_d = lm_markers[0:2]
        overview_markers = numpy.array([[0,0], [10,0], [10,6], [0,6]]) + [1,2]
        overview_markers_d = numpy.array([[0,0], [10,0]])
        overview_detail = numpy.array([[3,4], [6,4], [5,2]])
        search_detail = numpy.array([[2.5, 4], [4, 4], [3.5, 3]])
        c1 = EmLmCorrelation(
            mode='move search', lmMarkersGl=lm_markers, lmMarkersD=lm_markers_d,
            overviewMarkersGl=overview_markers,
            overviewMarkersD=overview_markers_d,
            overviewDetail=overview_detail, searchDetail=search_detail)
        c1.establish()
        c1.decompose(order='qpsm')
        np_test.assert_almost_equal(c1.lm2overview.d, [-2, -4])
        np_test.assert_almost_equal(c1.lm2overview.scale, [2, 2])
        np_test.assert_almost_equal(c1.overview2search.d, [1, 2])
        np_test.assert_almost_equal(c1.overview2search.scale, [0.5, 0.5])
        np_test.assert_almost_equal(c1.lm2search.scale, [1, 1])
        np_test.assert_almost_equal(c1.lm2search.d, [0, 0])
        np_test.assert_almost_equal(c1.lm2search.gl, [[1, 0], [0, 1]])

        # medium with parity
        lm_markers = numpy.array(
            [[ 0.95096189, 2.08493649],
             [ 3.45096189, 6.41506351],
             [ 6.04903811, 4.91506351],
             [ 3.54903811, 0.58493649]])
        lm_markers_d = lm_markers[2:4]
        overview_markers = numpy.array([[0,0], [10,0], [10,6], [0,6]]) + [2,3]
        overview_markers_d = numpy.array([[10,6], [0,6]])
        overview_detail = numpy.array(
            [[ 3.76794919,  4.8660254 ],
             [ 6.3660254 ,  3.3660254 ],
             [ 4.5       ,  2.1339746 ]])
        search_detail = numpy.array([[2.5, 4], [4, 4], [3.5, 3]])
        c3 = EmLmCorrelation(
            mode='move search', lmMarkersGl=lm_markers, lmMarkersD=lm_markers_d,
            overviewMarkersGl=overview_markers, 
            overviewMarkersD=overview_markers_d,
            overviewDetail=overview_detail, searchDetail=search_detail)
        c3.establish()
        np_test.assert_almost_equal(c3.lm2overview.phi, numpy.pi/3)
        np_test.assert_equal(c3.lm2overview.parity, -1)
        np_test.assert_almost_equal(c3.overview2search.phi, numpy.pi/6)
        np_test.assert_almost_equal(c3.overview2search.d, 
                                    [2.08493649, 0.95096189])
        np_test.assert_almost_equal(c3.lm2search.phi, numpy.pi/2) 
        np_test.assert_almost_equal(c3.lm2search.parity, -1)
        np_test.assert_almost_equal(c3.lm2search.scale, 1)
        np_test.assert_almost_equal(c3.lm2search.d, [0, 0])
        np_test.assert_almost_equal(c3.lm2search.gl, [[0, 1], [1, 0]])
        spots = numpy.array([[1,2], [3,4], [5,-2]])
        corr_spots = c3.lm2search.transform(spots)
        desired = numpy.array([[2,1], [4,3], [-2,5]])
        np_test.assert_almost_equal(corr_spots, desired)

    def testEstablishMoveOverview(self):
        """
        Tests establish(mode='move overview')
        """

        # simple 
        lm_markers = numpy.array([[1, 2], [6, 2], [6, 5], [1, 5]])
        overview_markers = numpy.array([[0,0], [10,0], [10,6], [0,6]])
        overview_detail = numpy.array([[3,4], [5, 4], [5, 8], [3, 8]])
        search_detail = numpy.array([[1, 2], [0, 2], [0, 0], [1, 0]])
        c1 = EmLmCorrelation(
            mode='move overview', 
            lmMarkers=lm_markers, overviewMarkers=overview_markers,
            overviewDetail=overview_detail, searchDetail=search_detail,
            searchMain=search_detail[0], overviewCenter=[5, 4])
        c1.establish()
        np_test.assert_almost_equal(c1.lm2overview.d, [-2, -4])
        np_test.assert_almost_equal(c1.lm2overview.scale, [2, 2])
        np_test.assert_almost_equal(c1.lm2overview.rmsError, 0)
        np_test.assert_almost_equal(c1.overview2search.gl, [[0.5, 0], [0, 0.5]])
        np_test.assert_almost_equal(c1.overview2search.scale, [0.5, 0.5])
        np_test.assert_almost_equal(c1.overview2search.d, [-1.5, 0])
        np_test.assert_almost_equal(c1.overview2search.rmsError, 0)
        np_test.assert_almost_equal(c1.lm2search.scale, [1, 1])
        np_test.assert_almost_equal(c1.lm2search.gl, [[1, 0], [0, 1]])
        np_test.assert_almost_equal(c1.lm2search.d, [-2.5, -2])
        np_test.assert_almost_equal(
            c1.lm2search.transform(lm_markers),
            numpy.array(lm_markers) + [-2.5, -2])
        np_test.assert_almost_equal(c1.lm2search.rmsError is None, True)
        np_test.assert_almost_equal(c1.lm2search.rmsErrorEst, 0)
        np_test.assert_almost_equal(c1.lm2search.transform([2.5, 4]), [0, 2])
        np_test.assert_almost_equal(c1.lm2search.transform([1, 2]), [-1.5, 0])
        np_test.assert_almost_equal(c1.lm2search.transform([6, 5]), [3.5, 3])

        # simple with different centers
        for index in range(len(overview_detail)):
            c1 = EmLmCorrelation(
                mode='move overview', 
                lmMarkers=lm_markers, overviewMarkers=overview_markers,
                overviewDetail=overview_detail, searchDetail=search_detail,
                searchMain=search_detail[0], 
                overviewCenter=overview_detail[index])
            c1.establish()
            np_test.assert_almost_equal(
                c1.overview2search.transform(overview_detail[index]), 
                search_detail[0])

        # medium
        lm_markers = numpy.array(
            [[-0.1339746 ,  2.23205081],
             [ 0.73205081,  2.73205081],
             [ 0.43205081,  3.25166605],
             [-0.4339746 ,  2.75166605]])      
        overview_markers = numpy.array([[0,0], [10,0], [10,6], [0,6]])
        overview_detail = numpy.array([[3,4], [3, 6], [7, 6], [7, 4]])
        search_detail = numpy.array([[1, 2], [0, 2], [0, 4], [1, 4]])
        c1 = EmLmCorrelation(mode='move overview', 
                   lmMarkers=lm_markers, overviewMarkers=overview_markers,
                   overviewDetail=overview_detail, searchDetail=search_detail,
                   searchMain=search_detail[0], overviewCenter=[5, 4])
        c1.establish()
        np_test.assert_almost_equal(c1.lm2overview.phi, -numpy.pi/6)
        np_test.assert_almost_equal(c1.lm2overview.scale, [10, 10])
        np_test.assert_almost_equal(c1.lm2overview.d, [-10, -20])
        np_test.assert_almost_equal(c1.overview2search.phi, -numpy.pi/2)
        np_test.assert_almost_equal(c1.overview2search.scale, [0.5, 0.5])
        np_test.assert_almost_equal(c1.overview2search.rmsError, 0)
        np_test.assert_almost_equal(c1.overview2search.d, [-1, 4.5])
        np_test.assert_almost_equal(c1.lm2search.scale, [5, 5])
        np_test.assert_almost_equal(c1.lm2search.phi, -2*numpy.pi/3)
        np_test.assert_almost_equal(c1.lm2search.d, [-11, 9.5])

        # medium with different centers
        for index in range(len(overview_detail)):
            c1 = EmLmCorrelation(
                mode='move overview', 
                lmMarkers=lm_markers, overviewMarkers=overview_markers,
                overviewDetail=overview_detail, searchDetail=search_detail,
                searchMain=search_detail[0], 
                overviewCenter=overview_detail[index])
            c1.establish()
            np_test.assert_almost_equal(
                c1.overview2search.transform(overview_detail[index]), 
                search_detail[0])

        # medium with parity
        # Need to redo
        #lm_markers = numpy.array(
        #    [[ 0.95096189, 2.08493649],
        #     [ 3.45096189, 6.41506351],
        #     [ 6.04903811, 4.91506351],
        #     [ 3.54903811, 0.58493649]])
        #c3 = EmLmCorrelation(mode='move overview', 
        #           lmMarkers=lm_markers, overviewMarkers=overview_markers,
        #           overviewDetail=overview_detail, searchDetail=search_detail,
        #           searchMain=search_detail[0], overviewCenter=[5, 4])
        #c3.establish()
        #np_test.assert_almost_equal(c3.lm2overview.phi, numpy.pi/3)
        #np_test.assert_equal(c3.lm2overview.parity, -1)
        #np_test.assert_almost_equal(c3.overview2search.phi, numpy.pi/6)
        #np_test.assert_almost_equal(c3.overview2search.d, 
        #                            [2.08493649,  0.95096188])
        #np_test.assert_almost_equal(c3.lm2search.phi, numpy.pi/2) 
        #np_test.assert_almost_equal(c3.lm2search.parity, -1)
        #np_test.assert_almost_equal(c3.lm2search.scale, 1)
        #np_test.assert_almost_equal(c3.lm2search.d, [0, 0])
        #np_test.assert_almost_equal(c3.lm2search.gl, [[0, 1], [1, 0]])
        #spots = numpy.array([[1,2], [3,4], [5,-2]])
        #corr_spots = c3.lm2search.transform(spots)
        #desired = numpy.array([[2,1], [4,3], [-2,5]])
        #np_test.assert_almost_equal(corr_spots, desired)

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmLmCorrelation)
    unittest.TextTestRunner(verbosity=2).run(suite)
