"""

Tests module basic

# Author: Vladan Lucic
# $Id: test_basic.py 1311 2016-06-13 12:41:50Z vladan $
"""

__version__ = "$Revision: 1311 $"

from copy import copy, deepcopy
import os
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.geometry.affine import Affine
from pyto.geometry.affine_2d import Affine2D
from pyto.correlative.basic import Basic

class TestBasic(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        From ..geometry.test.test_affine
        """

        # parallelogram, rotation, scale, exact
        self.d1 = [-1, 2]
        self.x1 = numpy.array([[0., 0], [2, 0], [2, 1], [0, 1]])
        self.y1 = numpy.array([[0., 0], [4, 2], [3, 4], [-1, 2]]) + self.d1
        self.y1m = numpy.array([[0., 0], [-4, 2], [-3, 4], [1, 2]]) + self.d1
        self.af1_gl = numpy.array([[2,-1],[1,2]])
        self.af1_d = self.d1
        self.af1_phi = numpy.arctan(0.5)
        self.af1_scale = numpy.array([numpy.sqrt(5)] * 2)
        self.af1_parity = 1
        self.af1_shear = 0

        # parallelogram, rotation, scale, not exact
        self.d2 = [-1, 2]
        self.x2 = numpy.array([[0.1, -0.2], [2.2, 0.1], [1.9, 0.8], [0.2, 1.1]])
        self.y2 = numpy.array([[0., 0], [4, 2], [3, 4], [-1, 2]]) + self.d2
        self.y2m = numpy.array([[0., 0], [-4, 2], [-3, 4], [1, 2]]) + self.d2

        # test file
        curr_dir, base = os.path.split(os.path.abspath(__file__))
        self.imagej_file =  os.path.join(curr_dir, 'imagej.txt')


    def testGetTransformation(self):
        """
        Tasts getTransformation()
        """

        corr = Basic()
        corr.markers_1 = self.x1
        corr.markers_2 = self.y1
        corr.establish()
        
        np_test.assert_almost_equal(corr.getTransformation(from_=1, to=2).gl,
                                    corr.transf_1_to_2.gl)
        np_test.assert_almost_equal(corr.getTransformation(from_=2, to=1).gl,
                                    corr.transf_2_to_1.gl)
        #np_test.assert_raises(ValueError, corr.getTransformation, 
        #                      **{'from_':1, 'to':3})
   
    def testReadPositions(self):
        """
        Tests readPositions()
        """

        corr = Basic()

        # indexing 1 (default)
        points = {'markers': (self.imagej_file, [3, 4, 5]), 
                  'targets': (self.imagej_file, [6, 7, 8])}
        corr.readPositions(points=points, format_='imagej', columns=[8,9])
        desired_markers = numpy.array([[120,	114.667],
                                       [103.333,	108],
                                       [126,	130.667]])
        np_test.assert_almost_equal(corr.markers, desired_markers)
        desired_targets = numpy.array([[221.333,	72.667],
                                       [121.333,	172.667],
                                       [106.667,	278.667]])
        np_test.assert_almost_equal(corr.targets, desired_targets)

        # indexing 0 (old way)
        points = {'markers': (self.imagej_file, [2, 3, 4]), 
                  'targets': (self.imagej_file, [5, 6, 7])}
        corr.readPositions(points=points, format_='imagej', columns=[7, 8], 
                           indexing=0)
        np_test.assert_almost_equal(corr.markers, desired_markers)
        np_test.assert_almost_equal(corr.targets, desired_targets)
                           
    def testEstablish(self):
        """
        Tests establish()
        """

        # markers read from a file
        corr = Basic()
        points = {'markers_1': (self.imagej_file, [3, 4, 5]), 
                  'markers_2': (self.imagej_file, [6, 7, 8])}
        corr.establish(points=points, format_='imagej', columns=[8,9])
        desired_markers = numpy.array([[120,	114.667],
                                       [103.333,	108],
                                       [126,	130.667]])
        np_test.assert_almost_equal(corr.markers_1, desired_markers)
        desired_markers = numpy.array([[221.333,	72.667],
                                       [121.333,	172.667],
                                       [106.667,	278.667]])
        np_test.assert_almost_equal(corr.markers_2, desired_markers)
        
        # markers given in arguments
        corr = Basic()
        corr.establish(markers_1=self.x1, markers_2=self.y1)

        np_test.assert_equal(isinstance(corr.getTransformation(from_=1, to=2), 
                                        Affine2D), True)
        np_test.assert_almost_equal(corr.markers_1, self.x1)
        np_test.assert_almost_equal(corr.markers_2, self.y1)

        np_test.assert_almost_equal(corr.getTransformation(from_=1, to=2).gl, 
                                    self.af1_gl)
        np_test.assert_almost_equal(corr.getTransformation(from_=1, to=2).d, 
                                    self.af1_d)

        # markers set
        corr = Basic()
        corr.markers_1 = self.x1
        corr.markers_2 = self.y1
        corr.establish()
                              
        np_test.assert_almost_equal(corr.markers_1, self.x1)
        np_test.assert_almost_equal(corr.markers_2, self.y1)

        transf_1_to_2 = corr.getTransformation(from_=1, to=2)
        np_test.assert_almost_equal(corr.getTransformation(from_=1, to=2).gl, 
                                    self.af1_gl)
        np_test.assert_almost_equal(corr.getTransformation(from_=1, to=2).d, 
                                    self.af1_d)
        transf_2_to_1 = corr.getTransformation(from_=2, to=1)
        np_test.assert_almost_equal(transf_2_to_1.gl, 
                                    transf_1_to_2.inverse().gl)
        np_test.assert_almost_equal(transf_2_to_1.d, 
                                    transf_1_to_2.inverse().d)

        # markers mixed
        corr = Basic()
        corr.markers_1 = self.x1 *2.1 # set wrong to check if overwritten 
        corr.markers_2 = self.y1
        corr.establish(markers_1=self.x1)

        np_test.assert_almost_equal(corr.getTransformation(from_=1, to=2).gl, 
                                    self.af1_gl)
        np_test.assert_almost_equal(corr.getTransformation(from_=1, to=2).d, 
                                    self.af1_d)

    def testDecompose(self):
        """
        Tests decompose()
        """

        corr = Basic()
        corr.establish(markers_1=self.x1, markers_2=self.y1)
        corr.decompose(order='qpsm')
        np_test.assert_almost_equal(corr.transf_1_to_2.phi, numpy.arctan(0.5))
        np_test.assert_almost_equal(corr.getTransformation(from_=1, to=2).scale,
                                    [numpy.sqrt(5)] * 2)
        np_test.assert_almost_equal(corr.getTransformation(from_=2, to=1).phi,
                                    -numpy.arctan(0.5))
        np_test.assert_almost_equal(corr.transf_2_to_1.scale, 
                                    [1/numpy.sqrt(5)] * 2)
         
    def testCorrelate(self):
        """
        Tests correlate
        """

        # establish correlation
        corr = Basic()
        corr.establish(markers_1=self.x1, markers_2=self.y1)

        # test mixed set and passed targets
        corr.targets_2 = self.y1[0:2]
        corr.correlate(targets_1=self.x1[2:4])
        np_test.assert_almost_equal(corr.correlated_1_to_2, self.y1[2:4])
        np_test.assert_almost_equal(corr.getTargets(from_=2, to=1), 
                                    self.x1[0:2])

        # test mixed set and passed targets, set need to be overwritten
        corr.targets_1 = self.x1[2:4] * 3.1 # bad
        corr.targets_2 = self.y1[0:2]
        corr.correlate(targets_1=self.x1[2:4])
        np_test.assert_almost_equal(corr.correlated_1_to_2, self.y1[2:4])
        np_test.assert_almost_equal(corr.getTargets(from_=2, to=1), 
                                    self.x1[0:2])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBasic)
    unittest.TextTestRunner(verbosity=2).run(suite)
