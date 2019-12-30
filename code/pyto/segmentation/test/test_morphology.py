"""

Tests module morphology

More tests needed

# Author: Vladan Lucic
# $Id: test_morphology.py 1338 2016-09-28 13:38:40Z vladan $
"""

__version__ = "$Revision: 1338 $"

import numpy
import scipy

import unittest
from numpy.testing import *

from pyto.segmentation.segment import Segment
from pyto.segmentation.morphology import Morphology
from pyto.segmentation.contact import Contact
import common

class TestMorphology(TestCase):
    """
    """

    def setUp(self):
        self.shapes = common.make_shapes()

    def testVS(self):

        mor = Morphology(segments=self.shapes)

        # volume
        actual = mor.getVolume()
        desired = numpy.array([50, 9, 0, 21, 13, 0, 7])
        assert_equal(actual[self.shapes.ids], desired[self.shapes.ids])
        assert_equal(actual[0], desired[0])

        # surface
        actual = mor.getSurface()
        desired = numpy.array([32, 8, 0, 10, 7, 0, 7])
        assert_equal(actual[self.shapes.ids], desired[self.shapes.ids])
        assert_equal(actual[0], desired[0])

    def testCenter(self):

        # int
        mor = Morphology(segments=self.shapes)
        mor.getCenter()
        assert_equal(mor.center[self.shapes.ids], numpy.array([[2, 2],
                                                               [2, 7],
                                                               [8, 3],
                                                               [7, 7]]))
 
        # real
        mor = Morphology(segments=self.shapes)
        mor.getCenter(real=True)
        assert_almost_equal(mor.center[self.shapes.ids], 
                            numpy.array([[ 2.        ,  2.        ],
                                         [ 2.        ,  7.        ],
                                         [ 8.15384615,  3.38461538],
                                         [ 7.28571429,  6.71428571]]))

    def testRadius(self):

        mor = Morphology(segments=self.shapes)

        # normal radius
        mor.getRadius(doSlices=True, axis=1)
        desired_mean_1 = (numpy.sqrt(2) + 1.) / 2
        assert_almost_equal(
            mor.radius.mean[self.shapes.ids], 
            [desired_mean_1, 2.18885441, 1.77213119, 1.32865187])
        desired_std_1 = numpy.sqrt(
            ((numpy.sqrt(2.) - desired_mean_1)**2 + 
             (1. - desired_mean_1)**2) /2)
        assert_almost_equal(
            mor.radius.std[self.shapes.ids], 
            [desired_std_1, 0.09442658,  0.65649035,  0.72138648  ],
            decimal=5)
        assert_almost_equal(mor.radius.min[self.shapes.ids], 
                            [1.,  2.,  1.,  0.])
        assert_almost_equal(mor.radius.max[self.shapes.ids], 
                            [1.41421354,  2.23606801,  3.1622777,  2.23606801])

        # radius of a central slice
        assert_almost_equal(mor.sliceRadius[1].mean[self.shapes.ids], 
                            [1., 2., 1., 0.5])
        assert_almost_equal(mor.sliceRadius[1].std[self.shapes.ids], 
                            [0.,  0.,  0.,  0.5])
        assert_almost_equal(mor.sliceRadius[1].min[self.shapes.ids], 
                            [ 1.,  2.,  1.,  0.])
        assert_almost_equal(mor.sliceRadius[1].max[self.shapes.ids], 
                            [ 1.,  2.,  1.,  1.])

    def testCircleRadius(self):
        """
        Makes a circle and calculates radius
        """

        # parameters
        inc_radius = [6.5, 2.5, 7.5, 2.5]
        mean_radius = [6.1, 2.2, 7.1, 2.3]
        center_image = []

        # circle
        center_im = numpy.ones((20, 20), dtype=int)
        center_im[10,10] = 0
        center_image.append(center_im)

        # small circle
        center_im = numpy.ones((20, 20), dtype=int)
        center_im[10,10] = 0
        center_image.append(center_im)

        # sphere
        center_im = numpy.ones((20, 20, 20), dtype=int)
        center_im[10,10,10] = 0
        center_image.append(center_im)

        # small sphere
        center_im = numpy.ones((20, 20, 20), dtype=int)
        center_im[10,10,10] = 0
        center_image.append(center_im)

        # calculate for all
        for inc_r, mean_r, cent in zip(inc_radius, mean_radius, center_image):

            if (cent > 0).all():  # workaround for scipy bug 1089
                raise ValueError("Can't calculate distance_function ",
                                 "(no background)")
            else:
                dist = scipy.ndimage.distance_transform_edt(cent)
            circle = numpy.where(dist <= inc_r, 1, 0)
            mor = Morphology(segments=circle)
            mor.getRadius()
            assert_almost_equal(mor.radius.mean[1], mean_r, decimal=1)

    def testLength(self):

        # setup
        bound_data = numpy.array(
            [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]])
        bound = Segment(bound_data)

        seg_data = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 3, 3, 0, 0, 5, 5, 0],
             [0, 1, 3, 0, 0, 0, 0, 5, 0, 0],
             [1, 1, 0, 3, 0, 5, 5, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        seg = Segment(seg_data)

        con = Contact()
        con.findContacts(segment=seg, boundary=bound)

        # b2b, straight mode
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='b2b', line='straight', position=True)
        assert_almost_equal(mor.length[seg.ids], [4, 4, numpy.sqrt(17)])
        assert_equal(mor.end1[seg.ids], numpy.array([[0,1], [0,3], [0,7]]))
        assert_equal(mor.end2[seg.ids], numpy.array([[4,1], [4,3], [4,6]]))
       
        # check lengths in straight mode
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='c2c', line='straight', position=True)
        assert_almost_equal(mor.length[seg.ids], [2, 2, numpy.sqrt(5)])
        assert_equal(mor.end1[seg.ids], numpy.array([[1,1], [1,3], [1,7]]))
        assert_equal(mor.end2[seg.ids], numpy.array([[3,1], [3,3], [3,6]]))

        # b2c, straight mode
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='b2c', line='straight', position=True)
        assert_almost_equal(mor.length[seg.ids], [3, 3, numpy.sqrt(10)])
        assert_equal(mor.end1[seg.ids], numpy.array([[0,1], [0,3], [0,7]]))
        assert_equal(mor.end2[seg.ids], numpy.array([[3,1], [3,3], [3,6]]))
       
        # c2b, straight mode
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='c2b', line='straight', position=True)
        assert_almost_equal(mor.length[seg.ids], [3, 3, numpy.sqrt(10)])
        assert_equal(mor.end1[seg.ids], numpy.array([[1,1], [1,3], [1,7]]))
        assert_equal(mor.end2[seg.ids], numpy.array([[4,1], [4,3], [4,6]]))
       
        # b2b in mid mode
        bound.data[1,0] = 1 
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='b2b', line='mid', position=True)
        assert_almost_equal(mor.length[seg.ids], 
                            [4, 2*numpy.sqrt(5), 2 + numpy.sqrt(5)])
        assert_equal(
            mor.end1[seg.ids], numpy.array([[0,1], [0,3], [0,7]]))
        assert_equal(
            mor.end2[seg.ids], numpy.array([[4,1], [4,3], [4,6]]))

        # c2c in mid mode
        bound.data[1,0] = 1 
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='c2c', line='mid', position=True)
        assert_almost_equal(mor.length[seg.ids], 
                            [2, 2*numpy.sqrt(2), 1 + numpy.sqrt(2)])
        assert_equal(
            mor.end1[seg.ids], numpy.array([[1,1], [1,3], [1,7]]))
        assert_equal(
            mor.end2[seg.ids], numpy.array([[3,1], [3,3], [3,6]]))

        # c2b in mid mode
        bound.data[1,0] = 1 
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='c2b', line='mid', position=True)
        assert_almost_equal(
            mor.length[seg.ids], 
            [3, numpy.sqrt(5) + numpy.sqrt(2), 1 + numpy.sqrt(5)])
        assert_equal(
            mor.end1[seg.ids], numpy.array([[1,1], [1,3], [1,7]]))
        assert_equal(
            mor.end2[seg.ids], numpy.array([[4,1], [4,3], [4,6]]))

        # b2c in mid mode
        bound.data[1,0] = 1 
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='b2c', line='mid', position=True)
        assert_almost_equal(
            mor.length[seg.ids], 
            [3, numpy.sqrt(5) + numpy.sqrt(2), 2 + numpy.sqrt(2)])
        assert_equal(
            mor.end1[seg.ids], numpy.array([[0,1], [0,3], [0,7]]))
        assert_equal(
            mor.end2[seg.ids], numpy.array([[3,1], [3,3], [3,6]]))

        # check length when segment far from the smallest inter-boundary
        # distance
        wedge_bound_data = numpy.array(
             [[3, 3, 3, 3, 3, 0],
              [3, 3, 3, 3, 0, 0],
              [3, 3, 3, 0, 0, 0],
              [3, 3, 0, 0, 0, 0],
              [3, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [2, 2, 2, 2, 2, 2]])
        wedge_bound = Segment(data=wedge_bound_data)

        segment_data = numpy.array(
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 5, 5, 5],
             [0, 0, 0, 0, 0, 5],
             [0, 0, 0, 0, 0, 5],
             [0, 0, 0, 5, 5, 5],
             [0, 0, 0, 0, 0, 0]])
        seg = Segment(data=segment_data)

        con = Contact()
        con.findContacts(segment=seg, boundary=wedge_bound)

        mor = Morphology()
        mor.getLength(segments=seg, boundaries=wedge_bound, contacts=con, 
                      distance='c2c', line='straight', position=True)
        assert_almost_equal(mor.length[seg.ids], [3])

        mor = Morphology()
        mor.getLength(segments=seg, boundaries=wedge_bound, contacts=con, 
                      distance='c2c', line='mid', position=True)
        assert_almost_equal(mor.length[seg.ids], [2 + numpy.sqrt(5.)])

        mor = Morphology()
        mor.getLength(segments=seg, boundaries=wedge_bound, contacts=con, 
                      distance='c2c', line='mid-seg', position=True)
        assert_almost_equal(mor.length[seg.ids], [2 + numpy.sqrt(5.)])

        # boundary not in the smallest segment inset
        bound_data = numpy.array(
            [[1, 1, 1],
             [0, 0, 0],
             [2, 2, 2]])
        bound = Segment(data=bound_data)
        seg_data = numpy.array(
            [[0, 0, 0],
             [0, 5, 0],
             [0, 0, 0]])
        seg = Segment(data=seg_data)
        con = Contact()
        con.findContacts(segment=seg, boundary=bound)
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='c2c', line='straight', position=False)
        assert_almost_equal(mor.length[seg.ids], [0])
        
        # boundary not in the given segment inset
        bound_data = numpy.array(
            [[1, 1, 1],
             [0, 0, 0],
             [2, 2, 2]])
        bound = Segment(data=bound_data)
        bound.inset = [slice(1, 4), slice(2, 5)]
        seg_data = numpy.array(
            [[0, 0, 0],
             [0, 5, 0],
             [0, 0, 0]])
        seg = Segment(data=seg_data)
        seg.inset = [slice(1, 4), slice(2, 5)]
        con = Contact()
        con.findContacts(segment=seg, boundary=bound)
        seg.useInset(inset=[slice(2, 4), slice(2, 5)], mode='abs')
        mor = Morphology()
        mor.getLength(segments=seg, boundaries=bound, contacts=con, 
                      distance='c2c', line='straight', position=False)
        assert_almost_equal(mor.length[seg.ids], [0])
        


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMorphology)
    unittest.TextTestRunner(verbosity=2).run(suite)
#    unittest.main()
