"""

Tests module cleft_regions

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

from pyto.core.image import Image
from pyto.segmentation.segment import Segment
from pyto.segmentation.cleft import Cleft
from pyto.segmentation.test.test_cleft import TestCleft
from pyto.scene.cleft_regions import CleftRegions
import pyto.scene.test.common as common
        
class TestCleftRegions(np_test.TestCase):
    """
    """

    def setUp(self):
        
        # make image
        self.image_1 = Image(numpy.arange(100).reshape(10,10)) 

        # make cleft 1 
        cleft_data_1 = numpy.zeros((10,10), dtype=int)
        cleft_data_1[slice(1,7), slice(3,9)] = numpy.array(\
            [[0, 5, 5, 5, 0, 0],
             [2, 5, 5, 5, 4, 4],
             [2, 5, 5, 5, 4, 4],
             [2, 2, 5, 5, 5, 4],
             [2, 5, 5, 5, 4, 4],
             [2, 5, 5, 5, 4, 4]])
        self.bound_1 = Segment(cleft_data_1)
        self.cleft_1 = Cleft(data=self.bound_1.data, bound1Id=2, bound2Id=4,
                             cleftId=5, copy=True, clean=False)
        self.cleft_1_ins = Cleft(data=self.bound_1.data[1:7, 3:9], bound1Id=2,
                                 bound2Id=4, cleftId=5, copy=True, clean=False)
        self.cleft_1_ins.inset = [slice(1,7), slice(3,9)]

        # make big cleft and image
        self.cleft_2, self.image_2 = common.make_cleft_layers_example()
        #self.cleft_2.inset = [slice(1,10), slice(2,7)]

    @classmethod
    def makeCleft3(cls):
        """
        Makes a Cleft object
        """

        # boundaries
        bound_3 = numpy.array(
            [[1, 1, 1, 1],
             [3, 1, 1, 3],
             [3, 3, 3, 3],
             [3, 2, 2, 3],
             [2, 2, 2, 2]])
        cleft = Cleft(data=bound_3, bound1Id=1, bound2Id=2, cleftId=3, 
                      copy=True, clean=False)

        return cleft

    def testMakeLayers(self):
        """
        Tests CleftRegions.makeLayers().

        Some tests same as in pyto.segmentation.test.test_cleft
        """

        # cleft_1
        cl_1 = CleftRegions(image=None, cleft=self.cleft_1)
        cl_1.makeLayers()
        cl_1_ins = CleftRegions(image=None, cleft=self.cleft_1_ins)
        cl_1_ins.makeLayers()

        # test width and nLayers, median width mode
        np_test.assert_almost_equal(cl_1.width, (numpy.sqrt(13) + 4) / 2 - 1) 
        np_test.assert_equal(cl_1.nLayers, 3)

        # test layers no insets
        desired_layers_ins = numpy.array(\
            [[0, 1, 2, 3, 0, 0],
             [0, 1, 2, 3, 0, 0],
             [0, 1, 2, 3, 0, 0],
             [0, 0, 1, 2, 3, 0],
             [0, 1, 2, 3, 0, 0],
             [0, 1, 2, 3, 0, 0]])            
        np_test.assert_equal(cl_1.regions.data, desired_layers_ins)

        # test layers, expand
        desired_layers = numpy.zeros((10,10), dtype='int')
        desired_layers[1:7, 3:9] = desired_layers_ins
        cl_1.regions.useInset(inset=[slice(0,10), slice(0,10)], mode='abs', 
                             expand=True)
        np_test.assert_equal(cl_1.regions.data, desired_layers)

        # test layers w inset
        cl_1_ins.regions.useInset(inset=[slice(0,10), slice(0,10)], mode='abs', 
                                 expand=True)
        np_test.assert_equal(cl_1_ins.regions.data, desired_layers)
        

        # test width and nLayers, mean width mode
        cl_1.makeLayers(widthMode='mean')
        desired = (1.5 * numpy.sqrt(10) + numpy.sqrt(13) + 2.5 * 4) / 5 - 1
        np_test.assert_almost_equal(cl_1.width, desired)
        np_test.assert_equal(cl_1.nLayers, 3)

        # cleft_2
        cl_2 = CleftRegions(image=self.image_2, cleft=self.cleft_2)

        # simple layers
        cl_2.makeLayers(nBoundLayers=2)
        cl_2.regions.useInset(inset=[slice(0,10), slice(0,10)], mode='abs', 
                             expand=True)

        # width
        np_test.assert_almost_equal(cl_2.width, 5.)
        np_test.assert_almost_equal(cl_2.widthVector.data, [6, 0])

        # ids
        np_test.assert_equal(cl_2.getLayerIds(), list(range(1, 10)))
        np_test.assert_equal(cl_2.getCleftLayerIds(), list(range(3, 8)))
        np_test.assert_equal(cl_2.getBound1LayerIds(), list(range(1, 3)))
        np_test.assert_equal(cl_2.getBound2LayerIds(), list(range(8, 10)))
        np_test.assert_equal(cl_2.getBound1LayerIds(thick=1), list(range(2, 3)))
        np_test.assert_equal(cl_2.getBound2LayerIds(thick=2), list(range(8, 10)))
        np_test.assert_equal(cl_2.getBoundLayerIds(thick=[1,2]), [2, 8, 9])
        cl_2.boundThick = 1
        np_test.assert_equal(cl_2.getBound1LayerIds(), list(range(2, 3)))
        np_test.assert_equal(cl_2.getBound2LayerIds(), list(range(8, 9)))

        # layers
        desired = numpy.zeros((10, 10), dtype=int)
        desired[1:, 2:8] = numpy.array(\
            [[1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3, 3],
             [4, 4, 4, 4, 4, 4],
             [5, 5, 5, 5, 5, 5],
             [6, 6, 6, 6, 6, 6],
             [7, 7, 7, 7, 7, 7],
             [8, 8, 8, 8, 8, 8],
             [9, 9, 9, 9, 9, 9]])
        np_test.assert_equal(cl_2.regions.data, desired)

        # parameters
        np_test.assert_equal(cl_2.widthMode, 'median')
        np_test.assert_equal(cl_2.nBoundLayers, 2)
        np_test.assert_equal(cl_2.maxDistance, None)

        # cleft 3
        cl_3 = CleftRegions(image=None, cleft=self.makeCleft3())
        cl_3.makeLayers(nLayers=None, widthMode='mean', maxDistance=2.,
                        nBoundLayers=1, adjust=False, refine=False)
        np_test.assert_equal(cl_3.nLayers, 2)
        np_test.assert_almost_equal(cl_3.width, (numpy.sqrt(10) + 2) / 2. - 1)

        # cleft 3 adjust and refine
        cl_3 = CleftRegions(image=None, cleft=self.makeCleft3())
        cl_3.makeLayers(nLayers=None, widthMode='mean', maxDistance=2.1,
                        nBoundLayers=1, adjust=True, refine=True)
        np_test.assert_almost_equal(cl_3.nLayers, 1)
        
    def testMakeColumns(self):
        """
        Tests makeColumns()
        """

        # simple
        cleft_2 = deepcopy(self.cleft_2)
        cleft_2.data[3:8, 7] = 0        
        cl = CleftRegions(image=self.image_2, cleft=cleft_2)
        cl.makeLayers(nBoundLayers=2)
        cc = cl.makeColumns(bins=[0,1,2,3], ids=None, system='radial', rimId=0)
        cc.regions.useInset(inset=(slice(0,10), slice(0,10)), 
                            mode='abs', useFull=True, expand=True)
        desired = numpy.zeros((10, 10), dtype=int)
        desired[3:8, 2:7] = numpy.array([[3,2,1,2,3],
                                         [3,2,1,2,3],
                                         [3,2,1,2,3],
                                         [3,2,1,2,3],
                                         [3,2,1,2,3]])
        np_test.assert_equal(cc.regions.data, desired)
        np_test.assert_equal(cc.system, 'radial')
        np_test.assert_equal(cc.rimId, 0)
        np_test.assert_equal(cc.metric, 'euclidean')

        # larger, normalized (checks bug fixed in r836)
        cleft_2 = deepcopy(self.cleft_2)
        cleft_2.data[1:3, 1] = 1
        cleft_2.data[3:8, 1] = 3
        cleft_2.data[8:10, 1] = 2
        cl = CleftRegions(image=self.image_2, cleft=cleft_2)
        cl.makeLayers(nBoundLayers=2)
        cc = cl.makeColumns(bins=[0, 0.25, 0.5, 0.75, 1], ids=None, 
                            system='radial', normalize=True, rimId=0)
        cc.regions.useInset(inset=(slice(0,10), slice(0,10)), 
                            mode='abs', useFull=True, expand=True)
        desired = numpy.zeros((10, 10), dtype=int)
        desired[3:8, 1:8] = numpy.array([[4,3,2,1,2,3,4],
                                         [4,3,2,1,2,3,4],
                                         [4,3,2,1,2,3,4],
                                         [4,3,2,1,2,3,4],
                                         [4,3,2,1,2,3,4]])
        np_test.assert_equal(cc.regions.data, desired)
        np_test.assert_equal(cc.system, 'radial')
        np_test.assert_equal(cc.rimId, 0)
        np_test.assert_equal(cc.metric, 'euclidean')

    def testFindDensity(self):
        """
        Tests CleftRegions.findDensity()
        """

        cl_2 = CleftRegions(image=self.image_2, cleft=self.cleft_2)
        cl_2.makeLayers(nBoundLayers=2)

        ########################################
        #
        # simple
        #
        cl_2.findDensity(regions=cl_2.regions, mode='layers')

        # layer density
        desired_mean = common.cleft_layers_density_mean
        np_test.assert_equal(cl_2.regionDensity.mean[1:], desired_mean)
        np_test.assert_equal(cl_2.regionDensity.volume[1:], [6] * 9)
        
        # group density
        np_test.assert_almost_equal(cl_2.groupDensity['cleft'].mean, 5)
        np_test.assert_almost_equal(cl_2.groupDensity['bound_1'].mean, 7.5)
        np_test.assert_almost_equal(cl_2.groupDensity['bound_2'].mean, 9)
        np_test.assert_almost_equal(cl_2.groupDensity['all'].mean, 58 / 9.)
        
        # group ids
        np_test.assert_equal(cl_2.groupIds['cleft'], cl_2.getCleftLayerIds())
        np_test.assert_equal(cl_2.groupIds['bound_2'], cl_2.getBound2LayerIds())
        np_test.assert_equal(cl_2.groupIds['bound_1'], [1,2])
        np_test.assert_equal(cl_2.groupIds['all'], [1,2,3,4,5,6,7,8,9])

        # other attributes
        np_test.assert_equal(cl_2.exclude, [0,0])
        np_test.assert_equal(cl_2.nBoundLayers, 2)
        np_test.assert_equal(cl_2.boundThick, [None, None])

        # minimum density layer
        np_test.assert_almost_equal(
            cl_2.minCleftDensity, cl_2.regionDensity.mean[3])
        np_test.assert_almost_equal(
            cl_2.relativeMinCleftDensity, (3 - 33/4.) / (5 - 33/4.))
        np_test.assert_equal(cl_2.minCleftDensityId, 3)
        np_test.assert_almost_equal(cl_2.minCleftDensityPosition, 0.1)

        #########################################
        #
        # with exclude and boundThick
        #
        cl_2.findDensity(regions=cl_2.regions, mode='layers', 
                         exclude=1, boundThick=1)

        # layer density
        desired_mean = numpy.array([5, 10, 3, 4, 5, 6, 7, 12, 6])
        np_test.assert_equal(cl_2.regionDensity.mean[1:], desired_mean)
        np_test.assert_equal(cl_2.regionDensity.volume[1:], [6] * 9)
        
        # group density
        np_test.assert_almost_equal(cl_2.groupDensity['cleft'].mean, 5)
        np_test.assert_almost_equal(cl_2.groupDensity['bound_1'].mean, 10)
        np_test.assert_almost_equal(cl_2.groupDensity['bound_2'].mean, 12)
        np_test.assert_almost_equal(cl_2.groupDensity['all'].mean, 47/7.)
        
        # group ids
        np_test.assert_equal(cl_2.groupIds['cleft'], 
                             cl_2.getCleftLayerIds(exclude=1))
        np_test.assert_equal(cl_2.groupIds['bound_2'], 
                             cl_2.getBound2LayerIds(thick=1))
        np_test.assert_equal(cl_2.groupIds['bound_1'], [2]) 
        np_test.assert_equal(cl_2.groupIds['all'], [2,3,4,5,6,7,8])

        # other attributes
        np_test.assert_equal(cl_2.exclude, [1,1])
        np_test.assert_equal(cl_2.nBoundLayers, 2)
        np_test.assert_equal(cl_2.boundThick, [1, 1])

        # min density layer
        np_test.assert_almost_equal(cl_2.minCleftDensity, 
                                    cl_2.regionDensity.mean[4])
        np_test.assert_almost_equal(cl_2.relativeMinCleftDensity, 
                                    (4 - 22/2.) / (5 - 22/2.))
        np_test.assert_equal(cl_2.minCleftDensityId, 4)
        np_test.assert_almost_equal(cl_2.minCleftDensityPosition, 0.3)
        #print 'cl_2.relativeMinCleftDensity: ', cl_2.relativeMinCleftDensity
        #print 'cl_2.minCleftDensityId: ', cl_2.minCleftDensityId
        #print 'cl_2.minCleftDensityPosition: ', cl_2.minCleftDensityPosition

        ########################################
        #
        # columns
        #

        cleft_2 = deepcopy(self.cleft_2)
        cleft_2.data[3:8, 7] = 0        
        cl = CleftRegions(image=self.image_2, cleft=cleft_2)
        cl.makeLayers(nBoundLayers=2)

        # layer density
        cc = cl.makeColumns(bins=[0,1,2,3], ids=None, system='radial', rimId=0)
        cc.findDensity(regions=cc.regions, mode='regions')
        desired_mean = common.cleft_layers_density_mean
        np_test.assert_equal(cc.regionDensity.mean[1:], [5, 5, 5])
        np_test.assert_equal(cc.regionDensity.volume[1:], [5, 10, 10])
        
        cc = cl.makeColumns(bins=[0,2,3], ids=None, system='radial', rimId=0)
        cc.findDensity(regions=cc.regions, mode='regions')
        desired_mean = common.cleft_layers_density_mean
        np_test.assert_equal(cc.regionDensity.mean[1:], [5, 5])
        np_test.assert_equal(cc.regionDensity.volume[1:], [15, 10])
        
    def testGetPoints(self):
        """
        Tests getPoints(). More extensive tests in 
        ..segmentation.test.testLabels.
        """

        # make cleft_1 and check it (see testMakeLayers)
        cl_1 = CleftRegions(image=None, cleft=self.cleft_1)
        cl_1.makeLayers()
        cl_1_ins = CleftRegions(image=None, cleft=self.cleft_1_ins)
        cl_1_ins.makeLayers()
        desired_layers_ins = numpy.array(\
            [[0, 1, 2, 3, 0, 0],
             [0, 1, 2, 3, 0, 0],
             [0, 1, 2, 3, 0, 0],
             [0, 0, 1, 2, 3, 0],
             [0, 1, 2, 3, 0, 0],
             [0, 1, 2, 3, 0, 0]])            
        np_test.assert_equal(cl_1.regions.data, desired_layers_ins)

        # mode all
        desired = (numpy.array([1, 2, 3, 4, 5, 6]), 
                   numpy.array([5, 5, 5, 6, 5, 5]))
        np_test.assert_equal(
            cl_1.getPoints(ids=[2], mode='all', format_='numpy'), desired)
        np_test.assert_equal(
            cl_1.getPoints(ids=[2], mode='all', format_='coordinates'), 
            numpy.array(desired).transpose())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCleftRegions)
    unittest.TextTestRunner(verbosity=2).run(suite)
